import torch
import torch.nn as nn
import torch.nn.functional as F


def _odd_kernel(k):
    k = int(k)
    if k < 1:
        return 1
    if k % 2 == 0:
        return k + 1
    return k


def _scale_channels(channels, width_mult, min_channels=16):
    scaled = int(round(float(channels) * float(width_mult)))
    return max(min_channels, scaled)


def _group_norm_groups(channels, max_groups=8):
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


def _norm3d(channels):
    return nn.GroupNorm(
        num_groups=_group_norm_groups(channels),
        num_channels=channels,
    )


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, temporal_kernel=3):
        super(BasicBlock3D, self).__init__()
        tk = _odd_kernel(temporal_kernel)
        tp = tk // 2

        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(tk, 3, 3),
            stride=(1, stride, stride),
            padding=(tp, 1, 1),
            bias=False,
        )
        self.bn1 = _norm3d(out_channels)
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(tk, 3, 3),
            stride=1,
            padding=(tp, 1, 1),
            bias=False,
        )
        self.bn2 = _norm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

# 定义 Bottleneck 块
class Bottleneck3D(nn.Module):
    # 每个 Bottleneck 块输出通道数是扩展因子的 4 倍
    expansion = 4 

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, temporal_kernel=5):
        super(Bottleneck3D, self).__init__()
        tk = _odd_kernel(temporal_kernel)
        tp = tk // 2
        
        # 1x1 卷积：压缩通道数，降低计算量
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = _norm3d(out_channels)
        
        # 3x3 卷积：进行主要的特征提取，stride 控制下采样
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(tk, 3, 3), stride=(1, stride, stride), padding=(tp, 1, 1), bias=False)
        self.bn2 = _norm3d(out_channels)
        
        # 1x1 卷积：恢复通道数 (乘以 expansion)
        self.conv3 = nn.Conv3d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = _norm3d(out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x # 保存原始输入

        # 主干路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # 如果输入输出维度不一致，需要用 1x1 卷积调整残差边
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差连接：将原始输入加到主干路径的输出上
        out += identity
        out = self.relu(out)

        return out
    

class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        vfn=1,
        width_mult=1.0,
        temporal_kernel=5,
        memory_channels=0,
        memory_fusion_enabled=False,
        memory_fusion_layers=None,
        memory_fusion_init_gate=-2.0,
    ):
        super(ResNet, self).__init__()
        self.temporal_kernel = _odd_kernel(temporal_kernel)
        c1 = _scale_channels(64, width_mult)
        c2 = _scale_channels(128, width_mult)
        c3 = _scale_channels(256, width_mult)
        c4 = _scale_channels(512, width_mult)
        self.block_expansion = block.expansion
        self.memory_fusion_enabled = bool(memory_fusion_enabled)
        
        # 初始卷积层 (ImageNet 输入 224x224 -> 112x112)
        self.in_channels = c1
        self.conv1 = nn.Conv3d(3, c1, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.bn1 = _norm3d(c1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        # 四个主要阶段
        # 每个阶段的第一个块负责下采样 (stride=2)
        self.layer1 = self._make_layer(block, c1, layers[0], stride=1)
        self.layer2 = self._make_layer(block, c2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, c3, layers[2], stride=2)
        self.layer4 = self._make_layer(block, c4, layers[3], stride=2)

        self._layer_input_channels = {
            "layer1": c1,
            "layer2": c1 * self.block_expansion,
            "layer3": c2 * self.block_expansion,
            "layer4": c3 * self.block_expansion,
        }
        self._valid_fusion_layers = ["layer1", "layer2", "layer3", "layer4"]
        requested_layers = memory_fusion_layers or ["layer3", "layer4"]
        self.memory_fusion_layers = [
            name for name in requested_layers if name in self._valid_fusion_layers
        ]
        if self.memory_fusion_enabled and memory_channels > 0 and self.memory_fusion_layers:
            self.mem_fuse_proj = nn.ModuleDict(
                {
                    name: nn.Conv3d(
                        int(memory_channels),
                        int(self._layer_input_channels[name]),
                        kernel_size=1,
                        bias=False,
                    )
                    for name in self.memory_fusion_layers
                }
            )
            self.mem_fuse_gate = nn.ParameterDict(
                {
                    name: nn.Parameter(torch.tensor(float(memory_fusion_init_gate)))
                    for name in self.memory_fusion_layers
                }
            )
        else:
            self.mem_fuse_proj = nn.ModuleDict()
            self.mem_fuse_gate = nn.ParameterDict()
        
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        
        # 如果步长大于1，或者输入通道数不等于输出通道数 * 扩展因子
        # 则需要通过 1x1 卷积调整残差边的维度
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=(1, stride, stride),
                    bias=False,
                ),
                _norm3d(out_channels * block.expansion),
            )

        layers = []
        # 第一个块负责下采样
        layers.append(
            block(
                self.in_channels,
                out_channels,
                stride,
                downsample,
                temporal_kernel=self.temporal_kernel,
            )
        )
        self.in_channels = out_channels * block.expansion
        
        # 剩下的块 stride=1，维度保持不变
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_channels,
                    out_channels,
                    temporal_kernel=self.temporal_kernel,
                )
            )

        return nn.Sequential(*layers)

    def _fuse_memory(self, x, mem, layer_name):
        if not self.memory_fusion_enabled:
            return x
        if layer_name not in self.mem_fuse_proj:
            return x
        if mem is None or (not torch.is_tensor(mem)) or mem.dim() != 5:
            return x

        m = mem
        if m.size(0) != x.size(0):
            if m.size(0) == 1:
                m = m.expand(x.size(0), -1, -1, -1, -1)
            else:
                return x

        if m.shape[2:] != x.shape[2:]:
            m = F.interpolate(m, size=x.shape[2:], mode="trilinear", align_corners=False)

        m = m.to(device=x.device, dtype=x.dtype)
        m = self.mem_fuse_proj[layer_name](m)
        gate = torch.sigmoid(self.mem_fuse_gate[layer_name]).to(dtype=x.dtype)
        return x + gate * m

    def forward(self, x, mem):
        # 初始处理
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # 四个阶段
        x = self._fuse_memory(x, mem, "layer1")
        x = self.layer1(x)
        x = self._fuse_memory(x, mem, "layer2")
        x = self.layer2(x)
        x = self._fuse_memory(x, mem, "layer3")
        x = self.layer3(x)
        x = self._fuse_memory(x, mem, "layer4")
        x = self.layer4(x)

        return x

# --- 构建 ResNet ---
def built_resnet(
    type='50',
    vfn=1,
    width_mult=1.0,
    temporal_kernel=5,
    memory_channels=0,
    memory_fusion_enabled=False,
    memory_fusion_layers=None,
    memory_fusion_init_gate=-2.0,
):
    if type == '10':
        return ResNet(
            BasicBlock3D,
            [1, 1, 1, 1],
            vfn,
            width_mult=width_mult,
            temporal_kernel=temporal_kernel,
            memory_channels=memory_channels,
            memory_fusion_enabled=memory_fusion_enabled,
            memory_fusion_layers=memory_fusion_layers,
            memory_fusion_init_gate=memory_fusion_init_gate,
        )
    elif type == '18':
        return ResNet(
            BasicBlock3D,
            [2, 2, 2, 2],
            vfn,
            width_mult=width_mult,
            temporal_kernel=temporal_kernel,
            memory_channels=memory_channels,
            memory_fusion_enabled=memory_fusion_enabled,
            memory_fusion_layers=memory_fusion_layers,
            memory_fusion_init_gate=memory_fusion_init_gate,
        )
    elif type == '34':
        return ResNet(
            BasicBlock3D,
            [3, 4, 6, 3],
            vfn,
            width_mult=width_mult,
            temporal_kernel=temporal_kernel,
            memory_channels=memory_channels,
            memory_fusion_enabled=memory_fusion_enabled,
            memory_fusion_layers=memory_fusion_layers,
            memory_fusion_init_gate=memory_fusion_init_gate,
        )
    elif type == '50':
        return ResNet(
            Bottleneck3D,
            [3, 4, 6, 3],
            vfn,
            width_mult=width_mult,
            temporal_kernel=temporal_kernel,
            memory_channels=memory_channels,
            memory_fusion_enabled=memory_fusion_enabled,
            memory_fusion_layers=memory_fusion_layers,
            memory_fusion_init_gate=memory_fusion_init_gate,
        )
    elif type == '101':
        return ResNet(
            Bottleneck3D,
            [3, 4, 23, 3],
            vfn,
            width_mult=width_mult,
            temporal_kernel=temporal_kernel,
            memory_channels=memory_channels,
            memory_fusion_enabled=memory_fusion_enabled,
            memory_fusion_layers=memory_fusion_layers,
            memory_fusion_init_gate=memory_fusion_init_gate,
        )
    else:
        raise ValueError("Unsupported ResNet type")

# --- 测试 ---
if __name__ == "__main__":
    # 统一模型和输入所在设备，避免 CPU/GPU 张量不匹配
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = built_resnet('50', vfn=1).to(device)
    model.eval()
    # print(model)
    
    # 模拟输入 [B, C, T, H, W]
    dummy_input = torch.randn(1, 3, 100, 720, 1080, device=device)
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"使用设备: {device}")
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")