import torch
import torch.nn as nn
from models import Blocks


def _group_norm_groups(channels, max_groups=8):
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


def _to_odd_kernel(k, default=3):
    try:
        k = int(k)
    except (TypeError, ValueError):
        k = int(default)
    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1
    return k


class _PreHeadMemoryRefine(nn.Module):
    """Lightweight residual refine block before detection head."""

    def __init__(self, in_channels, out_channels, temporal_kernel=3, dropout=0.0):
        super().__init__()
        k_t = _to_odd_kernel(temporal_kernel, default=3)
        p_t = k_t // 2
        self.block = nn.Sequential(
            # depthwise temporal filtering
            nn.Conv3d(
                in_channels,
                in_channels,
                kernel_size=(k_t, 1, 1),
                padding=(p_t, 0, 0),
                # groups=out_channels,
                bias=False,
            ),
            nn.GroupNorm(_group_norm_groups(in_channels), in_channels),
            nn.SiLU(),
            # pointwise channel mixing
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_norm_groups(out_channels), out_channels),
            nn.SiLU(),
            nn.Dropout3d(p=float(dropout)),
        )
        self.gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x):
        refined = self.block(x)
        gate = torch.sigmoid(self.gate).to(dtype=x.dtype)
        # return x + gate * (refined - x)
        return refined



class mem_tad(nn.Module):
    def __init__(self, cfg):
        super(mem_tad, self).__init__()
        self.memory_type = cfg["memory_size"]
        self.feature_size = cfg["feature_size"]
        if not isinstance(self.feature_size, (list, tuple)) or len(self.feature_size) != 4:
            raise ValueError(
                f"cfg['feature_size'] must be [C, T, H, W], got {self.feature_size}"
            )
        self.feature_channels = int(self.feature_size[0])
        self.feature_t = int(self.feature_size[1])
        self.feature_h = int(self.feature_size[2])
        self.feature_w = int(self.feature_size[3])
        self.lr = cfg["lr"]
        self.vfn = cfg["view_frames_num"]
        self.deep_mem_enabled = bool(cfg["deep_mem"])

        if cfg["memory_size"] == 'l':
            self.memory_size = [1, 16, 128, 17, 34] if cfg['mem_block'] == 'mem_block1' else [1, 16, 128, self.feature_h, self.feature_w]
        elif cfg["memory_size"] == 'm':
            self.memory_size = [1, 16, 64, 17, 34] if cfg['mem_block'] == 'mem_block1' else [1, 16, 64, self.feature_h, self.feature_w]
        else:
            self.memory_size = [1, 8, 64, 17, 34] if cfg['mem_block'] == 'mem_block1' else [1, 8, 64, self.feature_h, self.feature_w]

        # memory block
        self.mem = Blocks.__dict__[cfg['mem_block']](
            cfg,
            self.feature_channels,
            self.feature_t,
            self.feature_h,
            self.feature_w,
            self.memory_size,
        )
        if self.deep_mem_enabled:
            self.deep_mem = Blocks.__dict__['deep_mem_block'](cfg, self.memory_size[1], self.memory_size[2], self.memory_size[3], self.memory_size[4], self.memory_size)

        pre_head_in_channels = self.memory_size[1] * (2 if self.deep_mem_enabled else 1)
        self.pre_head_memory_refine_enabled = bool(cfg.get("pre_head_memory_refine_enabled", False))
        refine_dropout = float(cfg.get("pre_head_memory_refine_dropout", cfg.get("head_dropout", 0.2)))
        refine_kernel_t = _to_odd_kernel(cfg.get("pre_head_memory_refine_kernel_t", 3), default=3)

        if self.pre_head_memory_refine_enabled:
            self.pre_head_memory_refine = _PreHeadMemoryRefine(
                in_channels=pre_head_in_channels,
                out_channels=pre_head_in_channels*2,
                temporal_kernel=refine_kernel_t,
                dropout=refine_dropout,
            )
        else:
            self.pre_head_memory_refine = nn.Identity()

        head_channels = int(cfg.get("head_channels", pre_head_in_channels))
        head_dropout = float(cfg.get("head_dropout", 0.2))
        head_token_dropout = float(cfg.get("head_token_dropout", head_dropout))
        hidden_dim = int(cfg.get("det_head_hidden_dim", 256))
        self.max_detection_num = int(cfg["max_detection_num"])
        self.class_num = int(cfg["class_num"])

        self.head_stem = nn.Sequential(
            nn.Conv3d(pre_head_in_channels*2, head_channels, kernel_size=3, padding=1, bias=False),  # pre_head_in_channels
            # nn.Conv3d(pre_head_in_channels, head_channels, kernel_size=3, padding=1, bias=False),  # pre_head_in_channels
            nn.GroupNorm(num_groups=_group_norm_groups(head_channels), num_channels=head_channels),
            nn.SiLU(),
            nn.Dropout3d(p=head_dropout),
            nn.Conv3d(head_channels, head_channels, kernel_size=3, padding=1, stride=2, bias=False),
            nn.GroupNorm(num_groups=_group_norm_groups(head_channels), num_channels=head_channels),
            nn.SiLU(),
            nn.Dropout3d(p=head_dropout),
        )
        self.query_proj = nn.Conv3d(head_channels, self.max_detection_num, kernel_size=1)

        self.shared_token_mlp = nn.Sequential(
            nn.Linear((self.memory_size[2] // 2) * ((self.memory_size[3]+1) // 2) * ((self.memory_size[4]+1) // 2), hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=head_dropout),
        )

        self.reg_token_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=head_dropout),
        )

        self.cls_token_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=max(0.05, head_dropout * 0.5)),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.token_dropout = nn.Dropout(p=head_token_dropout)

        # decoupled detection heads: logits for conf/class and raw regressions for loc
        self.conf_head = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim*4),
            nn.SiLU(),
            nn.Linear(hidden_dim*4, hidden_dim*2),
            nn.SiLU(),
            nn.Linear(hidden_dim*2, 1),
        )
        self.loc_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.SiLU(),
            nn.Linear(hidden_dim*2, 2)
        )
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.SiLU(),
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.SiLU(),
            nn.Linear(hidden_dim*2, self.class_num),
        )

        self.cls_feat_proj = nn.Conv3d(
            head_channels,
            hidden_dim,
            kernel_size=1,
            bias=False
        )

        self.cls_query_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )

        self.cls_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=max(0.05, head_dropout * 0.5),
            batch_first=True
        )

        self.cls_fuse = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=max(0.05, head_dropout * 0.5))
        )
        

    def reset_memory(self):
        self.mem.reset_memory()
        if self.deep_mem_enabled:
            self.deep_mem.reset_memory()


    def forward(self, x, decode=False):
        if x.dim() != 5:
            raise ValueError(
                f"mem_tad expects feature input shape [B, C, T, H, W], got {tuple(x.shape)}"
            )
        if x.size(1) != self.feature_channels or x.size(2) != self.feature_t or x.size(3) != self.feature_h or x.size(4) != self.feature_w:
            raise ValueError(
                "mem_tad feature shape mismatch: "
                f"expected [B, {self.feature_channels}, {self.feature_t}, {self.feature_h}, {self.feature_w}], "
                f"got {tuple(x.shape)}"
            )

        pre_mem = self.mem.memory
        x = self.mem(x)
        if self.deep_mem_enabled:
            deep_mem_feat = self.deep_mem(x - pre_mem)
            x = torch.cat([x, deep_mem_feat], dim=1)

        if x.dim() == 4:
            x = x.unsqueeze(2)
        
        if decode:
            x = self.pre_head_memory_refine(x)
            feat = self.head_stem(x)

            raw_tokens = self.query_proj(feat).flatten(start_dim=2)

            shared_tokens = self.shared_token_mlp(raw_tokens)
            reg_tokens = self.reg_token_mlp(shared_tokens)
            cls_tokens = self.cls_token_mlp(shared_tokens)

            loc = self.loc_head(reg_tokens)

            # head_stem feature: [B, C, T, H, W]
            dense_feat = self.cls_feat_proj(feat)          # [B, hidden_dim, T, H, W]
            # ------------------------------------------------
            dense_feat = dense_feat.flatten(2).transpose(1, 2)
            # [B, T*H*W, hidden_dim]


            cls_query = self.cls_query_proj(reg_tokens)
            # [B, max_detection_num, hidden_dim]

            cls_context, _ = self.cls_cross_attn(
                query=cls_query,
                key=dense_feat,
                value=dense_feat
            )

            cls_tokens = self.cls_fuse(cls_query + cls_context)
            cls_logits = self.cls_head(cls_tokens)  # cls_tokens cls_context reg_tokens

            conf_tokens = torch.cat([reg_tokens, cls_context], dim=-1)
            conf_logits = self.conf_head(conf_tokens)

            x = torch.cat([conf_logits, loc, cls_logits], dim=-1)

        return x
    

