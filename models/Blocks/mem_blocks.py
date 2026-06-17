import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_out_dim(in_dim, kernel_size, stride=1, padding=0, dilation=1):
    return ((in_dim + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1


def _group_norm_groups(channels, max_groups=8):
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)
    

class mem_block1(nn.Module):
    """with Residual"""
    def __init__(self, cfg, c, t, h, w, memory_size):
        super(mem_block1, self).__init__()
        self.memory_size = memory_size
        self.mem_generator = nn.Sequential(
            nn.Conv3d(c, self.memory_size[1], kernel_size=1, bias=False),
            nn.GroupNorm(
                num_groups=_group_norm_groups(self.memory_size[1]),
                num_channels=self.memory_size[1],
            ),
            nn.SiLU(),
            nn.Flatten(start_dim=3),
            nn.Linear(h * w, self.memory_size[3] * self.memory_size[4]),
            nn.Unflatten(dim=3, unflattened_size=(self.memory_size[3], self.memory_size[4]))
        )
        self.mem_integrator = nn.Sequential(
            nn.Conv3d(self.memory_size[1] * 2, self.memory_size[1], kernel_size=1, bias=False),
            nn.GroupNorm(
                num_groups=_group_norm_groups(self.memory_size[1]),
                num_channels=self.memory_size[1],
            ),
            nn.SiLU(),
        )
        self.mem_gate = nn.Sequential(
            nn.Conv3d(self.memory_size[1] * 2, self.memory_size[1], kernel_size=1),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout3d(p=float(cfg.get("memory_dropout", 0.1)))

        self.register_buffer(
            "memory",
            torch.zeros(1, self.memory_size[1], self.memory_size[2], self.memory_size[3], self.memory_size[4], device=cfg["device"])
        )
    
    def reset_memory(self):
        with torch.no_grad():
            self.memory = self.memory.detach()
            self.memory.zero_()
    
    def forward(self, x):
        x = self.mem_generator(x)
        x = x.repeat(1, 1, self.memory_size[2], 1, 1)
        fusion = torch.cat([self.memory, x], dim=1)
        proposal = self.mem_integrator(fusion)
        gate = self.mem_gate(fusion)
        updated = self.memory * (1.0 - gate) + proposal * gate
        self.memory = self.dropout(updated)
        return self.memory


class mem_block2(nn.Module):
    """with Residual"""
    def __init__(self, cfg, c, t, h, w, memory_size):
        super(mem_block2, self).__init__()
        self.memory_size = memory_size
        self.mem_generator = nn.Sequential(
            nn.Conv3d(c, self.memory_size[2], kernel_size=(t, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
            nn.GroupNorm(
                num_groups=_group_norm_groups(self.memory_size[2]),
                num_channels=self.memory_size[2],
            ),
            nn.SiLU(),
            Permute([0, 2, 1, 3, 4]),
            nn.Conv3d(1, self.memory_size[1], kernel_size=1, bias=False),
            nn.GroupNorm(
                num_groups=_group_norm_groups(self.memory_size[1]),
                num_channels=self.memory_size[1],
            ),
            nn.SiLU(),
        )
        self.mem_integrator = nn.Sequential(
            nn.Conv3d(self.memory_size[1] * 2, self.memory_size[1], kernel_size=1, bias=False),
            nn.GroupNorm(
                num_groups=_group_norm_groups(self.memory_size[1]),
                num_channels=self.memory_size[1],
            ),
            nn.SiLU(),
        )
        self.mem_gate = nn.Sequential(
            nn.Conv3d(self.memory_size[1] * 2, self.memory_size[1], kernel_size=1),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout3d(p=float(cfg.get("memory_dropout", 0.1)))

        self.register_buffer(
            "memory",
            torch.zeros(1, self.memory_size[1], self.memory_size[2], self.memory_size[3], self.memory_size[4], device=cfg["device"])
        )
    
    def reset_memory(self):
        with torch.no_grad():
            self.memory = self.memory.detach()
            self.memory.zero_()
    
    def forward(self, x):
        current = self.mem_generator(x)
        fusion = torch.cat([self.memory, current], dim=1)
        proposal = self.mem_integrator(fusion)
        gate = self.mem_gate(fusion)
        updated = self.memory * (1.0 - gate) + proposal * gate
        self.memory = self.dropout(updated)
        return self.memory


class mem_block3(nn.Module):
    """with Residual"""
    def __init__(self, cfg, c, t, h, w, memory_size):
        super(mem_block3, self).__init__()
        self.memory_size = memory_size
        self.mem_generator = nn.Sequential(
            nn.Conv3d(c, self.memory_size[2], kernel_size=(t, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
            nn.GroupNorm(
                num_groups=_group_norm_groups(self.memory_size[2]),
                num_channels=self.memory_size[2],
            ),
            nn.SiLU(),
            Permute([0, 2, 1, 3, 4]),
            nn.Conv3d(1, self.memory_size[1], kernel_size=1, bias=False),
            nn.GroupNorm(
                num_groups=_group_norm_groups(self.memory_size[1]),
                num_channels=self.memory_size[1],
            ),
            nn.SiLU(),
            nn.Flatten(start_dim=3),
            nn.Linear(h * w, self.memory_size[3] * self.memory_size[4]),
            nn.Unflatten(dim=3, unflattened_size=(self.memory_size[3], self.memory_size[4])),
            nn.SiLU()
        )
        self.mem_integrator = nn.Sequential(
            nn.Conv3d(self.memory_size[1] * 2, self.memory_size[1], kernel_size=1, bias=False),
            nn.GroupNorm(
                num_groups=_group_norm_groups(self.memory_size[1]),
                num_channels=self.memory_size[1],
            ),
            nn.SiLU(),
        )
        self.mem_gate = nn.Sequential(
            nn.Conv3d(self.memory_size[1] * 2, self.memory_size[1], kernel_size=1),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout3d(p=float(cfg.get("memory_dropout", 0.1)))

        self.register_buffer(
            "memory",
            torch.zeros(1, self.memory_size[1], self.memory_size[2], self.memory_size[3], self.memory_size[4], device=cfg["device"])
        )
    
    def reset_memory(self):
        with torch.no_grad():
            self.memory = self.memory.detach()
            self.memory.zero_()
    
    def forward(self, x):
        current = self.mem_generator(x)
        fusion = torch.cat([self.memory, current], dim=1)
        proposal = self.mem_integrator(fusion)
        gate = self.mem_gate(fusion)
        updated = self.memory * (1.0 - gate) + proposal * gate
        self.memory = self.dropout(updated)
        return self.memory


class deep_mem_block(nn.Module):
    """with Residual"""
    def __init__(self, cfg, c, t, h, w, memory_size):
        super(deep_mem_block, self).__init__()
        self.memory_size = memory_size
        self.mem_integrator = nn.Sequential(
            nn.Conv3d(self.memory_size[1] * 2, self.memory_size[1], kernel_size=1, bias=False),
            nn.GroupNorm(
                num_groups=_group_norm_groups(self.memory_size[1]),
                num_channels=self.memory_size[1],
            ),
            nn.SiLU(),
        )
        self.mem_gate = nn.Sequential(
            nn.Conv3d(self.memory_size[1] * 2, self.memory_size[1], kernel_size=1),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout3d(p=float(cfg.get("memory_dropout", 0.1)))

        self.register_buffer(
            "memory",
            torch.zeros(1, self.memory_size[1], self.memory_size[2], self.memory_size[3], self.memory_size[4], device=cfg["device"])
        )
    
    def reset_memory(self):
        with torch.no_grad():
            self.memory = self.memory.detach()
            self.memory.zero_()
    
    def forward(self, x):
        fusion = torch.cat([self.memory, x], dim=1)
        proposal = self.mem_integrator(fusion)
        gate = self.mem_gate(fusion)
        updated = self.memory * (1.0 - gate) + proposal * gate
        self.memory = self.dropout(updated)
        return self.memory