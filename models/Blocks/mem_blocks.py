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


class _GatedMemoryDiagnostics:
    def _setup_gated_memory(self, cfg, gate_bias_key, default_gate_bias):
        gate_bias = float(cfg.get(gate_bias_key, default_gate_bias))
        nn.init.constant_(self.mem_gate[0].bias, gate_bias)
        self.dropout = nn.Dropout3d(
            p=float(cfg.get("memory_proposal_dropout", cfg.get("memory_dropout", 0.0)))
        )
        self.memory_diagnostics_enabled = bool(cfg.get("memory_diagnostics_enabled", False))
        self.memory_diagnostics_stride = max(
            int(cfg.get("memory_diagnostics_stride", 4)),
            1,
        )
        device = self.memory.device
        for name in (
            "diag_gate_sum",
            "diag_gate_sq_sum",
            "diag_gate_count",
            "diag_change_sum",
            "diag_retention_sum",
            "diag_state_count",
        ):
            self.register_buffer(name, torch.zeros((), device=device), persistent=False)

    def reset_diagnostics(self):
        for name in (
            "diag_gate_sum",
            "diag_gate_sq_sum",
            "diag_gate_count",
            "diag_change_sum",
            "diag_retention_sum",
            "diag_state_count",
        ):
            getattr(self, name).zero_()

    def _record_diagnostics(self, gate, memory, updated, active_mask=None):
        if not self.memory_diagnostics_enabled:
            return
        with torch.no_grad():
            if active_mask is not None:
                active = active_mask.to(device=gate.device, dtype=torch.bool).view(-1)
                gate = gate[active]
                memory = memory[active]
                updated = updated[active]

            stride = self.memory_diagnostics_stride
            gate = gate[..., ::stride, ::stride, ::stride]
            memory = memory[..., ::stride, ::stride, ::stride]
            updated = updated[..., ::stride, ::stride, ::stride]

            gate_float = gate.detach().float()
            self.diag_gate_sum.add_(gate_float.sum())
            self.diag_gate_sq_sum.add_(gate_float.square().sum())
            self.diag_gate_count.add_(gate_float.numel())

            memory_flat = memory.detach().float().flatten(start_dim=1)
            updated_flat = updated.detach().float().flatten(start_dim=1)
            memory_norm = torch.linalg.vector_norm(memory_flat, dim=1)
            updated_norm = torch.linalg.vector_norm(updated_flat, dim=1)
            valid = (memory_norm > 1e-6).to(dtype=memory_norm.dtype)
            delta_norm = torch.linalg.vector_norm(
                updated_flat - memory_flat,
                dim=1,
            )
            change_ratio = delta_norm / (
                memory_norm + updated_norm + 1e-6
            )
            retention = F.cosine_similarity(
                memory_flat,
                updated_flat,
                dim=1,
                eps=1e-6,
            )
            self.diag_change_sum.add_((change_ratio * valid).sum())
            self.diag_retention_sum.add_((retention * valid).sum())
            self.diag_state_count.add_(valid.sum())

    def get_diagnostics(self, reset=False):
        gate_count = float(self.diag_gate_count.item())
        state_count = float(self.diag_state_count.item())
        if gate_count > 0:
            gate_mean = float((self.diag_gate_sum / gate_count).item())
            gate_var = max(
                float((self.diag_gate_sq_sum / gate_count).item()) - gate_mean ** 2,
                0.0,
            )
            gate_std = gate_var ** 0.5
        else:
            gate_mean = 0.0
            gate_std = 0.0
        diagnostics = {
            "gate_mean": gate_mean,
            "gate_std": gate_std,
            "state_change_ratio": (
                float((self.diag_change_sum / state_count).item())
                if state_count > 0 else 0.0
            ),
            "state_retention": (
                float((self.diag_retention_sum / state_count).item())
                if state_count > 0 else 0.0
            ),
        }
        if reset:
            self.reset_diagnostics()
        return diagnostics

    def _update_memory(self, memory, proposal, gate, active_mask=None):
        # Regularize the proposal branch; never apply dropout to the persistent state.
        proposal = self.dropout(proposal)
        updated = memory + gate * (proposal - memory)
        if active_mask is not None:
            active = active_mask.to(
                device=updated.device,
                dtype=torch.bool,
            ).view(-1, 1, 1, 1, 1)
            updated = torch.where(active, updated, memory)
        self._record_diagnostics(gate, memory, updated, active_mask=active_mask)
        return updated


class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)
    

class mem_block1(nn.Module, _GatedMemoryDiagnostics):
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
        self.register_buffer(
            "memory",
            torch.zeros(1, self.memory_size[1], self.memory_size[2], self.memory_size[3], self.memory_size[4], device=cfg["device"])
        )
        self._setup_gated_memory(cfg, "shallow_gate_init_bias", 0.0)

    def init_memory(self, batch_size, device=None, dtype=None):
        return self.memory.new_zeros(
            (int(batch_size), self.memory_size[1], self.memory_size[2], self.memory_size[3], self.memory_size[4]),
            device=device or self.memory.device,
            dtype=dtype or self.memory.dtype,
        )
    
    def reset_memory(self):
        with torch.no_grad():
            self.memory = self.memory.detach()
            self.memory.zero_()
    
    def forward(self, x, memory=None, active_mask=None):
        x = self.mem_generator(x)
        x = x.repeat(1, 1, self.memory_size[2], 1, 1)
        explicit_memory = memory is not None
        if memory is None:
            if self.memory.size(0) != x.size(0):
                memory = self.init_memory(x.size(0), x.device, x.dtype)
            else:
                memory = self.memory.to(device=x.device, dtype=x.dtype)
        fusion = torch.cat([memory, x], dim=1)
        proposal = self.mem_integrator(fusion)
        gate = self.mem_gate(fusion)
        updated = self._update_memory(memory, proposal, gate, active_mask=active_mask)
        if not explicit_memory:
            self.memory = updated
        return updated


class mem_block4(nn.Module, _GatedMemoryDiagnostics):
    """mem_block1 with gated learnable temporal expansion.

    The initial behavior is intentionally close to mem_block1: the generated
    proposal is still repeated over temporal slots, then a small learnable
    temporal modulation is added through a near-zero gate.
    """
    def __init__(self, cfg, c, t, h, w, memory_size):
        super(mem_block4, self).__init__()
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
        self.temporal_scale = nn.Parameter(torch.zeros(
            1,
            self.memory_size[1],
            self.memory_size[2],
            1,
            1,
        ))
        self.temporal_bias = nn.Parameter(torch.empty(
            1,
            self.memory_size[1],
            self.memory_size[2],
            1,
            1,
        ))
        nn.init.normal_(
            self.temporal_bias,
            std=float(cfg.get("mem_block4_temporal_bias_init_std", 0.02)),
        )
        self.temporal_expansion_gate = nn.Parameter(torch.tensor(
            float(cfg.get("mem_block4_temporal_gate_init", -5.0))
        ))
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
        self.register_buffer(
            "memory",
            torch.zeros(1, self.memory_size[1], self.memory_size[2], self.memory_size[3], self.memory_size[4], device=cfg["device"])
        )
        self._setup_gated_memory(cfg, "shallow_gate_init_bias", 0.0)

    def init_memory(self, batch_size, device=None, dtype=None):
        return self.memory.new_zeros(
            (int(batch_size), self.memory_size[1], self.memory_size[2], self.memory_size[3], self.memory_size[4]),
            device=device or self.memory.device,
            dtype=dtype or self.memory.dtype,
        )
    
    def reset_memory(self):
        with torch.no_grad():
            self.memory = self.memory.detach()
            self.memory.zero_()

    def _temporal_expand(self, x):
        base = x.repeat(1, 1, self.memory_size[2], 1, 1)
        alpha = torch.sigmoid(self.temporal_expansion_gate).to(
            device=base.device,
            dtype=base.dtype,
        )
        scale = self.temporal_scale.to(device=base.device, dtype=base.dtype)
        bias = self.temporal_bias.to(device=base.device, dtype=base.dtype)
        return base * (1.0 + alpha * scale) + alpha * bias
    
    def forward(self, x, memory=None, active_mask=None):
        x = self.mem_generator(x)
        x = self._temporal_expand(x)
        explicit_memory = memory is not None
        if memory is None:
            if self.memory.size(0) != x.size(0):
                memory = self.init_memory(x.size(0), x.device, x.dtype)
            else:
                memory = self.memory.to(device=x.device, dtype=x.dtype)
        fusion = torch.cat([memory, x], dim=1)
        proposal = self.mem_integrator(fusion)
        gate = self.mem_gate(fusion)
        updated = self._update_memory(memory, proposal, gate, active_mask=active_mask)
        if not explicit_memory:
            self.memory = updated
        return updated


class mem_block2(nn.Module, _GatedMemoryDiagnostics):
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
        self.register_buffer(
            "memory",
            torch.zeros(1, self.memory_size[1], self.memory_size[2], self.memory_size[3], self.memory_size[4], device=cfg["device"])
        )
        self._setup_gated_memory(cfg, "shallow_gate_init_bias", 0.0)

    def init_memory(self, batch_size, device=None, dtype=None):
        return self.memory.new_zeros(
            (int(batch_size), self.memory_size[1], self.memory_size[2], self.memory_size[3], self.memory_size[4]),
            device=device or self.memory.device,
            dtype=dtype or self.memory.dtype,
        )
    
    def reset_memory(self):
        with torch.no_grad():
            self.memory = self.memory.detach()
            self.memory.zero_()
    
    def forward(self, x, memory=None, active_mask=None):
        current = self.mem_generator(x)
        explicit_memory = memory is not None
        if memory is None:
            if self.memory.size(0) != x.size(0):
                memory = self.init_memory(x.size(0), x.device, x.dtype)
            else:
                memory = self.memory.to(device=x.device, dtype=x.dtype)
        fusion = torch.cat([memory, current], dim=1)
        proposal = self.mem_integrator(fusion)
        gate = self.mem_gate(fusion)
        updated = self._update_memory(memory, proposal, gate, active_mask=active_mask)
        if not explicit_memory:
            self.memory = updated
        return updated


class convgru_mem_block(nn.Module, _GatedMemoryDiagnostics):
    """Single-state ConvGRU baseline with the mem_block2 input projection."""

    def __init__(self, cfg, c, t, h, w, memory_size):
        super().__init__()
        self.memory_size = memory_size
        self.mem_generator = nn.Sequential(
            nn.Conv3d(
                c,
                self.memory_size[2],
                kernel_size=(t, 1, 1),
                stride=(1, 1, 1),
                padding=(0, 0, 0),
            ),
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
        fusion_channels = self.memory_size[1] * 2
        self.mem_gate = nn.Sequential(
            nn.Conv3d(fusion_channels, self.memory_size[1], kernel_size=1),
            nn.Sigmoid(),
        )
        self.reset_gate = nn.Sequential(
            nn.Conv3d(fusion_channels, self.memory_size[1], kernel_size=1),
            nn.Sigmoid(),
        )
        self.candidate_head = nn.Sequential(
            nn.Conv3d(fusion_channels, self.memory_size[1], kernel_size=1, bias=False),
            nn.GroupNorm(
                num_groups=_group_norm_groups(self.memory_size[1]),
                num_channels=self.memory_size[1],
            ),
            nn.Tanh(),
        )
        self.register_buffer(
            "memory",
            torch.zeros(
                1,
                self.memory_size[1],
                self.memory_size[2],
                self.memory_size[3],
                self.memory_size[4],
                device=cfg["device"],
            ),
        )
        self._setup_gated_memory(cfg, "convgru_update_gate_init_bias", 0.0)
        nn.init.constant_(
            self.reset_gate[0].bias,
            float(cfg.get("convgru_reset_gate_init_bias", 0.0)),
        )

    def init_memory(self, batch_size, device=None, dtype=None):
        return self.memory.new_zeros(
            (
                int(batch_size),
                self.memory_size[1],
                self.memory_size[2],
                self.memory_size[3],
                self.memory_size[4],
            ),
            device=device or self.memory.device,
            dtype=dtype or self.memory.dtype,
        )

    def reset_memory(self):
        with torch.no_grad():
            self.memory = self.memory.detach()
            self.memory.zero_()

    def forward(self, x, memory=None, active_mask=None):
        current = self.mem_generator(x)
        explicit_memory = memory is not None
        if memory is None:
            if self.memory.size(0) != current.size(0):
                memory = self.init_memory(current.size(0), current.device, current.dtype)
            else:
                memory = self.memory.to(device=current.device, dtype=current.dtype)

        fusion = torch.cat([memory, current], dim=1)
        update = self.mem_gate(fusion)
        reset = self.reset_gate(fusion)
        candidate = self.candidate_head(
            torch.cat([reset * memory, current], dim=1)
        )
        updated = self._update_memory(
            memory,
            candidate,
            update,
            active_mask=active_mask,
        )
        if not explicit_memory:
            self.memory = updated
        return updated


class fifo_mem_block(nn.Module):
    """Non-recurrent FIFO history baseline with the same memory tensor shape."""

    def __init__(self, cfg, c, t, h, w, memory_size):
        super().__init__()
        self.memory_size = memory_size
        self.mem_generator = nn.Sequential(
            nn.Conv3d(
                c,
                self.memory_size[1],
                kernel_size=(t, 1, 1),
                stride=(1, 1, 1),
                padding=(0, 0, 0),
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=_group_norm_groups(self.memory_size[1]),
                num_channels=self.memory_size[1],
            ),
            nn.SiLU(),
        )
        self.register_buffer(
            "memory",
            torch.zeros(
                1,
                self.memory_size[1],
                self.memory_size[2],
                self.memory_size[3],
                self.memory_size[4],
                device=cfg["device"],
            ),
        )
        self.memory_diagnostics_enabled = bool(
            cfg.get("memory_diagnostics_enabled", False)
        )
        self.memory_diagnostics_stride = max(
            int(cfg.get("memory_diagnostics_stride", 4)),
            1,
        )
        for name in (
            "diag_change_sum",
            "diag_retention_sum",
            "diag_state_count",
        ):
            self.register_buffer(
                name,
                torch.zeros((), device=self.memory.device),
                persistent=False,
            )

    def init_memory(self, batch_size, device=None, dtype=None):
        return self.memory.new_zeros(
            (
                int(batch_size),
                self.memory_size[1],
                self.memory_size[2],
                self.memory_size[3],
                self.memory_size[4],
            ),
            device=device or self.memory.device,
            dtype=dtype or self.memory.dtype,
        )

    def reset_memory(self):
        with torch.no_grad():
            self.memory = self.memory.detach()
            self.memory.zero_()

    def reset_diagnostics(self):
        self.diag_change_sum.zero_()
        self.diag_retention_sum.zero_()
        self.diag_state_count.zero_()

    def _record_diagnostics(self, memory, updated, active_mask=None):
        if not self.memory_diagnostics_enabled:
            return
        with torch.no_grad():
            if active_mask is not None:
                active = active_mask.to(
                    device=updated.device,
                    dtype=torch.bool,
                ).view(-1)
                memory = memory[active]
                updated = updated[active]
            if memory.size(0) == 0:
                return

            stride = self.memory_diagnostics_stride
            memory = memory[..., ::stride, ::stride, ::stride]
            updated = updated[..., ::stride, ::stride, ::stride]
            memory_flat = memory.detach().float().flatten(start_dim=1)
            updated_flat = updated.detach().float().flatten(start_dim=1)
            memory_norm = torch.linalg.vector_norm(memory_flat, dim=1)
            updated_norm = torch.linalg.vector_norm(updated_flat, dim=1)
            valid = (memory_norm > 1e-6).to(dtype=memory_norm.dtype)
            change_ratio = torch.linalg.vector_norm(
                updated_flat - memory_flat,
                dim=1,
            ) / (memory_norm + updated_norm + 1e-6)
            retention = F.cosine_similarity(
                memory_flat,
                updated_flat,
                dim=1,
                eps=1e-6,
            )
            self.diag_change_sum.add_((change_ratio * valid).sum())
            self.diag_retention_sum.add_((retention * valid).sum())
            self.diag_state_count.add_(valid.sum())

    def get_diagnostics(self, reset=False):
        state_count = float(self.diag_state_count.item())
        diagnostics = {
            "gate_mean": 0.0,
            "gate_std": 0.0,
            "state_change_ratio": (
                float((self.diag_change_sum / state_count).item())
                if state_count > 0 else 0.0
            ),
            "state_retention": (
                float((self.diag_retention_sum / state_count).item())
                if state_count > 0 else 0.0
            ),
        }
        if reset:
            self.reset_diagnostics()
        return diagnostics

    def forward(self, x, memory=None, active_mask=None):
        current = self.mem_generator(x)
        explicit_memory = memory is not None
        if memory is None:
            if self.memory.size(0) != current.size(0):
                memory = self.init_memory(current.size(0), current.device, current.dtype)
            else:
                memory = self.memory.to(device=current.device, dtype=current.dtype)

        updated = torch.cat([memory[:, :, 1:], current], dim=2)
        if active_mask is not None:
            active = active_mask.to(
                device=updated.device,
                dtype=torch.bool,
            ).view(-1, 1, 1, 1, 1)
            updated = torch.where(active, updated, memory)
        self._record_diagnostics(memory, updated, active_mask=active_mask)
        if not explicit_memory:
            self.memory = updated
        return updated


class mem_block3(nn.Module, _GatedMemoryDiagnostics):
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
        self.register_buffer(
            "memory",
            torch.zeros(1, self.memory_size[1], self.memory_size[2], self.memory_size[3], self.memory_size[4], device=cfg["device"])
        )
        self._setup_gated_memory(cfg, "shallow_gate_init_bias", 0.0)

    def init_memory(self, batch_size, device=None, dtype=None):
        return self.memory.new_zeros(
            (int(batch_size), self.memory_size[1], self.memory_size[2], self.memory_size[3], self.memory_size[4]),
            device=device or self.memory.device,
            dtype=dtype or self.memory.dtype,
        )
    
    def reset_memory(self):
        with torch.no_grad():
            self.memory = self.memory.detach()
            self.memory.zero_()
    
    def forward(self, x, memory=None, active_mask=None):
        current = self.mem_generator(x)
        explicit_memory = memory is not None
        if memory is None:
            if self.memory.size(0) != x.size(0):
                memory = self.init_memory(x.size(0), x.device, x.dtype)
            else:
                memory = self.memory.to(device=x.device, dtype=x.dtype)
        fusion = torch.cat([memory, current], dim=1)
        proposal = self.mem_integrator(fusion)
        gate = self.mem_gate(fusion)
        updated = self._update_memory(memory, proposal, gate, active_mask=active_mask)
        if not explicit_memory:
            self.memory = updated
        return updated


class deep_mem_block(nn.Module, _GatedMemoryDiagnostics):
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
        self.register_buffer(
            "memory",
            torch.zeros(1, self.memory_size[1], self.memory_size[2], self.memory_size[3], self.memory_size[4], device=cfg["device"])
        )
        self._setup_gated_memory(cfg, "deep_gate_init_bias", -2.2)

    def init_memory(self, batch_size, device=None, dtype=None):
        return self.memory.new_zeros(
            (int(batch_size), self.memory_size[1], self.memory_size[2], self.memory_size[3], self.memory_size[4]),
            device=device or self.memory.device,
            dtype=dtype or self.memory.dtype,
        )
    
    def reset_memory(self):
        with torch.no_grad():
            self.memory = self.memory.detach()
            self.memory.zero_()
    
    def forward(self, x, memory=None, active_mask=None):
        explicit_memory = memory is not None
        if memory is None:
            if self.memory.size(0) != x.size(0):
                memory = self.init_memory(x.size(0), x.device, x.dtype)
            else:
                memory = self.memory.to(device=x.device, dtype=x.dtype)
        fusion = torch.cat([memory, x], dim=1)
        proposal = self.mem_integrator(fusion)
        gate = self.mem_gate(fusion)
        updated = self._update_memory(memory, proposal, gate, active_mask=active_mask)
        if not explicit_memory:
            self.memory = updated
        return updated
