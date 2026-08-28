import torch
import torch.nn as nn
import torch.nn.functional as F
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


def _inverse_sigmoid(x, eps=1e-4):
    x = x.clamp(min=eps, max=1.0 - eps)
    return torch.log(x / (1.0 - x))


def _build_temporal_references(max_detection_num, ref_widths, query_counts=None):
    if not isinstance(ref_widths, (list, tuple)):
        ref_widths = [ref_widths]
    ref_widths = [max(min(float(width), 1.0 - 1e-4), 1e-4) for width in ref_widths]
    if not ref_widths:
        raise ValueError("temporal_ref_widths must contain at least one width")

    if query_counts is None:
        base_count, remainder = divmod(int(max_detection_num), len(ref_widths))
        query_counts = [base_count + (scale_idx < remainder) for scale_idx in range(len(ref_widths))]
    elif not isinstance(query_counts, (list, tuple)):
        raise ValueError("temporal_ref_query_counts must be a list of integers")
    else:
        query_counts = [int(count) for count in query_counts]

    if len(query_counts) != len(ref_widths):
        raise ValueError(
            "temporal_ref_query_counts and temporal_ref_widths must have the same length"
        )
    if any(count <= 0 for count in query_counts):
        raise ValueError("every temporal reference scale must have at least one query")
    if sum(query_counts) != int(max_detection_num):
        raise ValueError(
            "sum(temporal_ref_query_counts) must equal max_detection_num, "
            f"got {sum(query_counts)} and {max_detection_num}"
        )

    centers_per_scale = []
    widths_per_scale = []
    for width, count in zip(ref_widths, query_counts):
        centers = (torch.arange(count, dtype=torch.float32) + 0.5) / count
        centers_per_scale.append(centers)
        widths_per_scale.append(torch.full((count,), width, dtype=torch.float32))
    return torch.cat(centers_per_scale), torch.cat(widths_per_scale), query_counts


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
        self.shallow_memory_update_mode = str(
            cfg.get("shallow_memory_update_mode", "recurrent")
        ).strip().lower()
        if self.shallow_memory_update_mode not in {"recurrent", "stateless"}:
            raise ValueError(
                "shallow_memory_update_mode must be 'recurrent' or 'stateless', "
                f"got {self.shallow_memory_update_mode!r}"
            )
        self.deep_memory_input_mode = str(
            cfg.get("deep_memory_input_mode", "residual")
        ).strip().lower()
        if self.deep_memory_input_mode not in {"residual", "shallow"}:
            raise ValueError(
                "deep_memory_input_mode must be 'residual' or 'shallow', "
                f"got {self.deep_memory_input_mode!r}"
            )
        self.deep_memory_reset_each_clip = bool(
            cfg.get("deep_memory_reset_each_clip", False)
        )

        fixed_spatial_memory = cfg['mem_block'] in {'mem_block1', 'mem_block4'}
        if cfg["memory_size"] == 'l':
            default_memory_channels, default_memory_temporal = 16, 64
        elif cfg["memory_size"] == 'm':
            default_memory_channels, default_memory_temporal = 32, 32
        else:
            default_memory_channels, default_memory_temporal = 16, 16
        memory_channels = max(
            int(cfg.get("memory_channels", default_memory_channels)),
            1,
        )
        memory_temporal = max(
            int(cfg.get("memory_temporal_size", default_memory_temporal)),
            1,
        )
        memory_h = 17 if fixed_spatial_memory else self.feature_h
        memory_w = 34 if fixed_spatial_memory else self.feature_w
        self.memory_size = [1, memory_channels, memory_temporal, memory_h, memory_w]

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
        self.memory_diagnostics_enabled = bool(cfg.get("memory_diagnostics_enabled", False))
        self.memory_diagnostics_stride = max(
            int(cfg.get("memory_diagnostics_stride", 4)),
            1,
        )
        self.register_buffer(
            "diag_shallow_deep_cosine_sum",
            torch.zeros((), device=cfg["device"]),
            persistent=False,
        )
        self.register_buffer(
            "diag_shallow_deep_count",
            torch.zeros((), device=cfg["device"]),
            persistent=False,
        )

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
        self.use_temporal_references = bool(cfg.get("use_temporal_references", True))
        configured_ref_widths = cfg.get("temporal_ref_widths")
        if configured_ref_widths is None:
            configured_ref_widths = cfg.get(
                "temporal_ref_width",
                1.0 / max(self.max_detection_num, 1),
            )
        ref_centers, ref_widths, self.temporal_ref_query_counts = _build_temporal_references(
            self.max_detection_num,
            configured_ref_widths,
            cfg.get("temporal_ref_query_counts"),
        )
        scale_ids = torch.cat([
            torch.full((count,), scale_idx, dtype=torch.long)
            for scale_idx, count in enumerate(self.temporal_ref_query_counts)
        ])
        scale_num = len(self.temporal_ref_query_counts)
        if scale_num == 1:
            scale_mix = torch.full(
                (self.max_detection_num,),
                0.5 if self.deep_mem_enabled else 0.0,
                dtype=torch.float32,
            )
        else:
            scale_mix = scale_ids.to(torch.float32) / float(scale_num - 1)
        self.register_buffer("temporal_ref_center_logits", _inverse_sigmoid(ref_centers), persistent=False)
        self.register_buffer("temporal_ref_width_logits", _inverse_sigmoid(ref_widths), persistent=False)
        self.register_buffer("temporal_ref_scale_ids", scale_ids, persistent=False)
        self.register_buffer("temporal_ref_scale_mix", scale_mix, persistent=False)
        unique_ref_widths = []
        ref_offset = 0
        for query_count in self.temporal_ref_query_counts:
            unique_ref_widths.append(ref_widths[ref_offset])
            ref_offset += query_count
        self.register_buffer(
            "temporal_proposal_ref_width_logits",
            _inverse_sigmoid(torch.stack(unique_ref_widths)),
            persistent=False,
        )

        self.memory_transformer_head_enabled = bool(
            cfg.get("memory_transformer_head_enabled", False)
        )
        self.memory_transformer_aux_loss_enabled = bool(
            cfg.get("memory_transformer_aux_loss_enabled", False)
        )
        self.memory_transformer_encoder_proposal_enabled = bool(
            cfg.get("memory_transformer_encoder_proposal_enabled", False)
        )
        self.memory_transformer_encoder_proposal_loss_enabled = bool(
            cfg.get(
                "memory_transformer_encoder_proposal_loss_enabled",
                self.memory_transformer_encoder_proposal_enabled,
            )
        )
        self.memory_transformer_query_mode = str(
            cfg.get(
                "memory_transformer_query_mode",
                "proposal" if self.memory_transformer_encoder_proposal_enabled else "fixed",
            )
            or "fixed"
        ).strip().lower()
        if self.memory_transformer_query_mode not in {"fixed", "proposal", "hybrid"}:
            self.memory_transformer_query_mode = "fixed"
        if (
            self.memory_transformer_query_mode in {"proposal", "hybrid"}
            and not self.memory_transformer_encoder_proposal_enabled
        ):
            self.memory_transformer_encoder_proposal_enabled = True
        if self.memory_transformer_query_mode == "hybrid":
            self.memory_transformer_hybrid_fixed_queries = max(
                min(
                    int(cfg.get("memory_transformer_hybrid_fixed_queries", 60)),
                    self.max_detection_num,
                ),
                0,
            )
            default_proposal_queries = (
                self.max_detection_num - self.memory_transformer_hybrid_fixed_queries
            )
            self.memory_transformer_hybrid_proposal_queries = max(
                min(
                    int(cfg.get(
                        "memory_transformer_hybrid_proposal_queries",
                        default_proposal_queries,
                    )),
                    self.max_detection_num - self.memory_transformer_hybrid_fixed_queries,
                ),
                0,
            )
            if self.memory_transformer_hybrid_fixed_queries == 0:
                self.memory_transformer_query_mode = "proposal"
            elif self.memory_transformer_hybrid_proposal_queries == 0:
                self.memory_transformer_query_mode = "fixed"
        else:
            self.memory_transformer_hybrid_fixed_queries = (
                self.max_detection_num if self.memory_transformer_query_mode == "fixed" else 0
            )
            self.memory_transformer_hybrid_proposal_queries = (
                self.max_detection_num if self.memory_transformer_query_mode == "proposal" else 0
            )
        self.denoising_enabled = bool(cfg.get("denoising_enabled", False))
        self.denoising_groups = max(int(cfg.get("denoising_groups", 5)), 1)
        self.denoising_label_noise_ratio = max(
            min(float(cfg.get("denoising_label_noise_ratio", 0.1)), 1.0),
            0.0,
        )
        self.denoising_box_noise_scale = max(
            float(cfg.get("denoising_box_noise_scale", 0.4)),
            0.0,
        )
        self.denoising_max_queries = max(
            int(cfg.get("denoising_max_queries", 100)),
            0,
        )
        self.memory_transformer_iterative_refine_enabled = bool(
            cfg.get("memory_transformer_iterative_refine_enabled", False)
        )
        self.memory_transformer_iterative_refine_detach = bool(
            cfg.get("memory_transformer_iterative_refine_detach", True)
        )
        self.memory_transformer_prior_source = str(
            cfg.get("memory_transformer_prior_source", "deep")
        ).strip().lower()
        if self.memory_transformer_prior_source not in {"deep", "shallow"}:
            raise ValueError(
                "memory_transformer_prior_source must be 'deep' or 'shallow', "
                f"got {self.memory_transformer_prior_source!r}"
            )
        prior_source_available = (
            self.deep_mem_enabled
            if self.memory_transformer_prior_source == "deep"
            else True
        )
        self.memory_transformer_deep_prior_enabled = (
            bool(cfg.get("memory_transformer_deep_prior_enabled", False))
            and prior_source_available
        )
        self.memory_transformer_deep_prior_loss_enabled = bool(
            cfg.get(
                "memory_transformer_deep_prior_loss_enabled",
                self.memory_transformer_deep_prior_enabled,
            )
        )
        self.memory_transformer_deep_prior_context_scale = max(
            float(cfg.get("memory_transformer_deep_prior_context_scale", 3.0)),
            0.1,
        )
        self.memory_transformer_deep_prior_delta_scale = max(
            float(cfg.get("memory_transformer_deep_prior_delta_scale", 0.75)),
            0.0,
        )
        transformer_dim = max(int(cfg.get("memory_transformer_dim", 256)), 32)
        transformer_heads = max(int(cfg.get("memory_transformer_heads", 8)), 1)
        while transformer_dim % transformer_heads != 0 and transformer_heads > 1:
            transformer_heads -= 1
        transformer_encoder_layers = max(
            int(cfg.get("memory_transformer_encoder_layers", 2)),
            0,
        )
        transformer_decoder_layers = max(
            int(cfg.get("memory_transformer_decoder_layers", 3)),
            1,
        )
        transformer_ff_dim = max(
            int(cfg.get("memory_transformer_ff_dim", transformer_dim * 4)),
            transformer_dim,
        )
        transformer_dropout = max(
            min(float(cfg.get("memory_transformer_dropout", head_dropout)), 0.8),
            0.0,
        )
        if self.memory_transformer_head_enabled:
            transformer_in_channels = self.memory_size[1]
            if self.deep_mem_enabled and not self.memory_transformer_deep_prior_enabled:
                transformer_in_channels *= 2
            self.memory_transformer_input_proj = nn.Sequential(
                nn.Conv1d(transformer_in_channels, transformer_dim, kernel_size=1, bias=False),
                nn.GroupNorm(_group_norm_groups(transformer_dim), transformer_dim),
                nn.SiLU(),
                nn.Dropout(transformer_dropout),
            )
            self.memory_transformer_pos_embed = nn.Embedding(
                self.memory_size[2],
                transformer_dim,
            )
            self.memory_transformer_query_embed = nn.Embedding(
                self.max_detection_num,
                transformer_dim,
            )
            self.memory_transformer_scale_embed = nn.Embedding(
                scale_num,
                transformer_dim,
            )
            self.memory_transformer_ref_embed = nn.Sequential(
                nn.Linear(2, transformer_dim),
                nn.LayerNorm(transformer_dim),
                nn.SiLU(),
                nn.Linear(transformer_dim, transformer_dim),
            )
            if self.denoising_enabled:
                self.memory_transformer_dn_query_embed = nn.Embedding(
                    1,
                    transformer_dim,
                )
                self.memory_transformer_dn_label_embed = nn.Embedding(
                    self.class_num,
                    transformer_dim,
                )
            if transformer_encoder_layers > 0:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=transformer_dim,
                    nhead=transformer_heads,
                    dim_feedforward=transformer_ff_dim,
                    dropout=transformer_dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.memory_transformer_encoder = nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=transformer_encoder_layers,
                    norm=nn.LayerNorm(transformer_dim),
                    enable_nested_tensor=False,
                )
            else:
                self.memory_transformer_encoder = nn.Identity()
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=transformer_dim,
                nhead=transformer_heads,
                dim_feedforward=transformer_ff_dim,
                dropout=transformer_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.memory_transformer_decoder = nn.TransformerDecoder(
                decoder_layer,
                num_layers=transformer_decoder_layers,
                norm=nn.LayerNorm(transformer_dim),
            )
            self.memory_transformer_quality_head = nn.Sequential(
                nn.LayerNorm(transformer_dim),
                nn.Linear(transformer_dim, transformer_dim),
                nn.SiLU(),
                nn.Dropout(transformer_dropout),
                nn.Linear(transformer_dim, 1),
            )
            self.memory_transformer_loc_head = nn.Sequential(
                nn.LayerNorm(transformer_dim),
                nn.Linear(transformer_dim, transformer_dim),
                nn.SiLU(),
                nn.Dropout(transformer_dropout),
                nn.Linear(transformer_dim, 2),
            )
            self.memory_transformer_cls_head = nn.Sequential(
                nn.LayerNorm(transformer_dim),
                nn.Linear(transformer_dim, transformer_dim),
                nn.SiLU(),
                nn.Dropout(transformer_dropout),
                nn.Linear(transformer_dim, self.class_num),
            )
            if self.memory_transformer_encoder_proposal_enabled:
                self.memory_transformer_proposal_quality_head = nn.Sequential(
                    nn.LayerNorm(transformer_dim),
                    nn.Linear(transformer_dim, transformer_dim),
                    nn.SiLU(),
                    nn.Dropout(transformer_dropout),
                    nn.Linear(transformer_dim, 1),
                )
                self.memory_transformer_proposal_cls_head = nn.Sequential(
                    nn.LayerNorm(transformer_dim),
                    nn.Linear(transformer_dim, transformer_dim),
                    nn.SiLU(),
                    nn.Dropout(transformer_dropout),
                    nn.Linear(transformer_dim, self.class_num),
                )
            if self.memory_transformer_deep_prior_enabled:
                self.memory_transformer_deep_prior_input_proj = nn.Sequential(
                    nn.Conv1d(
                        self.memory_size[1],
                        transformer_dim,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.GroupNorm(_group_norm_groups(transformer_dim), transformer_dim),
                    nn.SiLU(),
                    nn.Dropout(transformer_dropout),
                )
                self.memory_transformer_deep_prior_context_proj = nn.Sequential(
                    nn.LayerNorm(transformer_dim),
                    nn.Linear(transformer_dim, transformer_dim),
                    nn.SiLU(),
                    nn.Dropout(transformer_dropout),
                    nn.Linear(transformer_dim, transformer_dim),
                )
                self.memory_transformer_deep_prior_query_proj = nn.Sequential(
                    nn.LayerNorm(transformer_dim),
                    nn.Linear(transformer_dim, transformer_dim),
                    nn.SiLU(),
                    nn.Dropout(transformer_dropout),
                    nn.Linear(transformer_dim, transformer_dim),
                )
                self.memory_transformer_deep_prior_norm = nn.LayerNorm(transformer_dim)
                self.memory_transformer_deep_prior_query_norm = nn.LayerNorm(transformer_dim)
                self.memory_transformer_deep_prior_quality_head = nn.Sequential(
                    nn.LayerNorm(transformer_dim),
                    nn.Linear(transformer_dim, transformer_dim),
                    nn.SiLU(),
                    nn.Dropout(transformer_dropout),
                    nn.Linear(transformer_dim, 1),
                )
                self.memory_transformer_deep_prior_loc_head = nn.Sequential(
                    nn.LayerNorm(transformer_dim),
                    nn.Linear(transformer_dim, transformer_dim),
                    nn.SiLU(),
                    nn.Dropout(transformer_dropout),
                    nn.Linear(transformer_dim, 2),
                )
                self.memory_transformer_deep_prior_cls_head = nn.Sequential(
                    nn.LayerNorm(transformer_dim),
                    nn.Linear(transformer_dim, transformer_dim),
                    nn.SiLU(),
                    nn.Dropout(transformer_dropout),
                    nn.Linear(transformer_dim, self.class_num),
                )
                self.memory_transformer_deep_prior_query_gate = nn.Parameter(
                    torch.tensor(
                        float(cfg.get("memory_transformer_deep_prior_query_gate_bias", -2.0))
                    )
                )
                deep_prior_points = max(
                    int(cfg.get("memory_transformer_deep_prior_points", 7)),
                    2,
                )
                self.register_buffer(
                    "memory_transformer_deep_prior_offsets",
                    torch.linspace(-0.5, 0.5, deep_prior_points),
                    persistent=False,
                )
            nn.init.zeros_(self.memory_transformer_loc_head[-1].weight)
            nn.init.zeros_(self.memory_transformer_loc_head[-1].bias)
            if self.memory_transformer_deep_prior_enabled:
                nn.init.zeros_(self.memory_transformer_deep_prior_loc_head[-1].weight)
                nn.init.zeros_(self.memory_transformer_deep_prior_loc_head[-1].bias)
            nn.init.normal_(self.memory_transformer_query_embed.weight, std=0.02)
            nn.init.normal_(self.memory_transformer_scale_embed.weight, std=0.02)
            nn.init.normal_(self.memory_transformer_pos_embed.weight, std=0.02)
            if self.denoising_enabled:
                nn.init.normal_(self.memory_transformer_dn_query_embed.weight, std=0.02)
                nn.init.normal_(self.memory_transformer_dn_label_embed.weight, std=0.02)

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

        self.joint_memory_detection_enabled = bool(
            cfg.get("joint_memory_detection_enabled", False)
        )
        joint_dim = max(int(cfg.get("joint_memory_dim", 64)), 16)
        joint_dropout = float(cfg.get("joint_memory_dropout", head_dropout))
        if self.joint_memory_detection_enabled:
            self.joint_memory_context_proj = nn.Sequential(
                nn.LayerNorm(self.memory_size[1]),
                nn.Linear(self.memory_size[1], joint_dim, bias=False),
                nn.LayerNorm(joint_dim),
                nn.SiLU(),
                nn.Dropout(joint_dropout),
            )
            self.joint_memory_gate_mlp = nn.Sequential(
                nn.Linear(joint_dim * 4, joint_dim),
                nn.LayerNorm(joint_dim),
                nn.SiLU(),
                nn.Dropout(joint_dropout),
                nn.Linear(joint_dim, joint_dim * 2),
            )
            self.joint_memory_interaction_proj = nn.Linear(
                joint_dim,
                joint_dim,
                bias=False,
            )
            self.joint_memory_out = nn.Sequential(
                nn.Linear(joint_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(joint_dropout),
            )
            joint_points = max(int(cfg.get("joint_memory_points", 7)), 2)
            self.joint_memory_shallow_context_scale = max(
                float(cfg.get("joint_memory_shallow_context_scale", 1.0)),
                0.1,
            )
            self.joint_memory_deep_context_scale = max(
                float(cfg.get("joint_memory_deep_context_scale", 3.0)),
                self.joint_memory_shallow_context_scale,
            )
            self.joint_memory_max_residual_scale = max(
                float(cfg.get("joint_memory_max_residual_scale", 0.1)),
                0.0,
            )
            residual_gate_bias = float(
                cfg.get("joint_memory_residual_gate_bias", -1.0)
            )
            self.joint_memory_loc_residual_gate = nn.Parameter(
                torch.tensor(residual_gate_bias)
            )
            self.joint_memory_cls_residual_gate = nn.Parameter(
                torch.tensor(residual_gate_bias)
            )
            self.register_buffer(
                "joint_memory_sample_offsets",
                torch.linspace(-0.5, 0.5, joint_points),
                persistent=False,
            )
            self.register_buffer(
                "diag_joint_shallow_gate_sum",
                torch.zeros((), device=cfg["device"], dtype=torch.float64),
                persistent=False,
            )
            self.register_buffer(
                "diag_joint_deep_gate_sum",
                torch.zeros((), device=cfg["device"], dtype=torch.float64),
                persistent=False,
            )
            self.register_buffer(
                "diag_joint_shallow_gate_sq_sum",
                torch.zeros((), device=cfg["device"], dtype=torch.float64),
                persistent=False,
            )
            self.register_buffer(
                "diag_joint_deep_gate_sq_sum",
                torch.zeros((), device=cfg["device"], dtype=torch.float64),
                persistent=False,
            )
            self.register_buffer(
                "diag_joint_gate_count",
                torch.zeros((), device=cfg["device"], dtype=torch.float64),
                persistent=False,
            )
            nn.init.zeros_(self.joint_memory_gate_mlp[-1].weight)
            nn.init.zeros_(self.joint_memory_gate_mlp[-1].bias)

        self.reference_attention_enabled = bool(cfg.get("reference_attention_enabled", False))
        self.reference_attention_layers = max(int(cfg.get("reference_attention_layers", 3)), 1)
        self.reference_attention_points = max(int(cfg.get("reference_attention_points", 7)), 2)
        self.reference_attention_context_scale = max(
            float(cfg.get("reference_attention_context_scale", 1.5)),
            0.1,
        )
        reference_dropout = float(cfg.get("reference_attention_dropout", head_dropout))
        context_channels = max(int(cfg.get("reference_attention_context_dim", 128)), 16)

        if self.reference_attention_enabled:
            self.shallow_temporal_encoder = nn.Sequential(
                nn.Conv1d(
                    self.memory_size[1],
                    context_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(_group_norm_groups(context_channels), context_channels),
                nn.SiLU(),
            )
            self.deep_temporal_encoder = nn.Sequential(
                nn.Conv1d(
                    self.memory_size[1],
                    context_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(_group_norm_groups(context_channels), context_channels),
                nn.SiLU(),
            )
            self.reference_context_proj = nn.Sequential(
                nn.Linear(context_channels, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            )
            self.reference_query_embed = nn.Embedding(self.max_detection_num, hidden_dim)
            self.reference_scale_embed = nn.Embedding(scale_num, hidden_dim)
            self.reference_coord_embed = nn.Sequential(
                nn.Linear(2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.reference_fusion_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(reference_dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(self.reference_attention_layers)
            ])
            self.reference_fusion_norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim)
                for _ in range(self.reference_attention_layers)
            ])
            self.reference_fusion_gates = nn.Parameter(
                torch.full((self.reference_attention_layers,), -1.0)
            )
            self.register_buffer(
                "reference_sample_offsets",
                torch.linspace(-0.5, 0.5, self.reference_attention_points),
                persistent=False,
            )
            nn.init.normal_(self.reference_query_embed.weight, std=0.02)
            nn.init.normal_(self.reference_scale_embed.weight, std=0.02)

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

        # Decoupled detection heads. loc_head predicts deltas from temporal references.
        # Kept as conf_head for checkpoint compatibility; it now predicts temporal IoU quality.
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
        nn.init.zeros_(self.loc_head[-1].weight)
        nn.init.zeros_(self.loc_head[-1].bias)
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

        self.memory_proposal_refine_enabled = bool(
            cfg.get("memory_proposal_refine_enabled", False)
        )
        refine_dim = max(int(cfg.get("memory_proposal_refine_dim", 64)), 16)
        proposal_refine_dropout = float(
            cfg.get("memory_proposal_refine_dropout", head_dropout)
        )
        self.memory_proposal_refine_shallow_context_scale = max(
            float(cfg.get("memory_proposal_refine_shallow_context_scale", 1.0)),
            0.1,
        )
        self.memory_proposal_refine_deep_context_scale = max(
            float(cfg.get("memory_proposal_refine_deep_context_scale", 3.0)),
            self.memory_proposal_refine_shallow_context_scale,
        )
        self.memory_proposal_refine_max_boundary_shift = max(
            float(cfg.get("memory_proposal_refine_max_boundary_shift", 0.25)),
            0.0,
        )
        self.memory_proposal_refine_max_center_shift = max(
            float(cfg.get("memory_proposal_refine_max_center_shift", 0.25)),
            0.0,
        )
        self.memory_proposal_refine_max_log_width_delta = max(
            float(cfg.get("memory_proposal_refine_max_log_width_delta", 0.35)),
            0.0,
        )
        if self.memory_proposal_refine_enabled:
            self.memory_proposal_context_proj = nn.Sequential(
                nn.LayerNorm(self.memory_size[1]),
                nn.Linear(self.memory_size[1], refine_dim, bias=False),
                nn.LayerNorm(refine_dim),
                nn.SiLU(),
                nn.Dropout(proposal_refine_dropout),
            )
            self.memory_proposal_token_proj = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, refine_dim, bias=False),
                nn.LayerNorm(refine_dim),
                nn.SiLU(),
            )
            self.memory_proposal_shallow_boundary_head = nn.Sequential(
                nn.Linear(refine_dim * 2, refine_dim),
                nn.LayerNorm(refine_dim),
                nn.SiLU(),
                nn.Dropout(proposal_refine_dropout),
                nn.Linear(refine_dim, 2),
            )
            self.memory_proposal_deep_cw_head = nn.Sequential(
                nn.Linear(refine_dim * 2, refine_dim),
                nn.LayerNorm(refine_dim),
                nn.SiLU(),
                nn.Dropout(proposal_refine_dropout),
                nn.Linear(refine_dim, 2),
            )
            refine_points = max(int(cfg.get("memory_proposal_refine_points", 7)), 2)
            self.register_buffer(
                "memory_proposal_refine_offsets",
                torch.linspace(-0.5, 0.5, refine_points),
                persistent=False,
            )
            gate_bias = float(cfg.get("memory_proposal_refine_gate_bias", -6.0))
            self.memory_proposal_boundary_gate = nn.Parameter(torch.tensor(gate_bias))
            self.memory_proposal_semantic_gate = nn.Parameter(torch.tensor(gate_bias))
            nn.init.zeros_(self.memory_proposal_shallow_boundary_head[-1].weight)
            nn.init.zeros_(self.memory_proposal_shallow_boundary_head[-1].bias)
            nn.init.zeros_(self.memory_proposal_deep_cw_head[-1].weight)
            nn.init.zeros_(self.memory_proposal_deep_cw_head[-1].bias)

        # Training-only heads are initialized after the detection head so they
        # do not perturb its seeded initialization in Stage 1 ablations.
        self.memory_auxiliary_enabled = bool(
            cfg.get("memory_auxiliary_enabled", False)
        )
        self._last_memory_auxiliary = None
        if self.memory_auxiliary_enabled:
            auxiliary_dropout = float(cfg.get("memory_auxiliary_dropout", 0.1))
            memory_channels = self.memory_size[1]
            self.shallow_boundary_aux_head = nn.Sequential(
                nn.Conv1d(
                    memory_channels,
                    memory_channels,
                    kernel_size=3,
                    padding=1,
                    groups=memory_channels,
                    bias=False,
                ),
                nn.SiLU(),
                nn.Dropout(auxiliary_dropout),
                nn.Conv1d(memory_channels, 2, kernel_size=1),
            )
            self.deep_class_aux_head = nn.Sequential(
                nn.LayerNorm(memory_channels),
                nn.Dropout(auxiliary_dropout),
                nn.Linear(memory_channels, self.class_num),
            )
        
    def _decode_temporal_segments(self, loc_delta):
        if not self.use_temporal_references:
            return loc_delta

        ref_center = self.temporal_ref_center_logits.to(device=loc_delta.device, dtype=loc_delta.dtype)
        ref_width = self.temporal_ref_width_logits.to(device=loc_delta.device, dtype=loc_delta.dtype)
        center = torch.sigmoid(ref_center.view(1, -1) + loc_delta[..., 0])
        width = torch.sigmoid(ref_width.view(1, -1) + loc_delta[..., 1])
        start = (center - 0.5 * width).clamp(min=0.0, max=1.0)
        end = (center + 0.5 * width).clamp(min=0.0, max=1.0)
        return torch.stack([start, end], dim=-1)

    def _decode_temporal_segments_from_reference(self, loc_delta, ref_center, ref_width):
        if not self.use_temporal_references:
            return loc_delta

        center = torch.sigmoid(_inverse_sigmoid(ref_center) + loc_delta[..., 0])
        width = torch.sigmoid(_inverse_sigmoid(ref_width) + loc_delta[..., 1])
        return self._segments_from_center_width(center, width)

    @staticmethod
    def _segments_from_center_width(center, width):
        start = (center - 0.5 * width).clamp(min=0.0, max=1.0)
        end = (center + 0.5 * width).clamp(min=0.0, max=1.0)
        return torch.stack([start, end], dim=-1)

    def _initial_temporal_references(self, batch_size, device, dtype):
        center = torch.sigmoid(
            self.temporal_ref_center_logits.to(device=device, dtype=dtype)
        )
        width = torch.sigmoid(
            self.temporal_ref_width_logits.to(device=device, dtype=dtype)
        )
        return (
            center.unsqueeze(0).expand(batch_size, -1),
            width.unsqueeze(0).expand(batch_size, -1),
        )

    def _refine_temporal_references(self, center, width, delta):
        center = torch.sigmoid(_inverse_sigmoid(center) + delta[..., 0])
        width = torch.sigmoid(_inverse_sigmoid(width) + delta[..., 1])
        return center, width

    @staticmethod
    def _linear_sample_temporal_context(temporal_memory, sample_pos):
        batch_size, channels, temporal_size = temporal_memory.shape
        query_count = sample_pos.size(1)
        point_count = sample_pos.size(2)
        if temporal_size <= 1:
            return temporal_memory.transpose(1, 2).expand(
                batch_size,
                query_count,
                channels,
            )

        sample_idx = sample_pos * float(temporal_size - 1)
        left_idx = sample_idx.floor().to(dtype=torch.long)
        right_idx = (left_idx + 1).clamp(max=temporal_size - 1)
        right_weight = (sample_idx - left_idx.to(dtype=sample_idx.dtype)).unsqueeze(-1)
        left_weight = 1.0 - right_weight

        temporal_bt = temporal_memory.transpose(1, 2)
        left_flat = left_idx.reshape(batch_size, -1)
        right_flat = right_idx.reshape(batch_size, -1)
        left_values = torch.gather(
            temporal_bt,
            dim=1,
            index=left_flat.unsqueeze(-1).expand(-1, -1, channels),
        ).view(batch_size, query_count, point_count, channels)
        right_values = torch.gather(
            temporal_bt,
            dim=1,
            index=right_flat.unsqueeze(-1).expand(-1, -1, channels),
        ).view(batch_size, query_count, point_count, channels)
        sampled = left_values * left_weight + right_values * right_weight
        return sampled.mean(dim=2)

    def _sample_temporal_context(self, temporal_feat, center, width):
        offsets = self.reference_sample_offsets.to(
            device=temporal_feat.device,
            dtype=temporal_feat.dtype,
        )
        sample_pos = (
            center.unsqueeze(-1)
            + offsets.view(1, 1, -1)
            * width.unsqueeze(-1)
            * self.reference_attention_context_scale
        ).clamp(min=0.0, max=1.0)
        return self._linear_sample_temporal_context(temporal_feat, sample_pos)

    def _reference_conditioned_refine(self, base_tokens, shallow_memory, deep_memory):
        batch_size = base_tokens.size(0)
        center, width = self._initial_temporal_references(
            batch_size,
            base_tokens.device,
            base_tokens.dtype,
        )
        scale_ids = self.temporal_ref_scale_ids.to(device=base_tokens.device)
        scale_mix = self.temporal_ref_scale_mix.to(
            device=base_tokens.device,
            dtype=base_tokens.dtype,
        ).view(1, -1, 1)

        shallow_temporal = self.shallow_temporal_encoder(
            shallow_memory.mean(dim=(-1, -2))
        )
        if deep_memory is None:
            deep_temporal = shallow_temporal
        else:
            deep_temporal = self.deep_temporal_encoder(
                deep_memory.mean(dim=(-1, -2))
            )

        query = (
            base_tokens
            + self.reference_query_embed.weight.to(dtype=base_tokens.dtype).unsqueeze(0)
            + self.reference_scale_embed(scale_ids).to(dtype=base_tokens.dtype).unsqueeze(0)
            + self.reference_coord_embed(torch.stack([center, width], dim=-1))
        )

        for layer_idx, (fusion, norm) in enumerate(
            zip(self.reference_fusion_layers, self.reference_fusion_norms)
        ):
            shallow_context = self._sample_temporal_context(
                shallow_temporal,
                center,
                width,
            )
            deep_context = self._sample_temporal_context(
                deep_temporal,
                center,
                width,
            )
            context = shallow_context * (1.0 - scale_mix) + deep_context * scale_mix
            context = (
                self.reference_context_proj(context)
                + self.reference_coord_embed(torch.stack([center, width], dim=-1))
                + self.reference_scale_embed(scale_ids).to(dtype=query.dtype).unsqueeze(0)
            )
            update = fusion(torch.cat([query, context], dim=-1))
            gate = torch.sigmoid(self.reference_fusion_gates[layer_idx]).to(dtype=query.dtype)
            query = norm(query + gate * update)
            center, width = self._refine_temporal_references(
                center,
                width,
                self.loc_head(query),
            )

        return query, self._segments_from_center_width(center, width)

    def reset_memory(self):
        self.mem.reset_memory()
        if self.deep_mem_enabled:
            self.deep_mem.reset_memory()

    def reset_memory_diagnostics(self):
        self.mem.reset_diagnostics()
        if self.deep_mem_enabled:
            self.deep_mem.reset_diagnostics()
        self.diag_shallow_deep_cosine_sum.zero_()
        self.diag_shallow_deep_count.zero_()
        if self.joint_memory_detection_enabled:
            self.diag_joint_shallow_gate_sum.zero_()
            self.diag_joint_deep_gate_sum.zero_()
            self.diag_joint_shallow_gate_sq_sum.zero_()
            self.diag_joint_deep_gate_sq_sum.zero_()
            self.diag_joint_gate_count.zero_()

    def _record_memory_pair_diagnostics(self, shallow, deep, active_mask=None):
        if not self.memory_diagnostics_enabled or deep is None:
            return
        with torch.no_grad():
            if active_mask is not None:
                active = active_mask.to(device=shallow.device, dtype=torch.bool).view(-1)
                shallow = shallow[active]
                deep = deep[active]
            stride = self.memory_diagnostics_stride
            shallow = shallow[..., ::stride, ::stride, ::stride]
            deep = deep[..., ::stride, ::stride, ::stride]
            cosine = F.cosine_similarity(
                shallow.detach().float().flatten(start_dim=1),
                deep.detach().float().flatten(start_dim=1),
                dim=1,
                eps=1e-6,
            )
            self.diag_shallow_deep_cosine_sum.add_(cosine.sum())
            self.diag_shallow_deep_count.add_(cosine.numel())

    def get_memory_diagnostics(self, reset=False):
        shallow = self.mem.get_diagnostics(reset=False)
        if self.deep_mem_enabled:
            deep = self.deep_mem.get_diagnostics(reset=False)
        else:
            deep = {
                "gate_mean": 0.0,
                "gate_std": 0.0,
                "state_change_ratio": 0.0,
                "state_retention": 0.0,
            }
        pair_count = float(self.diag_shallow_deep_count.item())
        diagnostics = {
            "shallow_gate_mean": shallow["gate_mean"],
            "shallow_gate_std": shallow["gate_std"],
            "shallow_state_change_ratio": shallow["state_change_ratio"],
            "shallow_state_retention": shallow["state_retention"],
            "deep_gate_mean": deep["gate_mean"],
            "deep_gate_std": deep["gate_std"],
            "deep_state_change_ratio": deep["state_change_ratio"],
            "deep_state_retention": deep["state_retention"],
            "shallow_deep_cosine": (
                float((self.diag_shallow_deep_cosine_sum / pair_count).item())
                if pair_count > 0 else 0.0
            ),
        }
        if self.joint_memory_detection_enabled:
            count = float(self.diag_joint_gate_count.item())
            shallow_gate = (
                float((self.diag_joint_shallow_gate_sum / count).item())
                if count > 0 else 0.5
            )
            deep_gate = (
                float((self.diag_joint_deep_gate_sum / count).item())
                if count > 0 else 0.5
            )
            if count > 0:
                shallow_gate_std = float((
                    self.diag_joint_shallow_gate_sq_sum / count
                    - shallow_gate ** 2
                ).clamp_min(0.0).sqrt().item())
                deep_gate_std = float((
                    self.diag_joint_deep_gate_sq_sum / count
                    - deep_gate ** 2
                ).clamp_min(0.0).sqrt().item())
            else:
                shallow_gate_std = 0.0
                deep_gate_std = 0.0
            diagnostics.update({
                "joint_shallow_gate_mean": shallow_gate,
                "joint_deep_gate_mean": deep_gate,
                "joint_shallow_gate_std": shallow_gate_std,
                "joint_deep_gate_std": deep_gate_std,
                "joint_loc_residual_scale": float(
                    (
                        self.joint_memory_max_residual_scale
                        * torch.sigmoid(self.joint_memory_loc_residual_gate)
                    ).item()
                ),
                "joint_cls_residual_scale": float(
                    (
                        self.joint_memory_max_residual_scale
                        * torch.sigmoid(self.joint_memory_cls_residual_gate)
                    ).item()
                ),
            })
        else:
            diagnostics.update({
                "joint_shallow_gate_mean": 0.0,
                "joint_deep_gate_mean": 0.0,
                "joint_shallow_gate_std": 0.0,
                "joint_deep_gate_std": 0.0,
                "joint_loc_residual_scale": 0.0,
                "joint_cls_residual_scale": 0.0,
            })
        if reset:
            self.reset_memory_diagnostics()
        return diagnostics

    def _sample_joint_memory_context(
        self,
        temporal_memory,
        center,
        width,
        context_scale,
    ):
        offsets = self.joint_memory_sample_offsets.to(
            device=temporal_memory.device,
            dtype=temporal_memory.dtype,
        )
        sample_pos = (
            center.unsqueeze(-1)
            + offsets.view(1, 1, -1)
            * width.unsqueeze(-1)
            * context_scale
        ).clamp(min=0.0, max=1.0)
        return self._linear_sample_temporal_context(temporal_memory, sample_pos)

    def _record_joint_memory_gates(self, shallow_gate, deep_gate):
        if not self.memory_diagnostics_enabled:
            return
        with torch.no_grad():
            shallow_gate = shallow_gate.to(dtype=torch.float64)
            deep_gate = deep_gate.to(dtype=torch.float64)
            self.diag_joint_shallow_gate_sum.add_(shallow_gate.sum())
            self.diag_joint_deep_gate_sum.add_(deep_gate.sum())
            self.diag_joint_shallow_gate_sq_sum.add_(shallow_gate.square().sum())
            self.diag_joint_deep_gate_sq_sum.add_(deep_gate.square().sum())
            self.diag_joint_gate_count.add_(shallow_gate.numel())

    def _apply_joint_memory_detection(
        self,
        base_tokens,
        shallow_memory,
        deep_memory,
    ):
        shallow_temporal = shallow_memory.mean(dim=(-1, -2))
        if deep_memory is None:
            deep_temporal = shallow_temporal
        else:
            deep_temporal = deep_memory.mean(dim=(-1, -2))

        center, width = self._initial_temporal_references(
            base_tokens.size(0),
            base_tokens.device,
            base_tokens.dtype,
        )
        z_shallow = self.joint_memory_context_proj(
            self._sample_joint_memory_context(
                shallow_temporal,
                center,
                width,
                self.joint_memory_shallow_context_scale,
            )
        )
        z_deep = self.joint_memory_context_proj(
            self._sample_joint_memory_context(
                deep_temporal,
                center,
                width,
                self.joint_memory_deep_context_scale,
            )
        )
        interaction = z_shallow * z_deep
        gate_input = torch.cat(
            [z_shallow, z_deep, (z_shallow - z_deep).abs(), interaction],
            dim=-1,
        )
        shallow_gate, deep_gate = torch.sigmoid(
            self.joint_memory_gate_mlp(gate_input)
        ).chunk(2, dim=-1)
        self._record_joint_memory_gates(shallow_gate, deep_gate)

        interaction = self.joint_memory_interaction_proj(interaction)
        loc_context = z_shallow + deep_gate * z_deep + interaction
        cls_context = z_deep + shallow_gate * z_shallow + interaction
        loc_update = self.joint_memory_out(loc_context)
        cls_update = self.joint_memory_out(cls_context)
        loc_scale = self.joint_memory_max_residual_scale * torch.sigmoid(
            self.joint_memory_loc_residual_gate
        )
        cls_scale = self.joint_memory_max_residual_scale * torch.sigmoid(
            self.joint_memory_cls_residual_gate
        )
        return (
            base_tokens + loc_scale.to(base_tokens.dtype) * loc_update,
            base_tokens + cls_scale.to(base_tokens.dtype) * cls_update,
        )

    def _sample_memory_proposal_refine_context(
        self,
        temporal_memory,
        center,
        width,
        context_scale,
    ):
        offsets = self.memory_proposal_refine_offsets.to(
            device=temporal_memory.device,
            dtype=temporal_memory.dtype,
        )
        sample_pos = (
            center.unsqueeze(-1)
            + offsets.view(1, 1, -1)
            * width.unsqueeze(-1)
            * context_scale
        ).clamp(min=0.0, max=1.0)
        return self._linear_sample_temporal_context(temporal_memory, sample_pos)

    def _apply_memory_proposal_refinement(
        self,
        loc,
        reg_tokens,
        shallow_memory,
        deep_memory,
    ):
        if not self.memory_proposal_refine_enabled or shallow_memory is None:
            return loc

        start = torch.minimum(loc[..., 0], loc[..., 1])
        end = torch.maximum(loc[..., 0], loc[..., 1])
        center = 0.5 * (start + end)
        width = (end - start).clamp(min=1e-6)

        shallow_temporal = shallow_memory.mean(dim=(-1, -2))
        if deep_memory is None:
            deep_temporal = shallow_temporal
        else:
            deep_temporal = deep_memory.mean(dim=(-1, -2))

        token_context = self.memory_proposal_token_proj(reg_tokens)
        shallow_context = self.memory_proposal_context_proj(
            self._sample_memory_proposal_refine_context(
                shallow_temporal,
                center,
                width,
                self.memory_proposal_refine_shallow_context_scale,
            )
        )
        deep_context = self.memory_proposal_context_proj(
            self._sample_memory_proposal_refine_context(
                deep_temporal,
                center,
                width,
                self.memory_proposal_refine_deep_context_scale,
            )
        )

        semantic_delta = self.memory_proposal_deep_cw_head(
            torch.cat([token_context, deep_context], dim=-1)
        ).tanh()
        semantic_gate = torch.sigmoid(
            self.memory_proposal_semantic_gate
        ).to(dtype=loc.dtype)
        center_shift = (
            semantic_delta[..., 0]
            * width
            * self.memory_proposal_refine_max_center_shift
            * semantic_gate
        )
        log_width_delta = (
            semantic_delta[..., 1]
            * self.memory_proposal_refine_max_log_width_delta
            * semantic_gate
        )
        refined_center = (center + center_shift).clamp(min=0.0, max=1.0)
        refined_width = (width * torch.exp(log_width_delta)).clamp(
            min=1e-6,
            max=1.0,
        )
        refined_loc = self._segments_from_center_width(
            refined_center,
            refined_width,
        )

        boundary_delta = self.memory_proposal_shallow_boundary_head(
            torch.cat([token_context, shallow_context], dim=-1)
        ).tanh()
        boundary_gate = torch.sigmoid(
            self.memory_proposal_boundary_gate
        ).to(dtype=loc.dtype)
        boundary_shift = (
            boundary_delta
            * refined_width.unsqueeze(-1)
            * self.memory_proposal_refine_max_boundary_shift
            * boundary_gate
        )
        refined_start = refined_loc[..., 0] + boundary_shift[..., 0]
        refined_end = refined_loc[..., 1] + boundary_shift[..., 1]
        start = torch.minimum(refined_start, refined_end).clamp(min=0.0, max=1.0)
        end = torch.maximum(refined_start, refined_end).clamp(min=0.0, max=1.0)
        return torch.stack([start, end], dim=-1)

    def init_state(self, batch_size, device=None, dtype=None):
        state = {
            "shallow": self.mem.init_memory(batch_size, device=device, dtype=dtype),
            "deep": None,
        }
        if self.deep_mem_enabled:
            state["deep"] = self.deep_mem.init_memory(batch_size, device=device, dtype=dtype)
        return state

    def _memory_transformer_position(self, temporal_size, device, dtype):
        pos_weight = self.memory_transformer_pos_embed.weight
        if temporal_size == pos_weight.size(0):
            pos = pos_weight
        else:
            pos = F.interpolate(
                pos_weight.transpose(0, 1).unsqueeze(0),
                size=temporal_size,
                mode="linear",
                align_corners=True,
            ).squeeze(0).transpose(0, 1)
        return pos.to(device=device, dtype=dtype).unsqueeze(0)

    def _sample_memory_transformer_deep_prior_context(
        self,
        temporal_memory,
        center,
        width,
    ):
        offsets = self.memory_transformer_deep_prior_offsets.to(
            device=temporal_memory.device,
            dtype=temporal_memory.dtype,
        )
        sample_pos = (
            center.unsqueeze(-1)
            + offsets.view(1, 1, -1)
            * width.unsqueeze(-1)
            * self.memory_transformer_deep_prior_context_scale
        ).clamp(min=0.0, max=1.0)
        return self._linear_sample_temporal_context(temporal_memory, sample_pos)

    def _apply_memory_transformer_deep_prior(
        self,
        query,
        center,
        width,
        deep_prior_memory,
    ):
        deep_context = self._sample_memory_transformer_deep_prior_context(
            deep_prior_memory.transpose(1, 2),
            center,
            width,
        )
        prior_context = self.memory_transformer_deep_prior_context_proj(deep_context)
        prior_tokens = self.memory_transformer_deep_prior_norm(query + prior_context)
        prior_delta = (
            self.memory_transformer_deep_prior_loc_head(prior_tokens).tanh()
            * self.memory_transformer_deep_prior_delta_scale
        )
        prior_center, prior_width = self._refine_temporal_references(
            center,
            width,
            prior_delta,
        )
        prior_loc = self._segments_from_center_width(prior_center, prior_width)
        prior_quality_logits = self.memory_transformer_deep_prior_quality_head(
            prior_tokens
        )
        prior_cls_logits = self.memory_transformer_deep_prior_cls_head(prior_tokens)

        query_gate = torch.sigmoid(
            self.memory_transformer_deep_prior_query_gate
        ).to(dtype=query.dtype)
        query_update = self.memory_transformer_deep_prior_query_proj(deep_context)
        query = self.memory_transformer_deep_prior_query_norm(
            query + query_gate * query_update
        )
        prior_output = torch.cat(
            [prior_quality_logits, prior_loc, prior_cls_logits],
            dim=-1,
        )
        return query, prior_center, prior_width, prior_output

    def _memory_transformer_candidate_proposals(self, memory):
        batch_size, temporal_size, hidden_dim = memory.shape
        device = memory.device
        dtype = memory.dtype
        centers = (
            torch.arange(temporal_size, device=device, dtype=dtype) + 0.5
        ) / float(max(temporal_size, 1))
        widths = torch.sigmoid(
            self.temporal_proposal_ref_width_logits.to(device=device, dtype=dtype)
        )
        scale_num = widths.numel()

        proposal_center = centers.view(1, temporal_size, 1).expand(
            batch_size,
            temporal_size,
            scale_num,
        )
        proposal_width = widths.view(1, 1, scale_num).expand(
            batch_size,
            temporal_size,
            scale_num,
        )
        scale_ids = torch.arange(scale_num, device=device, dtype=torch.long)
        flat_scale_ids = scale_ids.view(1, scale_num).expand(
            temporal_size,
            scale_num,
        ).reshape(-1)

        flat_center = proposal_center.reshape(batch_size, -1)
        flat_width = proposal_width.reshape(batch_size, -1)
        flat_ref = torch.stack([flat_center, flat_width], dim=-1)
        candidate_tokens = memory.unsqueeze(2).expand(
            batch_size,
            temporal_size,
            scale_num,
            hidden_dim,
        ).reshape(batch_size, -1, hidden_dim)
        candidate_tokens = (
            candidate_tokens
            + self.memory_transformer_scale_embed(flat_scale_ids).to(dtype=dtype).unsqueeze(0)
            + self.memory_transformer_ref_embed(flat_ref)
        )
        candidate_segments = self._segments_from_center_width(flat_center, flat_width)
        return candidate_tokens, flat_center, flat_width, candidate_segments

    @staticmethod
    def _gather_query_candidates(values, indices):
        if values.dim() == 2:
            return torch.gather(values, 1, indices)
        return torch.gather(
            values,
            1,
            indices.unsqueeze(-1).expand(-1, -1, values.size(-1)),
        )

    def _as_denoising_target_list(self, dn_targets, batch_size, device, dtype):
        if dn_targets is None:
            return None
        if torch.is_tensor(dn_targets):
            if dn_targets.dim() == 3 and dn_targets.size(0) == batch_size:
                return [
                    dn_targets[idx].to(device=device, dtype=dtype)
                    for idx in range(batch_size)
                ]
            if dn_targets.dim() == 2 and batch_size == 1:
                return [dn_targets.to(device=device, dtype=dtype)]
            return None
        if not isinstance(dn_targets, (list, tuple)):
            return None

        target_list = []
        for idx in range(batch_size):
            target = dn_targets[idx] if idx < len(dn_targets) else None
            if target is None:
                target = torch.empty(
                    (0, self.class_num + 2),
                    device=device,
                    dtype=dtype,
                )
            elif torch.is_tensor(target):
                if target.dim() == 3 and target.size(0) == 1:
                    target = target[0]
                target = target.to(device=device, dtype=dtype)
            else:
                target = torch.as_tensor(target, device=device, dtype=dtype)
            target_list.append(target)
        return target_list

    def _build_memory_transformer_denoising_queries(
        self,
        dn_targets,
        batch_size,
        device,
        dtype,
    ):
        if (
            not self.training
            or not self.denoising_enabled
            or self.denoising_max_queries <= 0
        ):
            return None

        target_list = self._as_denoising_target_list(
            dn_targets,
            batch_size,
            device,
            dtype,
        )
        if target_list is None:
            return None

        query_count = self.denoising_max_queries
        dn_query = torch.zeros(
            batch_size,
            query_count,
            self.memory_transformer_query_embed.embedding_dim,
            device=device,
            dtype=dtype,
        )
        dn_center = torch.full(
            (batch_size, query_count),
            0.5,
            device=device,
            dtype=dtype,
        )
        dn_width = torch.full(
            (batch_size, query_count),
            1.0 / max(query_count, 1),
            device=device,
            dtype=dtype,
        )
        dn_supervision = torch.zeros(
            batch_size,
            query_count,
            self.class_num + 2,
            device=device,
            dtype=dtype,
        )
        dn_mask = torch.zeros(
            batch_size,
            query_count,
            device=device,
            dtype=torch.bool,
        )

        ref_widths = torch.sigmoid(
            self.temporal_proposal_ref_width_logits.to(device=device, dtype=dtype)
        ).clamp(min=1e-4)
        base_token = self.memory_transformer_dn_query_embed.weight[0].to(
            device=device,
            dtype=dtype,
        )

        for batch_idx, target in enumerate(target_list):
            if target.numel() == 0 or target.size(0) == 0:
                continue
            target = target[:, : self.class_num + 2]
            if target.size(-1) != self.class_num + 2:
                continue

            gt_start = torch.minimum(target[:, 0], target[:, 1]).clamp(0.0, 1.0)
            gt_end = torch.maximum(target[:, 0], target[:, 1]).clamp(0.0, 1.0)
            gt_width = (gt_end - gt_start).clamp(min=1e-4, max=1.0)
            gt_center = (0.5 * (gt_start + gt_end)).clamp(0.0, 1.0)
            gt_cls = target[:, 2:].argmax(dim=-1).to(dtype=torch.long)

            gt_count = target.size(0)
            repeated_idx = torch.arange(gt_count, device=device).repeat(
                self.denoising_groups
            )
            if repeated_idx.numel() > query_count:
                perm = torch.randperm(repeated_idx.numel(), device=device)[:query_count]
                repeated_idx = repeated_idx[perm]
            valid_count = repeated_idx.numel()
            if valid_count == 0:
                continue

            clean_center = gt_center[repeated_idx]
            clean_width = gt_width[repeated_idx]
            center_noise = (
                torch.rand(valid_count, device=device, dtype=dtype) * 2.0 - 1.0
            ) * clean_width * self.denoising_box_noise_scale
            log_width_noise = (
                torch.rand(valid_count, device=device, dtype=dtype) * 2.0 - 1.0
            ) * self.denoising_box_noise_scale
            noisy_center = (clean_center + center_noise).clamp(0.0, 1.0)
            noisy_width = (clean_width * torch.exp(log_width_noise)).clamp(
                min=1e-4,
                max=1.0,
            )

            noisy_cls = gt_cls[repeated_idx].clone()
            if self.denoising_label_noise_ratio > 0.0 and self.class_num > 1:
                noise_mask = (
                    torch.rand(valid_count, device=device)
                    < self.denoising_label_noise_ratio
                )
                if noise_mask.any():
                    noisy_cls[noise_mask] = torch.randint(
                        low=0,
                        high=self.class_num,
                        size=(int(noise_mask.sum().item()),),
                        device=device,
                    )

            scale_cost = (
                torch.log(noisy_width[:, None])
                - torch.log(ref_widths[None, :])
            ).abs()
            scale_ids = scale_cost.argmin(dim=1)
            ref = torch.stack([noisy_center, noisy_width], dim=-1)
            dn_query[batch_idx, :valid_count] = (
                base_token
                + self.memory_transformer_dn_label_embed(noisy_cls).to(dtype=dtype)
                + self.memory_transformer_scale_embed(scale_ids).to(dtype=dtype)
                + self.memory_transformer_ref_embed(ref)
            )
            dn_center[batch_idx, :valid_count] = noisy_center
            dn_width[batch_idx, :valid_count] = noisy_width
            dn_supervision[batch_idx, :valid_count, 0] = gt_start[repeated_idx]
            dn_supervision[batch_idx, :valid_count, 1] = gt_end[repeated_idx]
            dn_supervision[batch_idx, :valid_count, 2:] = target[repeated_idx, 2:]
            dn_mask[batch_idx, :valid_count] = True

        if not dn_mask.any():
            return None
        return {
            "query": dn_query,
            "center": dn_center,
            "width": dn_width,
            "target": dn_supervision,
            "mask": dn_mask,
        }

    @staticmethod
    def _denoising_attention_mask(dn_count, normal_count, device):
        total_count = int(dn_count) + int(normal_count)
        if dn_count <= 0 or normal_count <= 0:
            return None
        mask = torch.zeros(
            total_count,
            total_count,
            device=device,
            dtype=torch.bool,
        )
        mask[:dn_count, dn_count:] = True
        mask[dn_count:, :dn_count] = True
        return mask

    def _decode_memory_transformer_head(
        self,
        shallow_memory,
        deep_memory=None,
        dn_targets=None,
    ):
        shallow_temporal = shallow_memory.mean(dim=(-1, -2))
        use_prior = self.memory_transformer_deep_prior_enabled
        if self.memory_transformer_prior_source == "deep":
            use_prior = use_prior and self.deep_mem_enabled and deep_memory is not None
            prior_temporal = (
                deep_memory.mean(dim=(-1, -2)) if use_prior else None
            )
        else:
            prior_temporal = shallow_temporal if use_prior else None

        if use_prior:
            deep_temporal = (
                deep_memory.mean(dim=(-1, -2))
                if deep_memory is not None else None
            )
            temporal_memory = shallow_temporal
        elif self.deep_mem_enabled and deep_memory is not None:
            deep_temporal = deep_memory.mean(dim=(-1, -2))
            temporal_memory = torch.cat([shallow_temporal, deep_temporal], dim=1)
        else:
            deep_temporal = None
            temporal_memory = shallow_temporal

        src = self.memory_transformer_input_proj(temporal_memory).transpose(1, 2)
        src = src + self._memory_transformer_position(
            src.size(1),
            src.device,
            src.dtype,
        )
        memory = self.memory_transformer_encoder(src)
        deep_prior_memory = None
        if use_prior:
            deep_prior_memory = self.memory_transformer_deep_prior_input_proj(
                prior_temporal
            ).transpose(1, 2)
            deep_prior_memory = deep_prior_memory + self._memory_transformer_position(
                deep_prior_memory.size(1),
                deep_prior_memory.device,
                deep_prior_memory.dtype,
            )

        batch_size = src.size(0)
        center, width = self._initial_temporal_references(
            batch_size,
            src.device,
            src.dtype,
        )
        scale_ids = self.temporal_ref_scale_ids.to(device=src.device)
        fixed_query = (
            self.memory_transformer_query_embed.weight.to(dtype=src.dtype).unsqueeze(0)
            + self.memory_transformer_scale_embed(scale_ids).to(dtype=src.dtype).unsqueeze(0)
            + self.memory_transformer_ref_embed(torch.stack([center, width], dim=-1))
        ).expand(batch_size, -1, -1)

        proposal_query = None
        proposal_center = None
        proposal_width = None
        if self.memory_transformer_encoder_proposal_enabled:
            (
                candidate_tokens,
                candidate_center,
                candidate_width,
                candidate_segments,
            ) = self._memory_transformer_candidate_proposals(memory)
            proposal_quality_logits = self.memory_transformer_proposal_quality_head(
                candidate_tokens
            )
            proposal_cls_logits = self.memory_transformer_proposal_cls_head(
                candidate_tokens
            )
            proposal_scores = proposal_quality_logits.squeeze(-1)
            if self.memory_transformer_query_mode == "hybrid":
                proposal_query_count = min(
                    self.memory_transformer_hybrid_proposal_queries,
                    proposal_scores.size(1),
                )
                query_offset = self.memory_transformer_hybrid_fixed_queries
            elif self.memory_transformer_query_mode == "proposal":
                proposal_query_count = min(self.max_detection_num, proposal_scores.size(1))
                query_offset = 0
            else:
                proposal_query_count = 0
                query_offset = 0
            if proposal_query_count > 0:
                _, proposal_indices = torch.topk(
                    proposal_scores,
                    k=proposal_query_count,
                    dim=1,
                    largest=True,
                    sorted=True,
                )
                proposal_query = self._gather_query_candidates(
                    candidate_tokens,
                    proposal_indices,
                )
                proposal_center = self._gather_query_candidates(
                    candidate_center,
                    proposal_indices,
                )
                proposal_width = self._gather_query_candidates(
                    candidate_width,
                    proposal_indices,
                )
                proposal_query = (
                    proposal_query
                    + self.memory_transformer_query_embed.weight[
                        query_offset:query_offset + proposal_query_count
                    ].to(dtype=src.dtype).unsqueeze(0)
                )
            if (
                self.training
                and self.memory_transformer_encoder_proposal_loss_enabled
            ):
                if self._last_memory_auxiliary is None:
                    self._last_memory_auxiliary = {}
                self._last_memory_auxiliary[
                    "memory_transformer_encoder_proposal_outputs"
                ] = torch.cat(
                    [proposal_quality_logits, candidate_segments, proposal_cls_logits],
                    dim=-1,
                )

        if self.memory_transformer_query_mode == "hybrid":
            fixed_count = min(
                self.memory_transformer_hybrid_fixed_queries,
                self.max_detection_num,
            )
            query_parts = [fixed_query[:, :fixed_count]]
            center_parts = [center[:, :fixed_count]]
            width_parts = [width[:, :fixed_count]]
            if proposal_query is not None:
                query_parts.append(proposal_query)
                center_parts.append(proposal_center)
                width_parts.append(proposal_width)
            query = torch.cat(query_parts, dim=1)
            center = torch.cat(center_parts, dim=1)
            width = torch.cat(width_parts, dim=1)
        elif self.memory_transformer_query_mode == "proposal" and proposal_query is not None:
            query = proposal_query
            center = proposal_center
            width = proposal_width
        else:
            query = fixed_query

        if use_prior:
            query, center, width, deep_prior_output = (
                self._apply_memory_transformer_deep_prior(
                    query,
                    center,
                    width,
                    deep_prior_memory,
                )
            )
            if self.training and self.memory_transformer_deep_prior_loss_enabled:
                if self._last_memory_auxiliary is None:
                    self._last_memory_auxiliary = {}
                self._last_memory_auxiliary[
                    "memory_transformer_deep_prior_outputs"
                ] = deep_prior_output

        dn_info = self._build_memory_transformer_denoising_queries(
            dn_targets,
            batch_size,
            src.device,
            src.dtype,
        )
        normal_query_count = query.size(1)
        dn_count = 0
        decoder_tgt_mask = None
        decoder_key_padding_mask = None
        if dn_info is not None:
            dn_count = dn_info["query"].size(1)
            decoder_tgt_mask = self._denoising_attention_mask(
                dn_count,
                normal_query_count,
                src.device,
            )
            decoder_key_padding_mask = torch.cat(
                [
                    ~dn_info["mask"],
                    torch.zeros(
                        batch_size,
                        normal_query_count,
                        device=src.device,
                        dtype=torch.bool,
                    ),
                ],
                dim=1,
            )
            query = torch.cat([dn_info["query"], query], dim=1)
            center = torch.cat([dn_info["center"], center], dim=1)
            width = torch.cat([dn_info["width"], width], dim=1)

        aux_outputs = []
        dn_aux_outputs = []
        tokens = query
        current_center = center
        current_width = width
        final_loc = None
        final_quality_logits = None
        final_cls_logits = None
        final_dn_loc = None
        final_dn_quality_logits = None
        final_dn_cls_logits = None
        decoder_norm = getattr(self.memory_transformer_decoder, "norm", None)
        decoder_layers = self.memory_transformer_decoder.layers
        for layer_idx, decoder_layer in enumerate(decoder_layers):
            tokens = decoder_layer(
                tokens,
                memory,
                tgt_mask=decoder_tgt_mask,
                tgt_key_padding_mask=decoder_key_padding_mask,
            )
            pred_tokens = decoder_norm(tokens) if decoder_norm is not None else tokens
            loc_delta_all = self.memory_transformer_loc_head(pred_tokens)
            loc_all = self._decode_temporal_segments_from_reference(
                loc_delta_all,
                current_center,
                current_width,
            )
            quality_logits_all = self.memory_transformer_quality_head(pred_tokens)
            cls_logits_all = self.memory_transformer_cls_head(pred_tokens)

            normal_loc = loc_all[:, dn_count:] if dn_count > 0 else loc_all
            normal_quality_logits = (
                quality_logits_all[:, dn_count:] if dn_count > 0 else quality_logits_all
            )
            normal_cls_logits = (
                cls_logits_all[:, dn_count:] if dn_count > 0 else cls_logits_all
            )
            if dn_count > 0:
                dn_loc = loc_all[:, :dn_count]
                dn_quality_logits = quality_logits_all[:, :dn_count]
                dn_cls_logits = cls_logits_all[:, :dn_count]
            else:
                dn_loc = None
                dn_quality_logits = None
                dn_cls_logits = None

            if (
                self.training
                and self.memory_transformer_aux_loss_enabled
                and layer_idx + 1 < len(decoder_layers)
            ):
                aux_outputs.append(
                    torch.cat(
                        [normal_quality_logits, normal_loc, normal_cls_logits],
                        dim=-1,
                    )
                )
                if dn_count > 0:
                    dn_aux_outputs.append(
                        torch.cat([dn_quality_logits, dn_loc, dn_cls_logits], dim=-1)
                    )

            final_loc = normal_loc
            final_quality_logits = normal_quality_logits
            final_cls_logits = normal_cls_logits
            final_dn_loc = dn_loc
            final_dn_quality_logits = dn_quality_logits
            final_dn_cls_logits = dn_cls_logits

            if (
                self.memory_transformer_iterative_refine_enabled
                and layer_idx + 1 < len(decoder_layers)
            ):
                next_center, next_width = self._refine_temporal_references(
                    current_center,
                    current_width,
                    loc_delta_all,
                )
                if self.memory_transformer_iterative_refine_detach:
                    next_center = next_center.detach()
                    next_width = next_width.detach()
                current_center = next_center
                current_width = next_width

        loc = final_loc
        quality_logits = final_quality_logits
        cls_logits = final_cls_logits
        if aux_outputs:
            if self._last_memory_auxiliary is None:
                self._last_memory_auxiliary = {}
            self._last_memory_auxiliary["memory_transformer_aux_outputs"] = torch.stack(
                aux_outputs,
                dim=1,
            )
        if final_dn_loc is not None and dn_info is not None:
            if self._last_memory_auxiliary is None:
                self._last_memory_auxiliary = {}
            self._last_memory_auxiliary["memory_transformer_denoising_outputs"] = (
                torch.cat(
                    [final_dn_quality_logits, final_dn_loc, final_dn_cls_logits],
                    dim=-1,
                )
            )
            self._last_memory_auxiliary["memory_transformer_denoising_targets"] = (
                dn_info["target"]
            )
            self._last_memory_auxiliary["memory_transformer_denoising_mask"] = (
                dn_info["mask"]
            )
            if dn_aux_outputs:
                self._last_memory_auxiliary[
                    "memory_transformer_denoising_aux_outputs"
                ] = torch.stack(dn_aux_outputs, dim=1)
        return torch.cat([quality_logits, loc, cls_logits], dim=-1)

    def _decode_head(self, x, shallow_memory=None, deep_memory=None, dn_targets=None):
        if (
            self.training
            and self.memory_auxiliary_enabled
            and shallow_memory is not None
        ):
            shallow_temporal = shallow_memory.mean(dim=(-1, -2))
            semantic_memory = deep_memory if deep_memory is not None else shallow_memory
            deep_pooled = semantic_memory.mean(dim=(-1, -2, -3))
            self._last_memory_auxiliary = {
                "shallow_boundary_logits": self.shallow_boundary_aux_head(
                    shallow_temporal
                ),
                "deep_class_logits": self.deep_class_aux_head(deep_pooled),
            }

        if self.memory_transformer_head_enabled and shallow_memory is not None:
            return self._decode_memory_transformer_head(
                shallow_memory,
                deep_memory,
                dn_targets=dn_targets,
            )

        x = self.pre_head_memory_refine(x)
        feat = self.head_stem(x)

        raw_tokens = self.query_proj(feat).flatten(start_dim=2)

        shared_tokens = self.shared_token_mlp(raw_tokens)
        cls_shared_tokens = shared_tokens
        if self.joint_memory_detection_enabled and shallow_memory is not None:
            shared_tokens, cls_shared_tokens = self._apply_joint_memory_detection(
                shared_tokens,
                shallow_memory,
                deep_memory,
            )
        reg_tokens = self.reg_token_mlp(shared_tokens)
        cls_reg_tokens = (
            self.reg_token_mlp(cls_shared_tokens)
            if self.joint_memory_detection_enabled and shallow_memory is not None
            else reg_tokens
        )

        if self.reference_attention_enabled and shallow_memory is not None:
            reg_tokens, loc = self._reference_conditioned_refine(
                reg_tokens,
                shallow_memory,
                deep_memory,
            )
        else:
            loc_delta = self.loc_head(reg_tokens)
            loc = self._decode_temporal_segments(loc_delta)

        if self.memory_proposal_refine_enabled and shallow_memory is not None:
            loc = self._apply_memory_proposal_refinement(
                loc,
                reg_tokens,
                shallow_memory,
                deep_memory,
            )

        dense_feat = self.cls_feat_proj(feat)
        dense_feat = dense_feat.flatten(2).transpose(1, 2)

        cls_query = self.cls_query_proj(cls_reg_tokens)
        cls_context, _ = self.cls_cross_attn(
            query=cls_query,
            key=dense_feat,
            value=dense_feat
        )

        cls_tokens = self.cls_fuse(cls_query + cls_context)
        cls_logits = self.cls_head(cls_tokens)

        quality_tokens = torch.cat([reg_tokens, cls_context], dim=-1)
        quality_logits = self.conf_head(quality_tokens)

        return torch.cat([quality_logits, loc, cls_logits], dim=-1)

    def forward(
        self,
        x,
        state=None,
        decode=False,
        decode_mask=None,
        active_mask=None,
        detach_state=False,
        dn_targets=None,
    ):
        self._last_memory_auxiliary = None
        legacy_decode_only = False
        if isinstance(state, bool):
            decode = state
            state = None
            legacy_decode_only = True

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

        use_legacy_internal_state = state is None and legacy_decode_only
        if state is None and not use_legacy_internal_state:
            state = self.init_state(x.size(0), device=x.device, dtype=x.dtype)
            explicit_state = False
        elif use_legacy_internal_state:
            state = None
            explicit_state = False
        else:
            explicit_state = True

        if use_legacy_internal_state:
            if self.mem.memory.size(0) != x.size(0):
                shallow_prev = self.mem.init_memory(x.size(0), device=x.device, dtype=x.dtype)
            else:
                shallow_prev = self.mem.memory.to(device=x.device, dtype=x.dtype)
        else:
            shallow_prev = state["shallow"].to(device=x.device, dtype=x.dtype)
        if self.shallow_memory_update_mode == "stateless":
            current = self.mem.mem_generator(x)
            if active_mask is not None:
                active = active_mask.to(device=x.device, dtype=torch.bool).view(
                    -1, 1, 1, 1, 1
                )
                current = torch.where(active, current, shallow_prev)
            x = current
        elif use_legacy_internal_state:
            x = self.mem(x, memory=None, active_mask=active_mask)
        else:
            x = self.mem(x, memory=shallow_prev, active_mask=active_mask)
        shallow_mem_feat = x
        if self.deep_mem_enabled:
            if use_legacy_internal_state:
                if self.deep_mem.memory.size(0) != x.size(0):
                    deep_prev = self.deep_mem.init_memory(x.size(0), device=x.device, dtype=x.dtype)
                else:
                    deep_prev = self.deep_mem.memory.to(device=x.device, dtype=x.dtype)
            else:
                deep_prev = state.get("deep", None)
                if deep_prev is None:
                    deep_prev = self.deep_mem.init_memory(x.size(0), device=x.device, dtype=x.dtype)
                else:
                    deep_prev = deep_prev.to(device=x.device, dtype=x.dtype)
            if self.deep_memory_reset_each_clip:
                deep_prev = self.deep_mem.init_memory(
                    x.size(0), device=x.device, dtype=x.dtype
                )
            deep_input = (
                x - shallow_prev
                if self.deep_memory_input_mode == "residual"
                else x
            )
            if use_legacy_internal_state and not self.deep_memory_reset_each_clip:
                deep_mem_feat = self.deep_mem(
                    deep_input,
                    memory=None,
                    active_mask=active_mask,
                )
            else:
                deep_mem_feat = self.deep_mem(
                    deep_input,
                    memory=deep_prev,
                    active_mask=active_mask,
                )
            self._record_memory_pair_diagnostics(
                shallow_mem_feat,
                deep_mem_feat,
                active_mask=active_mask,
            )
            x = torch.cat([x, deep_mem_feat], dim=1)
        else:
            deep_mem_feat = None

        new_state = {
            "shallow": x[:, :self.memory_size[1]] if self.deep_mem_enabled else x,
            "deep": deep_mem_feat,
        }

        if detach_state:
            new_state = {
                k: v.detach() if v is not None else None
                for k, v in new_state.items()
            }

        if explicit_state:
            # Keep legacy buffers untouched when state is externally managed.
            pass

        if x.dim() == 4:
            x = x.unsqueeze(2)

        should_decode = bool(decode)
        if decode_mask is not None:
            should_decode = bool(torch.as_tensor(decode_mask).any().item())

        output = (
            self._decode_head(
                x,
                shallow_mem_feat,
                deep_mem_feat,
                dn_targets=dn_targets,
            )
            if should_decode
            else None
        )
        if decode_mask is not None and output is not None:
            decode_mask = decode_mask.to(device=output.device, dtype=torch.bool)
            output = output[decode_mask]
            if self._last_memory_auxiliary is not None:
                filtered_auxiliary = {}
                for name, value in self._last_memory_auxiliary.items():
                    filtered_auxiliary[name] = value[decode_mask]
                self._last_memory_auxiliary = filtered_auxiliary

        if legacy_decode_only:
            return output
        return output, new_state

    def pop_memory_auxiliary(self):
        auxiliary = self._last_memory_auxiliary
        self._last_memory_auxiliary = None
        return auxiliary
    
