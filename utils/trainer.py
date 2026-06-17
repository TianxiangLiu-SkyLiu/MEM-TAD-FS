import os
import io
import yaml
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import deque, OrderedDict
import math
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from torchvision import transforms
from utils.DataLoader import load_train_val_data
from models.mem_tad import mem_tad
from tqdm import tqdm
from PIL import Image
import utils.layout as ly
# from utils.losses import MemTADLoss
from utils.losses_improved import MemTADLoss
from utils.model_profile import profile_model_complexity, print_model_complexity
from utils.validator import (
    validate_one_epoch,
    _extract_pred_and_gt,
    _compute_metrics_at_iou,
    _nms_kwargs_from_cfg,
)

# lz4 is optional for cache decompression; if the package isn't available
# the loader will still function for plain .pt caches.
try:
    import lz4.frame
except ImportError:
    lz4 = None


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _to_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _to_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _set_optimizer_lr(optimizer, lr_value):
    for param_group in optimizer.param_groups:
        param_group["lr"] = float(lr_value)


def _get_optimizer_lr(optimizer):
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0].get("lr", 0.0))


def _compute_warmup_lr(base_lr, start_factor, warmup_epochs, warmup_epoch_idx):
    if warmup_epochs <= 0:
        return float(base_lr)

    if warmup_epochs == 1:
        return float(base_lr)

    progress = warmup_epoch_idx / float(max(warmup_epochs - 1, 1))
    factor = float(start_factor) + (1.0 - float(start_factor)) * progress
    return float(base_lr) * factor


class _PriorityFirstSampler(Sampler):
    def __init__(self, dataset_size, first_index, seed=None):
        self.dataset_size = int(dataset_size)
        self.first_index = int(first_index)
        generator = torch.Generator()
        if seed is None:
            seed = torch.seed()
        generator.manual_seed(int(seed))

        indices = torch.randperm(self.dataset_size, generator=generator).tolist()
        if self.first_index in indices:
            indices.remove(self.first_index)
        self.indices = [self.first_index] + indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def _find_video_dataset_index(dataset, target_video_id):
    if target_video_id in {None, ""}:
        return None

    target = str(target_video_id)
    video_list = getattr(dataset, "video_list", None)
    if not video_list:
        return None

    for idx, video_info in enumerate(video_list):
        if str(video_info.get("video_id", "")) == target:
            return idx
    return None


def record_metrics(csv_path, record_data):
    # if csv_path specified, dump per-video records
    if csv_path is not None:
        import csv
        
        csv_path = Path(csv_path)
        if not csv_path.exists():
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = [
                "Epoch", 
                "Train_Loss", "Train_Precision05", "Train_Recall05", "Train_mAP03", "Train_mAP04", "Train_mAP05", "Train_mAP06", "Train_mAP07", "Train_mAP09", "Train_Avg_mAP", 
                "Val_Loss", "Val_Precision05", "Val_Recall05", "Val_mAP03", "Val_mAP04", "Val_mAP05", "Val_mAP06", "Val_mAP07", "Val_mAP09", "Val_Avg_mAP"
            ]
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                row = record_data
                writer.writerow({k: v for k, v in zip(fieldnames, row)})
        else:      
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                row = record_data
                writer.writerow(row)

def _build_cache_version_tag(cfg):
    # view_frames_num removed from tag; caches contain raw frames
    return (
        f"h{cfg['input_size'][0]}_w{cfg['input_size'][1]}_"
        f"sf{cfg['starting_frame_number']}_"
        f"dtype{cfg['cache_pixel_dtype']}_"
        f"norm{int(cfg['cache_normalize'])}"
    )


def _init_dataset_cache_state(cfg):
    state = {
        "enabled": False,
        "root": None,
        "train_dir": None,
        "val_dir": None,
    }

    if not cfg.get("use_dataset_cache", True):
        return state

    cache_dir = str(cfg.get("dataset_cache_dir", "") or "").strip()
    if not cache_dir:
        return state

    base = Path(cache_dir)
    if base.is_dir() and any(base.glob("*.npy")):
        state.update(
            {
                "enabled": True,
                "root": base,
                "mode": "feature_npy_flat",
                "train_dir": None,
                "val_dir": None,
                "metadata": {},
            }
        )
        return state

    tag = _build_cache_version_tag(cfg)
    expected = base / tag
    # also accept prefix variants (lz4_ or pt_) for backwards/format names
    prefixed = [base / f"lz4_{tag}", base / f"pt_{tag}"]
    candidates = [base, expected] + prefixed

    cache_root = None
    for candidate in candidates:
        if (candidate / "train").is_dir() and (candidate / "val").is_dir():
            cache_root = candidate
            break

    # backwards compatibility: if base itself contains a single subdirectory
    # starting with "vf" (old version tag), or a known prefix, use that too.
    if cache_root is None:
        try:
            for child in base.iterdir():
                if child.is_dir():
                    name = child.name
                    if name.startswith("vf") or name.startswith("lz4_") or name.startswith("pt_"):
                        if (child / "train").is_dir() and (child / "val").is_dir():
                            cache_root = child
                            break
        except Exception:
            pass

    if cache_root is None:
        print(
            f"Warning: dataset cache not found under '{base}' (or expected version dir). "
            "Trainer will fallback to image loading."
        )
        return state

    metadata_path = cache_root / "metadata.yaml"
    metadata = {}
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = yaml.safe_load(f) or {}
        except Exception:
            metadata = {}

    state.update(
        {
            "enabled": True,
            "root": cache_root,
            "mode": "legacy_frame_cache",
            "train_dir": cache_root / "train",
            "val_dir": cache_root / "val",
            "metadata": metadata,
        }
    )
    return state


def _to_cpu_tensor(x):
    if torch.is_tensor(x):
        return x.detach().cpu()
    return torch.tensor(x)


def _extract_scalar_int(value):
    if torch.is_tensor(value):
        if value.numel() == 1:
            return int(value.item())
        return int(value[0].item())
    if isinstance(value, (list, tuple)) and value:
        return int(value[0])
    return int(value)


def _canonicalize_cached_clips(clips):
    if isinstance(clips, list):
        return [clip.detach().cpu() if torch.is_tensor(clip) else torch.tensor(clip) for clip in clips]

    if torch.is_tensor(clips):
        clips = clips.detach().cpu()
        if clips.dim() == 4:
            return [clips[i : i + 1] for i in range(clips.size(0))]
        if clips.dim() == 5:
            return [clips[i : i + 1] for i in range(clips.size(0))]
        if clips.dim() == 6:
            return [clips[i] for i in range(clips.size(0))]

    raise ValueError("Unsupported cached clip format. Expected list[tensor] or 4D/5D/6D tensor.")


def _canonicalize_feature_clips(features):
    if torch.is_tensor(features):
        features = features.detach().cpu()
    else:
        features = torch.from_numpy(np.asarray(features))

    if features.dim() == 2:
        # [H, W] -> [1, 1, 1, H, W]
        features = features.unsqueeze(0).unsqueeze(0).unsqueeze(0)

    if features.dim() == 3:
        # InternVideo2 2D map sequence: [N, H, W] -> [N, 1, 1, H, W]
        features = features.unsqueeze(1).unsqueeze(2)

    if features.dim() == 4:
        # [N, C, H, W] -> [N, C, 1, H, W]
        features = features.unsqueeze(2)

    if features.dim() != 5:
        raise ValueError(
            "feature tensor must be one of [N, C, T, H, W], [N, C, H, W], [N, H, W], [H, W], "
            f"got {tuple(features.shape)}"
        )

    features = features.to(torch.float32).contiguous()
    return [features[i : i + 1] for i in range(features.size(0))]


def _load_feature_file(feature_path):
    suffix = feature_path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(str(feature_path), allow_pickle=False)
        return _canonicalize_feature_clips(arr)
    if suffix == ".pt":
        payload = torch.load(str(feature_path), map_location="cpu")
        if isinstance(payload, dict):
            if "features" in payload:
                return _canonicalize_feature_clips(payload["features"])
            if "clips" in payload:
                return _canonicalize_feature_clips(payload["clips"])
        return _canonicalize_feature_clips(payload)
    raise ValueError(f"Unsupported feature file format: {feature_path}")


def _infer_feature_size_from_cache_dir(cache_dir):
    base = Path(cache_dir)
    if not base.exists():
        return None, None

    candidate_paths = []
    for pattern in ("*.npy", "*.pt"):
        candidate_paths = sorted(base.rglob(pattern))
        if candidate_paths:
            break

    for feature_path in candidate_paths:
        try:
            clips = _load_feature_file(feature_path)
        except Exception:
            continue

        if not clips:
            continue

        sample_clip = clips[0]
        if not torch.is_tensor(sample_clip) or sample_clip.dim() != 5:
            continue

        _, channels, temporal, height, width = sample_clip.shape
        return [int(channels), int(temporal), int(height), int(width)], feature_path

    return None, None


def _ensure_bcthw(clip_tensor):
    # strict target layout: [B, C, T, H, W]
    if clip_tensor.dim() != 5 or clip_tensor.size(1) != 3:
        raise ValueError(
            f"clip tensor must be [B, C, T, H, W] with C=3, got {tuple(clip_tensor.shape)}"
        )
    return clip_tensor.contiguous()


def _apply_imagenet_normalization(clip_tensor):
    # strict input layout: [B, C, T, H, W]
    if clip_tensor.dim() != 5 or clip_tensor.size(1) != 3:
        raise ValueError(
            f"normalization expects [B, C, T, H, W] with C=3, got {tuple(clip_tensor.shape)}"
        )

    x = clip_tensor.permute(0, 2, 1, 3, 4)

    mean = clip_tensor.new_tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
    std = clip_tensor.new_tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)
    x = (x - mean) / std
    return x.permute(0, 2, 1, 3, 4).contiguous()


def _frames_to_clips(frames, cfg, zero_frame):
    """Slice a full-frame tensor into clip tensors based on cfg.view_frames_num.

    frames: [T,3,H,W] CPU tensor.
    returns list of [1,3,L,H,W] tensors where L==view_frames_num (padding with
    zero_frame at end if needed).
    """
    clips = []
    T = frames.size(0)
    vfn = cfg["view_frames_num"]
    h, w = cfg["input_size"]
    for start in range(0, T, vfn):
        end = start + vfn
        chunk = frames[start:end]
        if chunk.size(0) < vfn:
            pad_n = vfn - chunk.size(0)
            pad = zero_frame.unsqueeze(0).repeat(pad_n, 1, 1, 1)
            chunk = torch.cat([chunk, pad], dim=0)
        clip_tensor = chunk.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()
        clips.append(clip_tensor)
    return clips


def decode_clip_for_model(clip_tensor, video_data, cfg):
    if video_data.get("source") == "feature_cache":
        return clip_tensor.to(torch.float32).contiguous()

    if video_data.get("source") != "cache":
        return clip_tensor

    cache_dtype = str(video_data.get("cache_pixel_dtype", "uint8")).lower()
    cache_normalize = _to_bool(video_data.get("cache_normalize", False))

    if cache_dtype == "uint8":
        clip_tensor = clip_tensor.to(torch.float32) / 255.0
    elif cache_dtype == "int8":
        # int8 cache stores uint8 pixels with offset 128.
        clip_tensor = (clip_tensor.to(torch.float32) + 128.0) / 255.0
    elif cache_dtype == "float32":
        clip_tensor = clip_tensor.to(torch.float32)
        if not cache_normalize:
            clip_tensor = clip_tensor / 255.0
    else:
        raise ValueError(f"Unsupported cache pixel dtype: {cache_dtype}")

    if _to_bool(cfg.get("cache_decode_apply_imagenet_norm", True)):
        clip_tensor = _apply_imagenet_normalization(clip_tensor)

    return clip_tensor.contiguous()


def build_zero_frame(cfg):
    h, w = cfg["input_size"]
    zero = torch.zeros(3, h, w, dtype=torch.float32)
    mean = zero.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = zero.new_tensor(IMAGENET_STD).view(3, 1, 1)
    return ((zero - mean) / std).contiguous()


def check_config(cfg, device):
    if cfg is None:
        print("Config file is empty or not properly formatted.")
        return False
    elif cfg.get("annotations_json_path") is None:
        print("Config file is missing required field: 'annotations_json_path'.")
        return False
    elif cfg.get("frames_dir") is None and cfg.get("dataset_cache_dir") is None:
        print("Config file requires at least one of: 'dataset_cache_dir' or 'frames_dir'.")
        return False
    else:
        if not cfg.get("encoder"):
            cfg["encoder"] = 'resnet50'
        if not cfg.get("memory_size"):
            cfg["memory_size"] = 'm'
        if not cfg.get("lr"):
            cfg["lr"] = 0.0001
        if not cfg.get("batch_size"):
            cfg["batch_size"] = 1
        if not cfg.get("input_size"):
            cfg["input_size"] = (720, 1280)
        if not cfg.get("output_dir"):
            cfg["output_dir"] = './result'
            if not Path(cfg["output_dir"]).exists() or not Path(cfg["output_dir"]).is_dir():
                cfg["output_dir"] = './result/1'
                os.makedirs(cfg["output_dir"])
            else:
                numeric_dirs = []
                for item in Path(cfg["output_dir"]).iterdir():
                    if item.is_dir() and item.name.isdigit():
                        numeric_dirs.append(item.name)

                numeric_dirs = sorted(numeric_dirs, key=lambda x: int(x))
                if not numeric_dirs:
                    cfg["output_dir"] = './result/1'
                    os.makedirs(cfg["output_dir"], exist_ok=True)
                elif not any(Path(f'./result/{numeric_dirs[-1]}').iterdir()):
                    cfg["output_dir"] = f'./result/{numeric_dirs[-1]}'
                else:
                    cfg["output_dir"] = f'./result/{int(numeric_dirs[-1]) + 1}'
                    os.mkdir(cfg["output_dir"])
        if not Path(cfg["output_dir"]).exists() or not Path(cfg["output_dir"]).is_dir():
            print(f"Output directory {cfg['output_dir']} does not exist. Creating it.")
            os.makedirs(cfg["output_dir"], exist_ok=True)

        cfg["input_size"] = [int(cfg["input_size"][0]), int(cfg["input_size"][1])]
        cfg["view_frames_num"] = _to_int(cfg.get("view_frames_num", 30), 30)
        cfg["starting_frame_number"] = _to_int(cfg.get("starting_frame_number", 1), 1)
        cfg["batch_size"] = _to_int(cfg.get("batch_size", 1), 1)
        cfg["first_epoch_first_video_id"] = str(cfg.get("first_epoch_first_video_id", "") or "").strip()
        cfg["lr"] = _to_float(cfg.get("lr", 1e-4), 1e-4)
        cfg["weight_decay"] = max(_to_float(cfg.get("weight_decay", 1e-4), 1e-4), 0.0)
        cfg["resume_use_cfg_lr"] = _to_bool(cfg.get("resume_use_cfg_lr", True))
        cfg["resume_use_cfg_plateau_min_lr"] = _to_bool(
            cfg.get("resume_use_cfg_plateau_min_lr", True)
        )
        cfg["lr_warmup_enabled"] = _to_bool(cfg.get("lr_warmup_enabled", True))
        cfg["lr_warmup_epochs"] = max(_to_int(cfg.get("lr_warmup_epochs", 5), 5), 0)
        cfg["lr_warmup_start_factor"] = max(
            min(_to_float(cfg.get("lr_warmup_start_factor", 0.1), 0.1), 1.0),
            1e-4,
        )
        cfg["lr_plateau_enabled"] = _to_bool(cfg.get("lr_plateau_enabled", True))
        cfg["lr_plateau_monitor"] = str(cfg.get("lr_plateau_monitor", "map05") or "map05").lower()
        if cfg["lr_plateau_monitor"] not in {
            "loss", "map03", "map04", "map05", "map06", "map07", "map09", "avg_map"
        }:
            cfg["lr_plateau_monitor"] = "map05"
        cfg["lr_plateau_mode"] = str(cfg.get("lr_plateau_mode", "max") or "max").lower()
        if cfg["lr_plateau_mode"] not in {"min", "max"}:
            cfg["lr_plateau_mode"] = "max"
        cfg["lr_plateau_factor"] = max(
            min(_to_float(cfg.get("lr_plateau_factor", 0.5), 0.5), 0.99),
            0.01,
        )
        cfg["lr_plateau_patience"] = max(_to_int(cfg.get("lr_plateau_patience", 3), 3), 0)
        cfg["lr_plateau_threshold"] = max(
            _to_float(cfg.get("lr_plateau_threshold", 1e-4), 1e-4),
            0.0,
        )
        cfg["lr_plateau_threshold_mode"] = str(
            cfg.get("lr_plateau_threshold_mode", "abs") or "abs"
        ).lower()
        if cfg["lr_plateau_threshold_mode"] not in {"rel", "abs"}:
            cfg["lr_plateau_threshold_mode"] = "abs"
        cfg["lr_plateau_cooldown"] = max(_to_int(cfg.get("lr_plateau_cooldown", 0), 0), 0)
        cfg["lr_plateau_min_lr"] = max(
            _to_float(cfg.get("lr_plateau_min_lr", 1e-6), 1e-6),
            0.0,
        )
        cfg["lr_plateau_eps"] = max(_to_float(cfg.get("lr_plateau_eps", 1e-8), 1e-8), 0.0)
        cfg["resnet_width_mult"] = max(min(_to_float(cfg.get("resnet_width_mult", 1.0), 1.0), 1.0), 0.25)
        cfg["resnet_temporal_kernel"] = _to_int(cfg.get("resnet_temporal_kernel", 5), 5)
        if cfg["resnet_temporal_kernel"] < 1:
            cfg["resnet_temporal_kernel"] = 1
        if cfg["resnet_temporal_kernel"] % 2 == 0:
            cfg["resnet_temporal_kernel"] += 1
        cfg["backbone_memory_fusion_enabled"] = _to_bool(
            cfg.get("backbone_memory_fusion_enabled", False)
        )
        fusion_layers_raw = cfg.get("backbone_memory_fusion_layers", ["layer3", "layer4"])
        if isinstance(fusion_layers_raw, str):
            fusion_layers = [s.strip() for s in fusion_layers_raw.split(",") if s.strip()]
        elif isinstance(fusion_layers_raw, (list, tuple)):
            fusion_layers = [str(s).strip() for s in fusion_layers_raw if str(s).strip()]
        else:
            fusion_layers = ["layer3", "layer4"]
        allowed_fusion_layers = {"layer1", "layer2", "layer3", "layer4"}
        cfg["backbone_memory_fusion_layers"] = [
            name for name in fusion_layers if name in allowed_fusion_layers
        ]
        if not cfg["backbone_memory_fusion_layers"]:
            cfg["backbone_memory_fusion_layers"] = ["layer3", "layer4"]
        cfg["backbone_memory_fusion_init_gate"] = _to_float(
            cfg.get("backbone_memory_fusion_init_gate", -2.0), -2.0
        )

        cfg["async_prefetch_next_video"] = _to_bool(cfg.get("async_prefetch_next_video", True))
        cfg["prefetch_videos_ahead"] = max(_to_int(cfg.get("prefetch_videos_ahead", 2), 2), 0)
        cfg["prefetch_workers"] = max(_to_int(cfg.get("prefetch_workers", 1), 1), 1)

        cfg["dataset_cache_dir"] = str(cfg.get("dataset_cache_dir", "./cache") or "./cache")
        cfg["use_dataset_cache"] = _to_bool(cfg.get("use_dataset_cache", True))
        cfg["cache_pixel_dtype"] = str(cfg.get("cache_pixel_dtype", "uint8")).lower()
        cfg["cache_normalize"] = _to_bool(cfg.get("cache_normalize", False))
        cfg["cache_decode_apply_imagenet_norm"] = _to_bool(
            cfg.get("cache_decode_apply_imagenet_norm", True)
        )

        # whether to push preprocessing (resize/normalize/decode) onto GPU
        cfg["preprocess_on_gpu"] = _to_bool(cfg.get("preprocess_on_gpu", False))
        cfg["dataloader_num_workers"] = max(_to_int(cfg.get("dataloader_num_workers", 8), 8), 0)
        cfg["dataloader_pin_memory"] = _to_bool(cfg.get("dataloader_pin_memory", True))

        # model head regularization
        cfg["head_channels"] = max(_to_int(cfg.get("head_channels", 0), cfg.get("max_detection_num", 128)), 32)
        # allow shrinking head hidden dim for lower VRAM; keep a safe minimum
        cfg["det_head_hidden_dim"] = max(_to_int(cfg.get("det_head_hidden_dim", 256), 64), 64)
        cfg["head_dropout"] = max(min(_to_float(cfg.get("head_dropout", 0.2), 0.2), 0.8), 0.0)
        cfg["head_token_dropout"] = max(
            min(_to_float(cfg.get("head_token_dropout", cfg["head_dropout"]), cfg["head_dropout"]), 0.8),
            0.0,
        )
        cfg["pre_head_memory_refine_enabled"] = _to_bool(
            cfg.get("pre_head_memory_refine_enabled", False)
        )
        cfg["pre_head_memory_refine_dropout"] = max(
            min(
                _to_float(
                    cfg.get("pre_head_memory_refine_dropout", cfg["head_dropout"]),
                    cfg["head_dropout"],
                ),
                0.8,
            ),
            0.0,
        )
        cfg["pre_head_memory_refine_kernel_t"] = _to_int(
            cfg.get("pre_head_memory_refine_kernel_t", 3),
            3,
        )
        if cfg["pre_head_memory_refine_kernel_t"] < 1:
            cfg["pre_head_memory_refine_kernel_t"] = 1
        if cfg["pre_head_memory_refine_kernel_t"] % 2 == 0:
            cfg["pre_head_memory_refine_kernel_t"] += 1
        cfg["memory_dropout"] = max(min(_to_float(cfg.get("memory_dropout", 0.1), 0.1), 0.8), 0.0)

        # loss algorithm options
        cfg["loss_focal_gamma"] = max(_to_float(cfg.get("loss_focal_gamma", 2.0), 2.0), 0.0)
        cfg["loss_focal_alpha"] = max(min(_to_float(cfg.get("loss_focal_alpha", 0.25), 0.25), 0.95), 0.01)
        cfg["loss_neg_pos_ratio"] = max(_to_float(cfg.get("loss_neg_pos_ratio", 3.0), 3.0), 1.0)
        cfg["loss_iou_weight"] = max(_to_float(cfg.get("loss_iou_weight", 0.5), 0.5), 0.0)
        cfg["loss_label_smoothing"] = max(min(_to_float(cfg.get("loss_label_smoothing", 0.05), 0.05), 0.2), 0.0)

        # train/eval stability
        cfg["grad_clip_norm"] = max(_to_float(cfg.get("grad_clip_norm", 5.0), 5.0), 0.0)
        cfg["eval_conf_threshold"] = max(min(_to_float(cfg.get("eval_conf_threshold", 0.05), 0.05), 1.0), 0.0)
        # postprocess temporal NMS for evaluation/validation
        cfg["postprocess_nms_enabled"] = _to_bool(cfg.get("postprocess_nms_enabled", False))
        cfg["postprocess_nms_type"] = str(cfg.get("postprocess_nms_type", "hard") or "hard").strip().lower()
        if cfg["postprocess_nms_type"] not in {"hard", "soft"}:
            cfg["postprocess_nms_type"] = "hard"
        cfg["postprocess_nms_iou_threshold"] = max(
            min(_to_float(cfg.get("postprocess_nms_iou_threshold", 0.5), 0.5), 1.0),
            0.0,
        )
        cfg["postprocess_nms_sigma"] = max(_to_float(cfg.get("postprocess_nms_sigma", 0.5), 0.5), 1e-6)
        cfg["postprocess_nms_min_score"] = max(
            min(_to_float(cfg.get("postprocess_nms_min_score", 1e-4), 1e-4), 1.0),
            0.0,
        )
        cfg["postprocess_nms_max_detections"] = max(
            _to_int(cfg.get("postprocess_nms_max_detections", cfg.get("max_detection_num", 150)), cfg.get("max_detection_num", 150)),
            1,
        )
        cfg["amp_enabled"] = _to_bool(cfg.get("amp_enabled", True)) and torch.cuda.is_available()
        cfg["amp_dtype"] = str(cfg.get("amp_dtype", "fp16")).lower()
        if cfg["amp_dtype"] not in {"fp16", "float16", "bf16", "bfloat16"}:
            cfg["amp_dtype"] = "fp16"

        if cfg["cache_pixel_dtype"] not in {"int8", "uint8", "float32"}:
            raise ValueError("cache_pixel_dtype must be one of: int8, uint8, float32")

        cfg["device"] = torch.device(device if torch.cuda.is_available() else "cpu")

        # Feature input compatibility:
        # - canonical model shape is [C, T, H, W]
        # - infer it from the first cached feature file whenever possible.
        feature_size = cfg.get("feature_size", None)
        inferred_feature_size, inferred_feature_path = _infer_feature_size_from_cache_dir(
            cfg["dataset_cache_dir"]
        )
        if inferred_feature_size is not None:
            cfg["feature_size"] = inferred_feature_size
            print(
                "[INFO] Inferred cfg['feature_size']={} from {}".format(
                    cfg["feature_size"],
                    inferred_feature_path,
                )
            )
        elif isinstance(feature_size, (list, tuple)) and len(feature_size) == 4:
            cfg["feature_size"] = [int(v) for v in feature_size]
        if isinstance(feature_size, (list, tuple)):
            if len(feature_size) == 3 and inferred_feature_size is None:
                c, h, w = feature_size
                cfg["feature_size"] = [int(c), 1, int(h), int(w)]
                print(
                    "[INFO] Interpreting cfg['feature_size']={} as [C,H,W] -> [C,T,H,W]={} "
                    "because no cached feature file was found for inference.".format(
                        list(feature_size),
                        cfg["feature_size"],
                    )
                )

    return cfg


def build_video_clips(video_folder, total_frames, cfg, transform, zero_frame):
    """Load one video's frames and package them into clip tensors on CPU."""
    num_clips = math.ceil(total_frames / cfg["view_frames_num"])
    clips = []

    for clip_idx in range(num_clips):
        frame_tensors = []
        start_idx = clip_idx * cfg["view_frames_num"]

        for sub_f in range(cfg["view_frames_num"]):
            frame_idx = start_idx + sub_f
            frame_path = os.path.join(
                video_folder,
                f"{frame_idx + cfg['starting_frame_number']:06d}.jpg",
            )

            try:
                with Image.open(frame_path) as img:
                    frame_tensors.append(transform(img.convert('RGB')))
            except (FileNotFoundError, OSError):
                frame_tensors.append(zero_frame)

        clip_tensor = torch.stack(frame_tensors, dim=0).unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()
        clips.append(clip_tensor)

    return clips


def prepare_video_batch(batch_train, cfg, transform, zero_frame, cache_state, split_name):
    """Prepare one dataloader batch into per-video clips for model consumption."""
    t_start = time.time()
    video_folder = batch_train.get('video_folder', [None])[0]
    feature_path_from_dataset = batch_train.get('feature_path', [None])[0]
    labels = _to_cpu_tensor(batch_train['labels'][0])
    total_frames = _extract_scalar_int(batch_train['total_frames'][0])
    video_id = batch_train['video_id'][0]

    # preferred path: pre-extracted feature files from dataset_cache_dir
    if cache_state.get("enabled", False) and cache_state.get("mode") == "feature_npy_flat":
        root = cache_state.get("root")
        candidates = []
        if feature_path_from_dataset:
            candidates.append(Path(str(feature_path_from_dataset)))
        candidates.append(Path(root) / f"{video_id}.npy")
        candidates.append(Path(root) / f"{video_id}.pt")

        feature_path = None
        for cand in candidates:
            if cand.exists():
                feature_path = cand
                break

        if feature_path is None:
            raise FileNotFoundError(
                f"Feature file not found for video_id={video_id} under {root}. "
                f"Expected {video_id}.npy (or .pt)."
            )

        clips = _load_feature_file(feature_path)
        expected = cfg.get("feature_size", None)
        if isinstance(expected, (list, tuple)) and len(expected) == 4 and clips:
            ec, et, eh, ew = [int(v) for v in expected]
            c = int(clips[0].size(1))
            t = int(clips[0].size(2))
            h = int(clips[0].size(3))
            w = int(clips[0].size(4))
            if (c, t, h, w) != (ec, et, eh, ew):
                raise ValueError(
                    "Feature shape mismatch for model config: "
                    f"expected [C,T,H,W]=[{ec},{et},{eh},{ew}], got [{c},{t},{h},{w}] from {feature_path}"
                )

        t_read = time.time() - t_start
        t_proc = 0.0
        return {
            "video_id": video_id,
            "labels": labels,
            "video_clips": clips,
            "source": "feature_cache",
            "feature_path": str(feature_path),
            "cache_pixel_dtype": None,
            "cache_normalize": False,
            "read_time": t_read,
            "proc_time": t_proc,
        }

    # attempt cache lookup
    if cache_state.get("enabled", False):
        split_dir = cache_state.get(f"{split_name}_dir")
        if split_dir is not None:
            # look for either plain torch cache or an lz4-compressed file
            cache_path = split_dir / f"{video_id}.pt"
            lz4_path = split_dir / f"{video_id}.lz4"
            payload = None
            cache_format = None
            if cache_path.exists():
                cache_format = "pt"
                payload = torch.load(cache_path, map_location="cpu")
            elif lz4_path.exists():
                cache_format = "lz4"
                if lz4 is None:
                    raise RuntimeError(
                        "found lz4 cache but 'lz4' package is not installed"
                    )
                buf = lz4_path.read_bytes()
                try:
                    decompressed = lz4.frame.decompress(buf)
                except Exception as e:
                    raise RuntimeError(f"failed to decompress {lz4_path}: {e}")
                payload = torch.load(io.BytesIO(decompressed), map_location="cpu")

            if payload is not None:
                labels = _to_cpu_tensor(payload.get("labels", labels))
                metadata = cache_state.get("metadata", {}) if isinstance(cache_state, dict) else {}
                fallback_dtype = str(metadata.get("cache_pixel_dtype", cfg["cache_pixel_dtype"])).lower()
                fallback_normalize = _to_bool(metadata.get("cache_normalize", cfg["cache_normalize"]))

                # derive clips from full frames if provided
                if "frames" in payload:
                    frames = payload["frames"].detach().cpu()
                    clips = _frames_to_clips(frames, cfg, zero_frame)
                else:
                    clips = _canonicalize_cached_clips(payload.get("clips", []))
                    clips = [_ensure_bcthw(clip) for clip in clips]
                t_read = time.time() - t_start

                # optionally push clips/labels to device and apply decoding there
                cache_info = {
                    "source": "cache",
                    "cache_format": cache_format,
                    "cache_pixel_dtype": str(payload.get("cache_pixel_dtype", fallback_dtype)).lower(),
                    "cache_normalize": _to_bool(payload.get("cache_normalize", fallback_normalize)),
                }

                t_proc = time.time() - t_start - t_read
                return {
                    "video_id": video_id,
                    "labels": labels,
                    "video_clips": clips,
                    "source": "cache",
                    "cache_format": cache_info.get("cache_format"),
                    "cache_pixel_dtype": cache_info["cache_pixel_dtype"],
                    "cache_normalize": cache_info["cache_normalize"],
                    "read_time": t_read,
                    "proc_time": t_proc,
                }

    # no cache available
    t_read = time.time() - t_start
    video_clips = build_video_clips(video_folder, total_frames, cfg, transform, zero_frame)
    t_read = time.time() - t_start
    t_proc = time.time() - t_start - t_read
    return {
        "video_id": video_id,
        "labels": labels,
        "video_clips": video_clips,
        "source": "frames",
        "cache_pixel_dtype": None,
        "cache_normalize": False,
        "read_time": t_read,
        "proc_time": t_proc,
    }


def train_one_video(model, optimizer, criterion, video_data, cfg, scaler=None):
    """Run forward/backward for one prepared video.

    Returns a tuple ``(loss_value, record)`` where ``record`` is a
    dictionary suitable for metric computation by the validator helpers.
    """
    labels = video_data["labels"].to(cfg["device"])
    video_clips = video_data["video_clips"]

    pbar_in = tqdm(total=len(video_clips), desc="Video Progress: ", leave=False)
    if len(video_clips) == 0:
        pbar_in.close()
        model.reset_memory()
        # create an empty record so calling code can still compute metrics
        rec = {
            "pred_conf": torch.empty((0,)),
            "pred_seg": torch.empty((0, 2)),
            "pred_cls": torch.empty((0,), dtype=torch.long),
            "gt_seg": torch.empty((0, 2)),
            "gt_cls": torch.empty((0,), dtype=torch.long),
            "video_id": video_data.get("video_id", ""),
        }
        return 0.0, rec

    amp_enabled = bool(cfg.get("amp_enabled", False))
    amp_dtype = torch.float16
    if str(cfg.get("amp_dtype", "fp16")).lower() in {"bf16", "bfloat16"}:
        amp_dtype = torch.bfloat16

    output = None
    with torch.amp.autocast(device_type="cuda", enabled=amp_enabled, dtype=amp_dtype):
        for clip_idx, clip_tensor in enumerate(video_clips):
            clip_tensor = clip_tensor.to(cfg["device"], non_blocking=True)
            clip_tensor = decode_clip_for_model(clip_tensor, video_data, cfg)
            if video_data.get("source") not in {"cache", "feature_cache"}:
                clip_tensor = _apply_imagenet_normalization(clip_tensor)
            output = model(clip_tensor, (clip_idx + 1 == len(video_clips)))
            pbar_in.update(1)

        if output is None or (not torch.isfinite(output).all()):
            pbar_in.close()
            optimizer.zero_grad(set_to_none=True)
            model.reset_memory()
            rec = {
                "pred_conf": torch.empty((0,)),
                "pred_seg": torch.empty((0, 2)),
                "pred_cls": torch.empty((0,), dtype=torch.long),
                "gt_seg": torch.empty((0, 2)),
                "gt_cls": torch.empty((0,), dtype=torch.long),
                "video_id": video_data.get("video_id", ""),
                "skipped_non_finite": True,
            }
            return 0.0, rec

        loss = criterion(output, labels)
    pbar_in.close()

    if not torch.isfinite(loss):
        optimizer.zero_grad(set_to_none=True)
        model.reset_memory()
        rec = {
            "pred_conf": torch.empty((0,)),
            "pred_seg": torch.empty((0, 2)),
            "pred_cls": torch.empty((0,), dtype=torch.long),
            "gt_seg": torch.empty((0, 2)),
            "gt_cls": torch.empty((0,), dtype=torch.long),
            "video_id": video_data.get("video_id", ""),
            "skipped_non_finite": True,
        }
        return 0.0, rec

    optimizer.zero_grad(set_to_none=True)
    if scaler is not None and scaler.is_enabled():
        scaler.scale(loss).backward()
        if cfg.get("grad_clip_norm", 0.0) > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg["grad_clip_norm"])
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if cfg.get("grad_clip_norm", 0.0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg["grad_clip_norm"])
        optimizer.step()

    # build metric record using same helper as validation
    rec = _extract_pred_and_gt(
        output,
        labels,
        cfg["class_num"],
        conf_threshold=float(cfg.get("eval_conf_threshold", 0.0)),
        **_nms_kwargs_from_cfg(cfg),
    )
    rec["video_id"] = video_data.get("video_id", "")
    rec["skipped_non_finite"] = False

    model.reset_memory()
    return loss.item(), rec




def train_model(args):
    with open(args.cfg, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    cfg = check_config(cfg, args.device)
    if cfg == False:
        return False
    
    cache_state = _init_dataset_cache_state(cfg)

    # keep queued samples on CPU and do normalization on GPU right before forward
    transform = transforms.Compose([
        transforms.Resize((cfg["input_size"][0], cfg["input_size"][1])),
        transforms.ToTensor(),
    ])

    train_dataset, val_dataset = load_train_val_data(
        frames_dir=cfg.get("frames_dir", None),
        json_path=cfg["annotations_json_path"],
        device="cpu",
        features_dir=cfg.get("dataset_cache_dir", None),
    )

    cfg["class_num"] = train_dataset.class_num
    train_video_num = len(train_dataset.video_list)
    val_video_num = len(val_dataset.video_list)
    
    ly.box(f"{cfg['dataset_cache_dir'].split('/')[-1]}\nNumber of training videos: {train_video_num}, Number of training frames: {train_dataset.dataset_totle_frame}\nNumber of validation videos: {val_video_num}, Number of validation frames: {val_dataset.dataset_totle_frame}")

    first_epoch_first_video_idx = _find_video_dataset_index(
        train_dataset,
        cfg.get("first_epoch_first_video_id", ""),
    )

    def _build_train_dataloader(prioritize_first_batch=False, epoch_seed=0):
        if prioritize_first_batch and first_epoch_first_video_idx is not None:
            sampler = _PriorityFirstSampler(
                dataset_size=len(train_dataset),
                first_index=first_epoch_first_video_idx,
                seed=epoch_seed,
            )
            return DataLoader(
                train_dataset,
                batch_size=cfg["batch_size"],
                shuffle=False,
                sampler=sampler,
                num_workers=cfg["dataloader_num_workers"],
                pin_memory=cfg["dataloader_pin_memory"],
            )

        return DataLoader(
            train_dataset,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=cfg["dataloader_num_workers"],
            pin_memory=cfg["dataloader_pin_memory"],
        )

    dataloader_train = _build_train_dataloader(prioritize_first_batch=False)
    dataloader_val = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["dataloader_num_workers"],
        pin_memory=cfg["dataloader_pin_memory"],
    )

    model = mem_tad(cfg)
    model = model.to(cfg["device"])
    model_profile = profile_model_complexity(model, cfg, cfg["device"])
    print_model_complexity(model_profile, prefix="Train")
    criterion = MemTADLoss(
        class_num=cfg["class_num"],
        conf_weight=cfg.get("loss_conf_weight", cfg.get("conf_weight", 1.0)),
        loc_weight=cfg.get("loss_loc_weight", cfg.get("loc_weight", 2.0)),
        cls_weight=cfg.get("loss_cls_weight", cfg.get("cls_weight", 1.0)),
        match_iou_threshold=cfg.get("loss_match_iou_threshold", cfg.get("match_iou_threshold", 0.1)),
        focal_gamma=cfg.get("loss_focal_gamma", 2.0),
        focal_alpha=cfg.get("loss_focal_alpha", 0.25),
        neg_pos_ratio=cfg.get("loss_neg_pos_ratio", 3.0),
        iou_loc_weight=cfg.get("loss_iou_weight", 0.5),
        label_smoothing=cfg.get("loss_label_smoothing", 0.05),
        eps=cfg.get("loss_eps", cfg.get("eps", 1e-6))
    )
    optimizer_cls = torch.optim.__dict__[cfg["optimizer"]]
    optimizer_kwargs = {
        "lr": cfg["lr"],
        "weight_decay": cfg.get("weight_decay", 0.0),
    }
    try:
        optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)
    except TypeError:
        optimizer_kwargs.pop("weight_decay", None)
        optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)
        print(
            f"Warning: optimizer '{cfg['optimizer']}' does not accept weight_decay; "
            "falling back without weight decay."
        )
    base_lr = float(cfg["lr"])
    if cfg.get("lr_warmup_enabled", False) and cfg.get("lr_warmup_epochs", 0) > 0:
        _set_optimizer_lr(
            optimizer,
            _compute_warmup_lr(
                base_lr,
                cfg["lr_warmup_start_factor"],
                cfg["lr_warmup_epochs"],
                0,
            ),
        )
    plateau_scheduler = None
    if cfg.get("lr_plateau_enabled", False):
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=cfg["lr_plateau_mode"],
            factor=cfg["lr_plateau_factor"],
            patience=cfg["lr_plateau_patience"],
            threshold=cfg["lr_plateau_threshold"],
            threshold_mode=cfg["lr_plateau_threshold_mode"],
            cooldown=cfg["lr_plateau_cooldown"],
            min_lr=cfg["lr_plateau_min_lr"],
            eps=cfg["lr_plateau_eps"],
        )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.get("amp_enabled", False))
    zero_frame = build_zero_frame(cfg)

    # checkpoint / resume support
    start_epoch = 0
    best_avg_map = 0.0
    if hasattr(args, "resume") and args.resume:
        resume_path = args.resume
        if os.path.isfile(resume_path):
            print(f"Loading checkpoint from {resume_path}")
            ckpt = torch.load(resume_path, map_location=cfg["device"])
            model.load_state_dict(ckpt.get("model", ckpt.get("state_dict", {})))
            optimizer.load_state_dict(ckpt.get("optimizer", {}))
            if cfg.get("resume_use_cfg_lr", True):
                resumed_lr = _get_optimizer_lr(optimizer)
                _set_optimizer_lr(optimizer, base_lr)
                print(
                    f">>> Resume LR override enabled: ckpt_lr={resumed_lr:.8f} -> cfg_lr={base_lr:.8f}"
                )
            if plateau_scheduler is not None and ckpt.get("plateau_scheduler"):
                plateau_scheduler.load_state_dict(ckpt["plateau_scheduler"])
                if cfg.get("resume_use_cfg_plateau_min_lr", True):
                    old_min_lrs = list(getattr(plateau_scheduler, "min_lrs", []))
                    target_min_lr = float(cfg.get("lr_plateau_min_lr", 1e-6))
                    plateau_scheduler.min_lrs = [target_min_lr] * len(optimizer.param_groups)
                    print(
                        f">>> Resume Plateau min_lr override enabled: "
                        f"ckpt_min_lrs={old_min_lrs} -> cfg_min_lr={target_min_lr:.8f}"
                    )
            start_epoch = ckpt.get("epoch", 0) + 1
            if "best_avg_map" in ckpt:
                best_avg_map = float(ckpt["best_avg_map"])
            elif "best_map" in ckpt:
                best_avg_map = float(ckpt["best_map"])
            print(f"Resumed from epoch {start_epoch}, best_avg_map={best_avg_map:.6f}")
        else:
            print(f"Warning: resume file {resume_path} not found, starting from scratch.")

    print(f">>> Output Path: {cfg['output_dir']}\nStarting training...")
    if cache_state.get("enabled", False):
        print(f">>> Dataset cache enabled: {cache_state['root']}")
    else:
        print(">>> Dataset cache disabled or unavailable; using on-the-fly frame loading.")
    if cfg.get("first_epoch_first_video_id"):
        if first_epoch_first_video_idx is not None:
            print(
                f">>> First epoch priority video enabled: {cfg['first_epoch_first_video_id']}"
            )
        else:
            print(
                f"Warning: first_epoch_first_video_id={cfg['first_epoch_first_video_id']} "
                "not found in training set; normal shuffle will be used."
            )
    print(f">>> Optimizer: {cfg['optimizer']} | weight_decay={cfg.get('weight_decay', 0.0):.6g}")
    if cfg.get("lr_warmup_enabled", False) and cfg.get("lr_warmup_epochs", 0) > 0:
        print(
            f">>> LR warmup enabled: epochs={cfg['lr_warmup_epochs']}, "
            f"start_factor={cfg['lr_warmup_start_factor']:.3f}"
        )
    if cfg.get("lr_plateau_enabled", False):
        print(
            f">>> LR plateau enabled: monitor={cfg['lr_plateau_monitor']}, "
            f"mode={cfg['lr_plateau_mode']}, factor={cfg['lr_plateau_factor']:.3f}, "
            f"patience={cfg['lr_plateau_patience']}"
        )

    model.train()

    def _prepare_train_batch(batch):
        return prepare_video_batch(batch, cfg, transform, zero_frame, cache_state, split_name="train")

    def _prepare_val_batch(batch):
        return prepare_video_batch(batch, cfg, transform, zero_frame, cache_state, split_name="val")

    def _prepare_clip(clip_tensor, video_data):
        if video_data.get("source") == "feature_cache":
            return clip_tensor.to(torch.float32).contiguous()
        clip_tensor = decode_clip_for_model(clip_tensor, video_data, cfg)
        if video_data.get("source") != "cache":
            clip_tensor = _apply_imagenet_normalization(clip_tensor)
        return clip_tensor

    print()
    total_epochs = _to_int(cfg.get("epochs", 15), 15)
    for epoch in range(start_epoch, total_epochs):
        prioritize_first_batch = (
            epoch == start_epoch and first_epoch_first_video_idx is not None
        )
        dataloader_train_epoch = _build_train_dataloader(
            prioritize_first_batch=prioritize_first_batch,
            epoch_seed=int(time.time() * 1000) + epoch,
        )
        if cfg.get("lr_warmup_enabled", False) and epoch < cfg.get("lr_warmup_epochs", 0):
            warmup_lr = _compute_warmup_lr(
                base_lr,
                cfg["lr_warmup_start_factor"],
                cfg["lr_warmup_epochs"],
                epoch,
            )
            _set_optimizer_lr(optimizer, warmup_lr)

        total_loss = 0.0
        valid_train_batches = 0
        skipped_non_finite_batches = 0
        eval_conf_thr = float(cfg.get("eval_conf_threshold", 0.0))
        #             "Precision05" "Recall05" "mAP03" "mAP04" "mAP05" "mAP06" "mAP07" "mAP09" "Avg. mAP"
        train_metrics = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        train_video_records = []
        current_lr = _get_optimizer_lr(optimizer)
        print(f">>> Epoch {epoch + 1}/{total_epochs} | LR: {current_lr:.8f}")
        if prioritize_first_batch:
            print(
                f">>> Forcing video {cfg['first_epoch_first_video_id']} into the first batch of this epoch"
            )
        pbar = tqdm(total=train_video_num, desc=f"Batch Progress: ")
        # secondary bar to display timing information on its own line
        timing_bar = tqdm(total=0, bar_format="{postfix}", position=1, leave=False)
        batch_iter = iter(dataloader_train_epoch)

        if cfg["async_prefetch_next_video"]:
            with ThreadPoolExecutor(max_workers=cfg["prefetch_workers"]) as executor:
                # ensure timing_bar exists inside executor scope
                prefetch_queue_size = cfg["prefetch_videos_ahead"] + 1
                future_queue = deque()

                def submit_next_video():
                    try:
                        next_batch = next(batch_iter)
                    except StopIteration:
                        return False

                    future_queue.append(
                        executor.submit(
                            _prepare_train_batch,
                            next_batch,
                        )
                    )
                    return True

                for _ in range(prefetch_queue_size):
                    if not submit_next_video():
                        break

                batch_idx = 0
                while future_queue:
                    # data preparation timing (async)
                    t0 = time.time()
                    current_video_data = future_queue.popleft().result()
                    t_data = time.time() - t0

                    # training timing
                    t1 = time.time()
                    loss_val, rec = train_one_video(model, optimizer, criterion, current_video_data, cfg, scaler=scaler)
                    if rec.get("skipped_non_finite", False):
                        skipped_non_finite_batches += 1
                        pbar.update(1)
                        while len(future_queue) < prefetch_queue_size:
                            if not submit_next_video():
                                break
                        continue

                    valid_train_batches += 1
                    total_loss += float(loss_val)
                    t_train = time.time() - t1

                    read = current_video_data.get("read_time", t_data)
                    prep = current_video_data.get("proc_time", 0.0)

                    # compute per-video metrics
                    # reuse validator helper to generate numbers
                    _, _, m03 = _compute_metrics_at_iou([rec], cfg["class_num"], 0.3, conf_threshold=eval_conf_thr)
                    _, _, m04 = _compute_metrics_at_iou([rec], cfg["class_num"], 0.4, conf_threshold=eval_conf_thr)
                    p, r, m05 = _compute_metrics_at_iou([rec], cfg["class_num"], 0.5, conf_threshold=eval_conf_thr)
                    _, _, m06 = _compute_metrics_at_iou([rec], cfg["class_num"], 0.6, conf_threshold=eval_conf_thr)
                    _, _, m07 = _compute_metrics_at_iou([rec], cfg["class_num"], 0.7, conf_threshold=eval_conf_thr)
                    _, _, m09 = _compute_metrics_at_iou([rec], cfg["class_num"], 0.9, conf_threshold=eval_conf_thr)
                    avgm = (m03 + m04 + m05 + m06 + m07 + m09) / 6.0
                    train_video_records.append(rec)


                    # disable automatic sorting so keys appear in the order we specify
                    timing_bar.set_postfix_str(
                        f"[ read:{read:.1f}s  pre:{prep:.1f}s  train:{t_train:.1f}s  loss:{loss_val:.4f}  Prec:{p:.3f}  Rec:{r:.3f}  m30:{m03:.3f}  m40:{m04:.3f}  m50:{m05:.3f}  m60:{m06:.3f}  m70:{m07:.3f}  m90:{m09:.3f}  Avg_m:{avgm:.3f} ]"
                    )
                    batch_idx += 1
                    pbar.update(1)

                    while len(future_queue) < prefetch_queue_size:
                        if not submit_next_video():
                            break
        else:
            batch_idx = 0
            for batch_train in dataloader_train_epoch:
                t0 = time.time()
                video_data = _prepare_train_batch(batch_train)
                t_data = time.time() - t0

                t1 = time.time()
                loss_val, rec = train_one_video(model, optimizer, criterion, video_data, cfg, scaler=scaler)
                if rec.get("skipped_non_finite", False):
                    skipped_non_finite_batches += 1
                    batch_idx += 1
                    pbar.update(1)
                    continue

                valid_train_batches += 1
                total_loss += float(loss_val)
                t_train = time.time() - t1

                read = video_data.get("read_time", t_data)
                prep = video_data.get("proc_time", 0.0)

                _, _, m03 = _compute_metrics_at_iou([rec], cfg["class_num"], 0.3, conf_threshold=eval_conf_thr)
                _, _, m04 = _compute_metrics_at_iou([rec], cfg["class_num"], 0.4, conf_threshold=eval_conf_thr)
                _, _, m05 = _compute_metrics_at_iou([rec], cfg["class_num"], 0.5, conf_threshold=eval_conf_thr)
                p, r, m06 = _compute_metrics_at_iou([rec], cfg["class_num"], 0.6, conf_threshold=eval_conf_thr)
                _, _, m07 = _compute_metrics_at_iou([rec], cfg["class_num"], 0.7, conf_threshold=eval_conf_thr)
                _, _, m09 = _compute_metrics_at_iou([rec], cfg["class_num"], 0.9, conf_threshold=eval_conf_thr)
                avgm = (m03 + m04 + m05 + m06 + m07 + m09) / 6.0
                train_video_records.append(rec)

                # disable automatic sorting so keys appear in the order we specify
                timing_bar.set_postfix(
                    read=f"{read:.3f}s", prep=f"{prep:.3f}s", backward=f"{t_train:.3f}s",
                    train_loss=f"{loss_val:.4f}", precision=f"{p:.3f}", recall=f"{r:.3f}",
                    m03=f"{m03:.3f}", m04=f"{m04:.3f}", m05=f"{m05:.3f}", m06=f"{m06:.3f}", m07=f"{m07:.3f}", m09=f"{m09:.3f}", avgm=f"{avgm:.3f}"
                )
                batch_idx += 1
                pbar.update(1)

        pbar.close()
        avg_loss = total_loss / max(valid_train_batches, 1)
        timing_bar.close()
        val_metrics = validate_one_epoch(
            model=model,
            dataloader_val=dataloader_val,
            criterion=criterion,
            cfg=cfg,
            prepare_video_batch_fn=_prepare_val_batch,
            prepare_clip_fn=_prepare_clip,
            csv_path=Path(cfg["output_dir"]) / f"val_metrics.csv",
            epoch=epoch + 1
        )
        # compute average mAP across the four thresholds for summary
        avg_map = (
            val_metrics.get('map03', 0.0)
            + val_metrics.get('map04', 0.0)
            + val_metrics.get('map05', 0.0)
            + val_metrics.get('map06', 0.0)
            + val_metrics.get('map07', 0.0)
            + val_metrics.get('map09', 0.0)
        ) / 6.0

        if train_video_records:
            p05_list = []
            r05_list = []
            m03_list = []
            m04_list = []
            m05_list = []
            m06_list = []
            m07_list = []
            m09_list = []

            for rec in train_video_records:
                p05, r05, m05 = _compute_metrics_at_iou(
                    [rec], cfg["class_num"], 0.5, conf_threshold=eval_conf_thr
                )
                _, _, m03 = _compute_metrics_at_iou(
                    [rec], cfg["class_num"], 0.3, conf_threshold=eval_conf_thr
                )
                _, _, m04 = _compute_metrics_at_iou(
                    [rec], cfg["class_num"], 0.4, conf_threshold=eval_conf_thr
                )
                _, _, m06 = _compute_metrics_at_iou(
                    [rec], cfg["class_num"], 0.6, conf_threshold=eval_conf_thr
                )
                _, _, m07 = _compute_metrics_at_iou(
                    [rec], cfg["class_num"], 0.7, conf_threshold=eval_conf_thr
                )
                _, _, m09 = _compute_metrics_at_iou(
                    [rec], cfg["class_num"], 0.9, conf_threshold=eval_conf_thr
                )

                p05_list.append(float(p05))
                r05_list.append(float(r05))
                m03_list.append(float(m03))
                m04_list.append(float(m04))
                m05_list.append(float(m05))
                m06_list.append(float(m06))
                m07_list.append(float(m07))
                m09_list.append(float(m09))

            p05 = float(sum(p05_list) / len(p05_list))
            r05 = float(sum(r05_list) / len(r05_list))
            m03 = float(sum(m03_list) / len(m03_list))
            m04 = float(sum(m04_list) / len(m04_list))
            m05 = float(sum(m05_list) / len(m05_list))
            m06 = float(sum(m06_list) / len(m06_list))
            m07 = float(sum(m07_list) / len(m07_list))
            m09 = float(sum(m09_list) / len(m09_list))
            avgm = (m03 + m04 + m05 + m06 + m07 + m09) / 6.0
            train_metrics = [p05, r05, m03, m04, m05, m06, m07, m09, avgm]
        else:
            train_metrics = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        if skipped_non_finite_batches > 0:
            print(
                f"Warning: skipped {skipped_non_finite_batches} training videos due to non-finite output/loss in this epoch."
            )

        plateau_metric_map = {
            "loss": float(val_metrics["loss"]),
            "map03": float(val_metrics.get("map03", 0.0)),
            "map04": float(val_metrics.get("map04", 0.0)),
            "map05": float(val_metrics.get("map05", 0.0)),
            "map06": float(val_metrics.get("map06", 0.0)),
            "map07": float(val_metrics.get("map07", 0.0)),
            "map09": float(val_metrics.get("map09", 0.0)),
            "avg_map": float(avg_map),
        }
        plateau_metric_value = plateau_metric_map[cfg.get("lr_plateau_monitor", "map05")]
        if (
            plateau_scheduler is not None
            and cfg.get("lr_plateau_enabled", False)
            and epoch + 1 > cfg.get("lr_warmup_epochs", 0)
        ):
            plateau_scheduler.step(plateau_metric_value)
        current_lr = _get_optimizer_lr(optimizer)

        print(
            f" => Metrics:\n| "
            f"TrainLoss: {avg_loss:.5f} | "
            f"Precision: {train_metrics[0]:.4f} | "
            f"Recall: {train_metrics[1]:.4f} | "
            f"mAP0.3: {train_metrics[2]:.4f} | "
            f"mAP0.4: {train_metrics[3]:.4f} | "
            f"mAP0.5: {train_metrics[4]:.4f} | "
            f"mAP0.6: {train_metrics[5]:.4f} | "
            f"mAP0.7: {train_metrics[6]:.4f} | "
            f"mAP0.9: {train_metrics[7]:.4f} | "
            f"Avg mAP: {train_metrics[8]:.4f} |\n|  "
            f"ValLoss : {val_metrics['loss']:.5f} | "
            f"Precision: {val_metrics['precision']:.4f} | "
            f"Recall: {val_metrics['recall']:.4f} | "
            f"mAP0.3: {val_metrics['map03']:.4f} | "
            f"mAP0.4: {val_metrics['map04']:.4f} | "
            f"mAP0.5: {val_metrics['map05']:.4f} | "
            f"mAP0.6: {val_metrics['map06']:.4f} | "
            f"mAP0.7: {val_metrics['map07']:.4f} | "
            f"mAP0.9: {val_metrics['map09']:.4f} | "
            f"Avg mAP: {avg_map:.4f} | "
            f"LR: {current_lr:.8f} |"
        )
        
        record_metrics(Path(cfg["output_dir"]) / f"train_metrics.csv", 
                       [epoch + 1, avg_loss] + train_metrics + [val_metrics['loss'], val_metrics['precision'], val_metrics['recall'], 
                        val_metrics['map03'], val_metrics['map04'], val_metrics['map05'], val_metrics['map06'], val_metrics['map07'], val_metrics['map09'], avg_map])
        # save checkpoints
        current_avg_map = float(avg_map)
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "plateau_scheduler": plateau_scheduler.state_dict() if plateau_scheduler is not None else None,
            "best_avg_map": best_avg_map,
        }
        last_path = Path(cfg["output_dir"]) / "last_checkpoint.pt"
        torch.save(checkpoint, last_path)
        if current_avg_map > best_avg_map:
            best_avg_map = current_avg_map
            checkpoint["best_avg_map"] = best_avg_map
            best_path = Path(cfg["output_dir"]) / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)
            print(f"New best model (avg_map={best_avg_map:.6f}) saved to {best_path}")
        print()

    return True


