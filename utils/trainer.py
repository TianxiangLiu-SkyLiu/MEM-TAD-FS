import os
import io
import yaml
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import deque, OrderedDict
import math
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
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
from utils.config import load_config
from utils.validator import (
    validate_one_epoch,
    _extract_pred_and_gt,
    _compute_metrics_at_iou,
    _nms_kwargs_from_cfg,
    _select_state,
    _scatter_state,
)

# lz4 is optional for cache decompression; if the package isn't available
# the loader will still function for plain .pt caches.
try:
    import lz4.frame
except ImportError:
    lz4 = None


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
COLOR_GREEN = "\033[92m"
COLOR_RESET = "\033[0m"


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


def _set_reproducibility(seed, deterministic=False):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = not bool(deterministic)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)


def _set_optimizer_lr(optimizer, lr_value):
    for param_group in optimizer.param_groups:
        lr_scale = float(param_group.get("lr_scale", 1.0))
        param_group["lr"] = float(lr_value) * lr_scale


def _get_optimizer_lr(optimizer):
    if not optimizer.param_groups:
        return 0.0
    param_group = optimizer.param_groups[0]
    lr_scale = max(float(param_group.get("lr_scale", 1.0)), 1e-12)
    return float(param_group.get("lr", 0.0)) / lr_scale


def _optimizer_min_lrs(optimizer, base_min_lr):
    return [
        float(base_min_lr) * float(group.get("lr_scale", 1.0))
        for group in optimizer.param_groups
    ]


def _clone_state_dict(state_dict, device=None):
    cloned = OrderedDict()
    for name, value in state_dict.items():
        if torch.is_tensor(value):
            tensor = value.detach().clone()
            if device is not None:
                tensor = tensor.to(device=device)
            cloned[name] = tensor
        else:
            cloned[name] = value
    return cloned


def _init_ema_state(model):
    return _clone_state_dict(model.state_dict())


@torch.no_grad()
def _update_ema_state(model, ema_state, decay):
    if ema_state is None:
        return
    model_state = model.state_dict()
    decay = float(decay)
    for name, value in model_state.items():
        if name not in ema_state:
            ema_state[name] = value.detach().clone()
            continue
        if torch.is_tensor(value):
            ema_value = ema_state[name]
            if ema_value.device != value.device:
                ema_value = ema_value.to(device=value.device)
                ema_state[name] = ema_value
            if torch.is_floating_point(value):
                ema_value.mul_(decay).add_(value.detach(), alpha=1.0 - decay)
            else:
                ema_value.copy_(value.detach())
        else:
            ema_state[name] = value


def _load_ema_state_from_checkpoint(ckpt, device):
    ema_state = ckpt.get("ema_model", None)
    if ema_state is None:
        return None
    return _clone_state_dict(ema_state, device=device)


def _swap_to_model_state(model, state_dict):
    backup = _clone_state_dict(model.state_dict(), device="cpu")
    model.load_state_dict(state_dict, strict=True)
    return backup


def _restore_model_state(model, backup):
    model.load_state_dict(backup, strict=True)


def _build_optimizer_param_groups(model, cfg):
    base_lr = float(cfg["lr"])
    base_weight_decay = float(cfg.get("weight_decay", 0.0))
    joint_lr_scale = float(cfg.get("joint_memory_lr_scale", 1.0))
    joint_weight_decay_scale = float(
        cfg.get("joint_memory_weight_decay_scale", 1.0)
    )
    transformer_lr_scale = float(cfg.get("memory_transformer_lr_scale", 1.0))
    transformer_weight_decay_scale = float(
        cfg.get("memory_transformer_weight_decay_scale", 1.0)
    )

    base_params = []
    joint_memory_params = []
    memory_transformer_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("memory_transformer_"):
            memory_transformer_params.append(parameter)
        elif name.startswith("joint_memory_"):
            joint_memory_params.append(parameter)
        else:
            base_params.append(parameter)

    groups = [{
        "params": base_params,
        "lr": base_lr,
        "lr_scale": 1.0,
        "weight_decay": base_weight_decay,
        "group_name": "base",
    }]
    if joint_memory_params:
        groups.append({
            "params": joint_memory_params,
            "lr": base_lr * joint_lr_scale,
            "lr_scale": joint_lr_scale,
            "weight_decay": base_weight_decay * joint_weight_decay_scale,
            "group_name": "joint_memory",
        })
    if memory_transformer_params:
        groups.append({
            "params": memory_transformer_params,
            "lr": base_lr * transformer_lr_scale,
            "lr_scale": transformer_lr_scale,
            "weight_decay": base_weight_decay * transformer_weight_decay_scale,
            "group_name": "memory_transformer",
        })
    return groups


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


class _PriorityIndicesSampler(Sampler):
    def __init__(self, dataset_size, first_indices, seed=None):
        self.dataset_size = int(dataset_size)
        seen = set()
        self.first_indices = []
        for idx in first_indices:
            idx = int(idx)
            if 0 <= idx < self.dataset_size and idx not in seen:
                self.first_indices.append(idx)
                seen.add(idx)

        generator = torch.Generator()
        if seed is None:
            seed = torch.seed()
        generator.manual_seed(int(seed))

        rest = torch.randperm(self.dataset_size, generator=generator).tolist()
        rest = [idx for idx in rest if idx not in seen]
        self.indices = self.first_indices + rest

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def _video_collate(batch):
    if not batch:
        return {}
    return {key: [sample[key] for sample in batch] for key in batch[0].keys()}


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


def _find_largest_video_indices(dataset, count):
    video_list = getattr(dataset, "video_list", None)
    if not video_list:
        return []

    def _size_key(video_info):
        feature_path = video_info.get("feature_path", None)
        if feature_path:
            try:
                path = Path(str(feature_path))
                if path.exists():
                    return int(path.stat().st_size)
            except OSError:
                pass
        return int(video_info.get("total_frames", 0) or 0)

    ranked = sorted(
        enumerate(video_list),
        key=lambda item: _size_key(item[1]),
        reverse=True,
    )
    return [idx for idx, _ in ranked[:max(int(count), 0)]]


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


def record_memory_diagnostics(csv_path, epoch, train_diag, val_diag):
    import csv

    metric_names = [
        "shallow_gate_mean",
        "shallow_gate_std",
        "shallow_state_change_ratio",
        "shallow_state_retention",
        "deep_gate_mean",
        "deep_gate_std",
        "deep_state_change_ratio",
        "deep_state_retention",
        "shallow_deep_cosine",
        "joint_shallow_gate_mean",
        "joint_deep_gate_mean",
        "joint_shallow_gate_std",
        "joint_deep_gate_std",
        "joint_loc_residual_scale",
        "joint_cls_residual_scale",
    ]
    fieldnames = ["Epoch"]
    fieldnames.extend(f"Train_{name}" for name in metric_names)
    fieldnames.extend(f"Val_{name}" for name in metric_names)
    row = {"Epoch": int(epoch)}
    row.update({f"Train_{name}": float(train_diag.get(name, 0.0)) for name in metric_names})
    row.update({f"Val_{name}": float(val_diag.get(name, 0.0)) for name in metric_names})

    csv_path = Path(csv_path)
    write_header = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
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
        cfg["seed"] = max(_to_int(cfg.get("seed", 3407), 3407), 0)
        cfg["deterministic_training"] = _to_bool(
            cfg.get("deterministic_training", False)
        )
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
        cfg["map_eval_scope"] = str(cfg.get("map_eval_scope", "global") or "global").strip().lower()
        if cfg["map_eval_scope"] not in {"global", "video"}:
            cfg["map_eval_scope"] = "global"
        cfg["detection_score_mode"] = str(
            cfg.get("detection_score_mode", "sigmoid_class") or "sigmoid_class"
        ).strip().lower()
        if cfg["detection_score_mode"] not in {
            "sigmoid_class",
            "objectness",
            "objectness_x_class",
            "objectness*class",
            "fused",
            "quality_x_class",
            "quality*class",
            "iou_x_class",
        }:
            cfg["detection_score_mode"] = "sigmoid_class"

        # model head regularization
        cfg["head_channels"] = max(_to_int(cfg.get("head_channels", 0), cfg.get("max_detection_num", 128)), 32)
        # allow shrinking head hidden dim for lower VRAM; keep a safe minimum
        cfg["det_head_hidden_dim"] = max(_to_int(cfg.get("det_head_hidden_dim", 256), 64), 64)
        cfg["head_dropout"] = max(min(_to_float(cfg.get("head_dropout", 0.2), 0.2), 0.8), 0.0)
        cfg["head_token_dropout"] = max(
            min(_to_float(cfg.get("head_token_dropout", cfg["head_dropout"]), cfg["head_dropout"]), 0.8),
            0.0,
        )
        cfg["memory_transformer_head_enabled"] = _to_bool(
            cfg.get("memory_transformer_head_enabled", False)
        )
        cfg["memory_transformer_aux_loss_enabled"] = _to_bool(
            cfg.get("memory_transformer_aux_loss_enabled", False)
        )
        cfg["memory_transformer_aux_loss_weight"] = max(
            _to_float(cfg.get("memory_transformer_aux_loss_weight", 0.4), 0.4),
            0.0,
        )
        cfg["memory_transformer_encoder_proposal_enabled"] = _to_bool(
            cfg.get("memory_transformer_encoder_proposal_enabled", False)
        )
        cfg["memory_transformer_encoder_proposal_loss_enabled"] = _to_bool(
            cfg.get(
                "memory_transformer_encoder_proposal_loss_enabled",
                cfg["memory_transformer_encoder_proposal_enabled"],
            )
        )
        cfg["memory_transformer_encoder_proposal_loss_weight"] = max(
            _to_float(
                cfg.get("memory_transformer_encoder_proposal_loss_weight", 0.3),
                0.3,
            ),
            0.0,
        )
        cfg["memory_transformer_deep_prior_enabled"] = _to_bool(
            cfg.get("memory_transformer_deep_prior_enabled", False)
        )
        cfg["memory_transformer_deep_prior_loss_enabled"] = _to_bool(
            cfg.get(
                "memory_transformer_deep_prior_loss_enabled",
                cfg["memory_transformer_deep_prior_enabled"],
            )
        )
        cfg["memory_transformer_deep_prior_loss_weight"] = max(
            _to_float(cfg.get("memory_transformer_deep_prior_loss_weight", 0.2), 0.2),
            0.0,
        )
        cfg["memory_transformer_deep_prior_context_scale"] = max(
            _to_float(
                cfg.get("memory_transformer_deep_prior_context_scale", 3.0),
                3.0,
            ),
            0.1,
        )
        cfg["memory_transformer_deep_prior_points"] = max(
            _to_int(cfg.get("memory_transformer_deep_prior_points", 7), 7),
            2,
        )
        cfg["memory_transformer_deep_prior_delta_scale"] = max(
            _to_float(cfg.get("memory_transformer_deep_prior_delta_scale", 0.75), 0.75),
            0.0,
        )
        cfg["memory_transformer_deep_prior_query_gate_bias"] = _to_float(
            cfg.get("memory_transformer_deep_prior_query_gate_bias", -2.0),
            -2.0,
        )
        cfg["memory_transformer_query_mode"] = str(
            cfg.get(
                "memory_transformer_query_mode",
                "proposal" if cfg["memory_transformer_encoder_proposal_enabled"] else "fixed",
            )
            or "fixed"
        ).strip().lower()
        if cfg["memory_transformer_query_mode"] not in {"fixed", "proposal", "hybrid"}:
            cfg["memory_transformer_query_mode"] = "fixed"
        if cfg["memory_transformer_query_mode"] in {"proposal", "hybrid"}:
            cfg["memory_transformer_encoder_proposal_enabled"] = True
        cfg["memory_transformer_hybrid_fixed_queries"] = max(
            min(
                _to_int(cfg.get("memory_transformer_hybrid_fixed_queries", 60), 60),
                cfg["max_detection_num"],
            ),
            0,
        )
        default_hybrid_proposal_queries = (
            cfg["max_detection_num"] - cfg["memory_transformer_hybrid_fixed_queries"]
        )
        cfg["memory_transformer_hybrid_proposal_queries"] = max(
            min(
                _to_int(
                    cfg.get(
                        "memory_transformer_hybrid_proposal_queries",
                        default_hybrid_proposal_queries,
                    ),
                    default_hybrid_proposal_queries,
                ),
                cfg["max_detection_num"] - cfg["memory_transformer_hybrid_fixed_queries"],
            ),
            0,
        )
        cfg["memory_transformer_dim"] = max(
            _to_int(cfg.get("memory_transformer_dim", 256), 256),
            32,
        )
        cfg["memory_transformer_heads"] = max(
            _to_int(cfg.get("memory_transformer_heads", 8), 8),
            1,
        )
        while (
            cfg["memory_transformer_dim"] % cfg["memory_transformer_heads"] != 0
            and cfg["memory_transformer_heads"] > 1
        ):
            cfg["memory_transformer_heads"] -= 1
        cfg["memory_transformer_encoder_layers"] = max(
            _to_int(cfg.get("memory_transformer_encoder_layers", 2), 2),
            0,
        )
        cfg["memory_transformer_decoder_layers"] = max(
            _to_int(cfg.get("memory_transformer_decoder_layers", 3), 3),
            1,
        )
        cfg["memory_transformer_ff_dim"] = max(
            _to_int(
                cfg.get(
                    "memory_transformer_ff_dim",
                    cfg["memory_transformer_dim"] * 4,
                ),
                cfg["memory_transformer_dim"] * 4,
            ),
            cfg["memory_transformer_dim"],
        )
        cfg["memory_transformer_dropout"] = max(
            min(
                _to_float(
                    cfg.get("memory_transformer_dropout", cfg["head_dropout"]),
                    cfg["head_dropout"],
                ),
                0.8,
            ),
            0.0,
        )
        cfg["memory_transformer_lr_scale"] = max(
            _to_float(cfg.get("memory_transformer_lr_scale", 1.0), 1.0),
            0.0,
        )
        cfg["memory_transformer_weight_decay_scale"] = max(
            _to_float(
                cfg.get("memory_transformer_weight_decay_scale", 1.0),
                1.0,
            ),
            0.0,
        )
        cfg["memory_transformer_iterative_refine_enabled"] = _to_bool(
            cfg.get("memory_transformer_iterative_refine_enabled", False)
        )
        cfg["memory_transformer_iterative_refine_detach"] = _to_bool(
            cfg.get("memory_transformer_iterative_refine_detach", True)
        )
        cfg["denoising_enabled"] = _to_bool(cfg.get("denoising_enabled", False))
        cfg["denoising_groups"] = max(
            _to_int(cfg.get("denoising_groups", 5), 5),
            1,
        )
        cfg["denoising_label_noise_ratio"] = max(
            min(
                _to_float(
                    cfg.get("denoising_label_noise_ratio", 0.1),
                    0.1,
                ),
                1.0,
            ),
            0.0,
        )
        cfg["denoising_box_noise_scale"] = max(
            _to_float(cfg.get("denoising_box_noise_scale", 0.4), 0.4),
            0.0,
        )
        cfg["denoising_loss_weight"] = max(
            _to_float(cfg.get("denoising_loss_weight", 1.0), 1.0),
            0.0,
        )
        cfg["denoising_aux_loss_weight"] = max(
            _to_float(
                cfg.get(
                    "denoising_aux_loss_weight",
                    cfg.get("memory_transformer_aux_loss_weight", 0.4),
                ),
                cfg.get("memory_transformer_aux_loss_weight", 0.4),
            ),
            0.0,
        )
        cfg["denoising_max_queries"] = max(
            _to_int(cfg.get("denoising_max_queries", 100), 100),
            0,
        )
        cfg["ema_enabled"] = _to_bool(cfg.get("ema_enabled", False))
        cfg["ema_decay"] = max(
            min(_to_float(cfg.get("ema_decay", 0.999), 0.999), 0.99999),
            0.0,
        )
        cfg["ema_eval_enabled"] = _to_bool(
            cfg.get("ema_eval_enabled", cfg["ema_enabled"])
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
        cfg["use_temporal_references"] = _to_bool(cfg.get("use_temporal_references", True))
        raw_ref_widths = cfg.get("temporal_ref_widths")
        if raw_ref_widths is None:
            cfg["temporal_ref_width"] = max(
                min(_to_float(cfg.get("temporal_ref_width", 0.03), 0.03), 1.0 - 1e-4),
                1e-4,
            )
            cfg["temporal_ref_widths"] = [cfg["temporal_ref_width"]]
        else:
            if not isinstance(raw_ref_widths, (list, tuple)) or not raw_ref_widths:
                raise ValueError("temporal_ref_widths must be a non-empty list")
            cfg["temporal_ref_widths"] = [
                max(min(_to_float(width, 0.03), 1.0 - 1e-4), 1e-4)
                for width in raw_ref_widths
            ]

        raw_query_counts = cfg.get("temporal_ref_query_counts")
        if raw_query_counts is None:
            scale_num = len(cfg["temporal_ref_widths"])
            base_count, remainder = divmod(cfg["max_detection_num"], scale_num)
            cfg["temporal_ref_query_counts"] = [
                base_count + (scale_idx < remainder) for scale_idx in range(scale_num)
            ]
        else:
            if not isinstance(raw_query_counts, (list, tuple)):
                raise ValueError("temporal_ref_query_counts must be a list")
            cfg["temporal_ref_query_counts"] = [
                _to_int(count, 0) for count in raw_query_counts
            ]
        if len(cfg["temporal_ref_query_counts"]) != len(cfg["temporal_ref_widths"]):
            raise ValueError(
                "temporal_ref_query_counts and temporal_ref_widths must have the same length"
            )
        if any(count <= 0 for count in cfg["temporal_ref_query_counts"]):
            raise ValueError("every temporal reference scale must have at least one query")
        if sum(cfg["temporal_ref_query_counts"]) != cfg["max_detection_num"]:
            raise ValueError(
                "sum(temporal_ref_query_counts) must equal max_detection_num"
            )
        cfg["reference_attention_enabled"] = _to_bool(
            cfg.get("reference_attention_enabled", False)
        )
        cfg["reference_attention_layers"] = max(
            _to_int(cfg.get("reference_attention_layers", 3), 3),
            1,
        )
        cfg["reference_attention_points"] = max(
            _to_int(cfg.get("reference_attention_points", 7), 7),
            2,
        )
        cfg["reference_attention_context_scale"] = max(
            _to_float(cfg.get("reference_attention_context_scale", 1.5), 1.5),
            0.1,
        )
        cfg["reference_attention_context_dim"] = max(
            _to_int(cfg.get("reference_attention_context_dim", 128), 128),
            16,
        )
        cfg["reference_attention_dropout"] = max(
            min(
                _to_float(
                    cfg.get("reference_attention_dropout", cfg["head_dropout"]),
                    cfg["head_dropout"],
                ),
                0.8,
            ),
            0.0,
        )
        cfg["joint_memory_detection_enabled"] = _to_bool(
            cfg.get("joint_memory_detection_enabled", False)
        )
        cfg["joint_memory_dim"] = max(
            _to_int(cfg.get("joint_memory_dim", 64), 64),
            16,
        )
        cfg["joint_memory_dropout"] = max(
            min(
                _to_float(
                    cfg.get("joint_memory_dropout", cfg["head_dropout"]),
                    cfg["head_dropout"],
                ),
                0.8,
            ),
            0.0,
        )
        cfg["joint_memory_lr_scale"] = max(
            _to_float(cfg.get("joint_memory_lr_scale", 1.0), 1.0),
            0.0,
        )
        cfg["joint_memory_weight_decay_scale"] = max(
            _to_float(
                cfg.get("joint_memory_weight_decay_scale", 1.0),
                1.0,
            ),
            0.0,
        )
        cfg["joint_memory_points"] = max(
            _to_int(cfg.get("joint_memory_points", 7), 7),
            2,
        )
        cfg["joint_memory_shallow_context_scale"] = max(
            _to_float(
                cfg.get("joint_memory_shallow_context_scale", 1.0),
                1.0,
            ),
            0.1,
        )
        cfg["joint_memory_deep_context_scale"] = max(
            _to_float(cfg.get("joint_memory_deep_context_scale", 3.0), 3.0),
            cfg["joint_memory_shallow_context_scale"],
        )
        cfg["joint_memory_max_residual_scale"] = max(
            _to_float(cfg.get("joint_memory_max_residual_scale", 0.1), 0.1),
            0.0,
        )
        cfg["joint_memory_residual_gate_bias"] = _to_float(
            cfg.get("joint_memory_residual_gate_bias", -1.0),
            -1.0,
        )
        cfg["memory_proposal_refine_enabled"] = _to_bool(
            cfg.get("memory_proposal_refine_enabled", False)
        )
        cfg["memory_proposal_refine_dim"] = max(
            _to_int(cfg.get("memory_proposal_refine_dim", 64), 64),
            16,
        )
        cfg["memory_proposal_refine_points"] = max(
            _to_int(cfg.get("memory_proposal_refine_points", 7), 7),
            2,
        )
        cfg["memory_proposal_refine_dropout"] = max(
            min(
                _to_float(
                    cfg.get("memory_proposal_refine_dropout", cfg["head_dropout"]),
                    cfg["head_dropout"],
                ),
                0.8,
            ),
            0.0,
        )
        cfg["memory_proposal_refine_shallow_context_scale"] = max(
            _to_float(
                cfg.get("memory_proposal_refine_shallow_context_scale", 1.0),
                1.0,
            ),
            0.1,
        )
        cfg["memory_proposal_refine_deep_context_scale"] = max(
            _to_float(
                cfg.get("memory_proposal_refine_deep_context_scale", 3.0),
                3.0,
            ),
            cfg["memory_proposal_refine_shallow_context_scale"],
        )
        cfg["memory_proposal_refine_max_boundary_shift"] = max(
            _to_float(
                cfg.get("memory_proposal_refine_max_boundary_shift", 0.25),
                0.25,
            ),
            0.0,
        )
        cfg["memory_proposal_refine_max_center_shift"] = max(
            _to_float(
                cfg.get("memory_proposal_refine_max_center_shift", 0.25),
                0.25,
            ),
            0.0,
        )
        cfg["memory_proposal_refine_max_log_width_delta"] = max(
            _to_float(
                cfg.get("memory_proposal_refine_max_log_width_delta", 0.35),
                0.35,
            ),
            0.0,
        )
        cfg["memory_proposal_refine_gate_bias"] = _to_float(
            cfg.get("memory_proposal_refine_gate_bias", -6.0),
            -6.0,
        )
        cfg["memory_auxiliary_enabled"] = _to_bool(
            cfg.get("memory_auxiliary_enabled", False)
        )
        cfg["shallow_boundary_aux_weight"] = max(
            _to_float(cfg.get("shallow_boundary_aux_weight", 0.1), 0.1),
            0.0,
        )
        cfg["shallow_boundary_aux_sigma"] = max(
            _to_float(cfg.get("shallow_boundary_aux_sigma", 0.02), 0.02),
            1e-4,
        )
        cfg["shallow_boundary_aux_pos_weight"] = max(
            _to_float(cfg.get("shallow_boundary_aux_pos_weight", 4.0), 4.0),
            0.0,
        )
        cfg["deep_class_aux_weight"] = max(
            _to_float(cfg.get("deep_class_aux_weight", 0.1), 0.1),
            0.0,
        )
        cfg["memory_auxiliary_dropout"] = max(
            min(
                _to_float(cfg.get("memory_auxiliary_dropout", 0.1), 0.1),
                0.8,
            ),
            0.0,
        )
        cfg["memory_dropout"] = max(min(_to_float(cfg.get("memory_dropout", 0.1), 0.1), 0.8), 0.0)
        cfg["memory_proposal_dropout"] = max(
            min(
                _to_float(
                    cfg.get("memory_proposal_dropout", cfg["memory_dropout"]),
                    cfg["memory_dropout"],
                ),
                0.8,
            ),
            0.0,
        )
        cfg["shallow_gate_init_bias"] = _to_float(
            cfg.get("shallow_gate_init_bias", 0.0),
            0.0,
        )
        cfg["deep_gate_init_bias"] = _to_float(
            cfg.get("deep_gate_init_bias", -2.2),
            -2.2,
        )
        cfg["memory_diagnostics_enabled"] = _to_bool(
            cfg.get("memory_diagnostics_enabled", False)
        )
        cfg["memory_diagnostics_stride"] = max(
            _to_int(cfg.get("memory_diagnostics_stride", 4), 4),
            1,
        )

        # loss algorithm options
        cfg["loss_focal_gamma"] = max(_to_float(cfg.get("loss_focal_gamma", 2.0), 2.0), 0.0)
        cfg["loss_focal_alpha"] = max(min(_to_float(cfg.get("loss_focal_alpha", 0.25), 0.25), 0.95), 0.01)
        cfg["loss_neg_pos_ratio"] = max(_to_float(cfg.get("loss_neg_pos_ratio", 3.0), 3.0), 1.0)
        cfg["loss_iou_weight"] = max(_to_float(cfg.get("loss_iou_weight", 0.5), 0.5), 0.0)
        cfg["loss_label_smoothing"] = max(min(_to_float(cfg.get("loss_label_smoothing", 0.05), 0.05), 0.2), 0.0)
        cfg["loss_force_gt_match"] = _to_bool(cfg.get("loss_force_gt_match", True))
        cfg["loss_missed_gt_weight"] = max(_to_float(cfg.get("loss_missed_gt_weight", 0.5), 0.5), 0.0)
        cfg["loss_matcher"] = str(cfg.get("loss_matcher", "hungarian") or "hungarian").strip().lower()
        if cfg["loss_matcher"] not in {"hungarian", "greedy"}:
            cfg["loss_matcher"] = "hungarian"
        cfg["loss_match_cost_class"] = max(_to_float(cfg.get("loss_match_cost_class", 1.0), 1.0), 0.0)
        cfg["loss_match_class_warmup_epochs"] = max(
            _to_int(cfg.get("loss_match_class_warmup_epochs", 0), 0),
            0,
        )
        cfg["loss_match_cost_l1"] = max(_to_float(cfg.get("loss_match_cost_l1", 1.0), 1.0), 0.0)
        cfg["loss_match_cost_iou"] = max(_to_float(cfg.get("loss_match_cost_iou", 2.0), 2.0), 0.0)
        cfg["loss_match_topk_per_gt"] = max(_to_int(cfg.get("loss_match_topk_per_gt", 0), 0), 0)
        cfg["loss_quality_focal_beta"] = max(
            _to_float(cfg.get("loss_quality_focal_beta", 2.0), 2.0),
            0.0,
        )

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
        cfg["postprocess_pre_nms_topk"] = max(
            _to_int(cfg.get("postprocess_pre_nms_topk", max(cfg["postprocess_nms_max_detections"], 1000)), max(cfg["postprocess_nms_max_detections"], 1000)),
            1,
        )
        cfg["postprocess_class_selection"] = str(
            cfg.get("postprocess_class_selection", "query_max") or "query_max"
        ).strip().lower()
        if cfg["postprocess_class_selection"] not in {"query_max", "flatten_topk", "all_class_topk", "query_class_topk"}:
            cfg["postprocess_class_selection"] = "query_max"
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
    if isinstance(batch_train, dict):
        video_ids = batch_train.get("video_id", [])
        if isinstance(video_ids, (list, tuple)) and len(video_ids) > 1:
            videos = []
            read_time = 0.0
            proc_time = 0.0
            for i in range(len(video_ids)):
                single = {}
                for key, value in batch_train.items():
                    if isinstance(value, (list, tuple)):
                        single[key] = [value[i]]
                    else:
                        single[key] = value
                prepared = prepare_video_batch(single, cfg, transform, zero_frame, cache_state, split_name)
                videos.append(prepared)
                read_time += float(prepared.get("read_time", 0.0))
                proc_time += float(prepared.get("proc_time", 0.0))
            return {
                "videos": videos,
                "read_time": read_time,
                "proc_time": proc_time,
            }

    t_start = time.time()
    video_folder = batch_train.get('video_folder', [None])[0]
    feature_path_from_dataset = batch_train.get('feature_path', [None])[0]
    labels = _to_cpu_tensor(batch_train['labels'][0])
    total_frames = _extract_scalar_int(batch_train['total_frames'][0])
    video_id = batch_train['video_id'][0]
    fps_value = batch_train.get('fps', [0.0])[0]
    if torch.is_tensor(fps_value):
        fps_value = fps_value.item()
    fps = float(fps_value or 0.0)
    duration = float(total_frames / fps) if fps > 0 else 0.0

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
            "fps": fps,
            "total_frames": total_frames,
            "duration": duration,
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
                    "fps": fps,
                    "total_frames": total_frames,
                    "duration": duration,
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
        "fps": fps,
        "total_frames": total_frames,
        "duration": duration,
        "source": "frames",
        "cache_pixel_dtype": None,
        "cache_normalize": False,
        "read_time": t_read,
        "proc_time": t_proc,
    }


def train_one_video(
    model,
    optimizer,
    criterion,
    video_data,
    cfg,
    scaler=None,
    ema_state=None,
    ema_decay=0.999,
):
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
        state = model.init_state(1, device=cfg["device"], dtype=torch.float32)
        for clip_idx, clip_tensor in enumerate(video_clips):
            clip_tensor = clip_tensor.to(cfg["device"], non_blocking=True)
            clip_tensor = decode_clip_for_model(clip_tensor, video_data, cfg)
            if video_data.get("source") not in {"cache", "feature_cache"}:
                clip_tensor = _apply_imagenet_normalization(clip_tensor)
            should_decode = clip_idx + 1 == len(video_clips)
            output, state = model(
                clip_tensor,
                state=state,
                decode=should_decode,
                detach_state=False,
                dn_targets=labels if (
                    should_decode and cfg.get("denoising_enabled", False)
                ) else None,
            )
            pbar_in.update(1)
        memory_auxiliary = model.pop_memory_auxiliary()

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
        auxiliary_loss = _memory_auxiliary_loss(memory_auxiliary, labels, cfg, criterion)
        if auxiliary_loss is not None:
            loss = loss + auxiliary_loss
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
        _update_ema_state(model, ema_state, ema_decay)
    else:
        loss.backward()
        if cfg.get("grad_clip_norm", 0.0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg["grad_clip_norm"])
        optimizer.step()
        _update_ema_state(model, ema_state, ema_decay)

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


def _prepare_clip_for_model(clip_tensor, video_data, cfg):
    clip_tensor = clip_tensor.to(cfg["device"], non_blocking=True)
    if video_data.get("source") == "feature_cache":
        return clip_tensor.to(torch.float32).contiguous()
    clip_tensor = decode_clip_for_model(clip_tensor, video_data, cfg)
    if video_data.get("source") != "cache":
        clip_tensor = _apply_imagenet_normalization(clip_tensor)
    return clip_tensor


def _canonical_segments_for_loss(seg):
    start = torch.minimum(seg[..., 0], seg[..., 1])
    end = torch.maximum(seg[..., 0], seg[..., 1])
    return torch.stack([start, end], dim=-1)


def _aligned_temporal_iou(pred_seg, target_seg, eps=1e-6):
    pred_seg = _canonical_segments_for_loss(pred_seg)
    target_seg = _canonical_segments_for_loss(target_seg)
    inter_start = torch.maximum(pred_seg[:, 0], target_seg[:, 0])
    inter_end = torch.minimum(pred_seg[:, 1], target_seg[:, 1])
    inter = (inter_end - inter_start).clamp(min=0.0)
    pred_len = (pred_seg[:, 1] - pred_seg[:, 0]).clamp(min=eps)
    target_len = (target_seg[:, 1] - target_seg[:, 0]).clamp(min=eps)
    return inter / (pred_len + target_len - inter + eps)


def _quality_focal_bce_with_logits(logits, target, beta):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    prob = torch.sigmoid(logits)
    modulating = (target - prob).abs().pow(float(beta))
    return bce * modulating


def _denoising_direct_loss(outputs, targets, mask, cfg, criterion):
    if outputs is None or targets is None or mask is None:
        return None
    if outputs.dim() == 3 and outputs.size(0) == 1:
        outputs = outputs[0]
    if targets.dim() == 3 and targets.size(0) == 1:
        targets = targets[0]
    if mask.dim() == 2 and mask.size(0) == 1:
        mask = mask[0]
    if outputs.dim() != 2 or targets.dim() != 2 or mask.dim() != 1:
        return None
    mask = mask.to(device=outputs.device, dtype=torch.bool)
    if not mask.any():
        return outputs.new_zeros(())

    outputs = outputs[mask]
    targets = targets.to(device=outputs.device, dtype=outputs.dtype)[mask]
    pred_quality_logits = outputs[:, 0]
    pred_seg = _canonical_segments_for_loss(outputs[:, 1:3])
    pred_cls = outputs[:, 3:]
    target_seg = _canonical_segments_for_loss(targets[:, :2])
    target_cls = targets[:, 2:2 + pred_cls.size(-1)].clamp(min=0.0, max=1.0)

    eps = float(getattr(criterion, "eps", cfg.get("eps", 1e-6)))
    aligned_iou = _aligned_temporal_iou(pred_seg, target_seg, eps=eps)
    quality_target = aligned_iou.detach().clamp(min=0.0, max=1.0).to(
        dtype=pred_quality_logits.dtype
    )
    cls_target = target_cls * quality_target.unsqueeze(-1)
    normalizer = max(int(mask.sum().item()), 1)

    beta = float(
        getattr(
            criterion,
            "quality_focal_beta",
            cfg.get("loss_quality_focal_beta", 2.0),
        )
    )
    quality_loss = (
        _quality_focal_bce_with_logits(
            pred_quality_logits,
            quality_target,
            beta,
        ).sum()
        / normalizer
    )
    cls_loss = (
        _quality_focal_bce_with_logits(pred_cls, cls_target, beta).sum()
        / normalizer
    )
    loc_reg = F.smooth_l1_loss(
        pred_seg,
        target_seg,
        reduction="sum",
    ) / normalizer
    iou_loss = 1.0 - aligned_iou.mean()
    loc_loss = loc_reg + float(
        getattr(criterion, "iou_loc_weight", cfg.get("loss_iou_weight", 1.0))
    ) * iou_loss

    return (
        float(getattr(criterion, "conf_weight", cfg.get("conf_weight", 1.0)))
        * quality_loss
        + float(getattr(criterion, "loc_weight", cfg.get("loc_weight", 1.0)))
        * loc_loss
        + float(getattr(criterion, "cls_weight", cfg.get("cls_weight", 1.0)))
        * cls_loss
    )


def _memory_auxiliary_loss(auxiliary, labels, cfg, criterion):
    if not auxiliary:
        return None

    labels = labels[0] if labels.dim() == 3 and labels.size(0) == 1 else labels
    loss_terms = []

    if (
        cfg.get("memory_transformer_aux_loss_enabled", False)
        and "memory_transformer_aux_outputs" in auxiliary
    ):
        aux_outputs = auxiliary["memory_transformer_aux_outputs"]
        if aux_outputs.dim() == 4 and aux_outputs.size(0) == 1:
            aux_outputs = aux_outputs[0]
        if aux_outputs.dim() == 3:
            aux_weight = float(cfg.get("memory_transformer_aux_loss_weight", 0.4))
            aux_losses = [
                criterion(aux_outputs[layer_idx], labels)
                for layer_idx in range(aux_outputs.size(0))
            ]
            if aux_losses:
                loss_terms.append(aux_weight * torch.stack(aux_losses).mean())

    if (
        cfg.get("memory_transformer_encoder_proposal_loss_enabled", False)
        and "memory_transformer_encoder_proposal_outputs" in auxiliary
    ):
        proposal_outputs = auxiliary["memory_transformer_encoder_proposal_outputs"]
        if proposal_outputs.dim() == 3 and proposal_outputs.size(0) == 1:
            proposal_outputs = proposal_outputs[0]
        if proposal_outputs.dim() == 2:
            proposal_weight = float(
                cfg.get("memory_transformer_encoder_proposal_loss_weight", 0.3)
            )
            loss_terms.append(proposal_weight * criterion(proposal_outputs, labels))

    if (
        cfg.get("memory_transformer_deep_prior_loss_enabled", False)
        and "memory_transformer_deep_prior_outputs" in auxiliary
    ):
        deep_prior_outputs = auxiliary["memory_transformer_deep_prior_outputs"]
        if deep_prior_outputs.dim() == 3 and deep_prior_outputs.size(0) == 1:
            deep_prior_outputs = deep_prior_outputs[0]
        if deep_prior_outputs.dim() == 2:
            deep_prior_weight = float(
                cfg.get("memory_transformer_deep_prior_loss_weight", 0.2)
            )
            loss_terms.append(deep_prior_weight * criterion(deep_prior_outputs, labels))

    if (
        cfg.get("denoising_enabled", False)
        and "memory_transformer_denoising_outputs" in auxiliary
        and "memory_transformer_denoising_targets" in auxiliary
        and "memory_transformer_denoising_mask" in auxiliary
    ):
        dn_outputs = auxiliary["memory_transformer_denoising_outputs"]
        dn_targets = auxiliary["memory_transformer_denoising_targets"]
        dn_mask = auxiliary["memory_transformer_denoising_mask"]
        dn_loss = _denoising_direct_loss(
            dn_outputs,
            dn_targets,
            dn_mask,
            cfg,
            criterion,
        )
        if (
            dn_loss is not None
            and "memory_transformer_denoising_aux_outputs" in auxiliary
        ):
            dn_aux_outputs = auxiliary["memory_transformer_denoising_aux_outputs"]
            if dn_aux_outputs.dim() == 4 and dn_aux_outputs.size(0) == 1:
                dn_aux_outputs = dn_aux_outputs[0]
            if dn_aux_outputs.dim() == 3:
                dn_aux_losses = [
                    _denoising_direct_loss(
                        dn_aux_outputs[layer_idx],
                        dn_targets,
                        dn_mask,
                        cfg,
                        criterion,
                    )
                    for layer_idx in range(dn_aux_outputs.size(0))
                ]
                dn_aux_losses = [
                    loss for loss in dn_aux_losses if loss is not None
                ]
                if dn_aux_losses:
                    dn_loss = dn_loss + float(
                        cfg.get(
                            "denoising_aux_loss_weight",
                            cfg.get("memory_transformer_aux_loss_weight", 0.4),
                        )
                    ) * torch.stack(dn_aux_losses).mean()
        if dn_loss is not None:
            loss_terms.append(
                float(cfg.get("denoising_loss_weight", 1.0)) * dn_loss
            )

    if (
        cfg.get("memory_auxiliary_enabled", False)
        and "shallow_boundary_logits" in auxiliary
        and "deep_class_logits" in auxiliary
    ):
        boundary_logits = auxiliary["shallow_boundary_logits"]
        class_logits = auxiliary["deep_class_logits"]
        if boundary_logits.dim() == 3 and boundary_logits.size(0) == 1:
            boundary_logits = boundary_logits[0]
        if class_logits.dim() == 2 and class_logits.size(0) == 1:
            class_logits = class_logits[0]

        temporal_size = boundary_logits.size(-1)
        positions = (
            torch.arange(
                temporal_size,
                device=boundary_logits.device,
                dtype=boundary_logits.dtype,
            )
            + 0.5
        ) / float(temporal_size)
        boundary_target = torch.zeros_like(boundary_logits)
        class_target = torch.zeros_like(class_logits)

        if labels.numel() > 0:
            starts = torch.minimum(labels[:, 0], labels[:, 1]).to(
                device=boundary_logits.device,
                dtype=boundary_logits.dtype,
            )
            ends = torch.maximum(labels[:, 0], labels[:, 1]).to(
                device=boundary_logits.device,
                dtype=boundary_logits.dtype,
            )
            sigma = float(cfg.get("shallow_boundary_aux_sigma", 0.02))
            start_target = torch.exp(
                -0.5 * ((positions[:, None] - starts[None, :]) / sigma).square()
            ).amax(dim=1)
            end_target = torch.exp(
                -0.5 * ((positions[:, None] - ends[None, :]) / sigma).square()
            ).amax(dim=1)
            boundary_target = torch.stack([start_target, end_target], dim=0)

            gt_classes = labels[:, 2:2 + class_logits.size(-1)].to(
                device=class_logits.device,
                dtype=class_logits.dtype,
            )
            if gt_classes.numel() > 0:
                class_target = gt_classes.amax(dim=0).clamp(min=0.0, max=1.0)

        boundary_bce = torch.nn.functional.binary_cross_entropy_with_logits(
            boundary_logits,
            boundary_target,
            reduction="none",
        )
        boundary_weight = 1.0 + float(
            cfg.get("shallow_boundary_aux_pos_weight", 4.0)
        ) * boundary_target
        boundary_loss = (boundary_bce * boundary_weight).mean()
        class_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            class_logits,
            class_target,
        )
        loss_terms.append(
            float(cfg.get("shallow_boundary_aux_weight", 0.1)) * boundary_loss
            + float(cfg.get("deep_class_aux_weight", 0.1)) * class_loss
        )

    if not loss_terms:
        return None
    return torch.stack(loss_terms).sum()


def train_video_batch(
    model,
    optimizer,
    criterion,
    batch_video_data,
    cfg,
    scaler=None,
    ema_state=None,
    ema_decay=0.999,
):
    videos = batch_video_data.get("videos", None)
    if not videos:
        return train_one_video(
            model,
            optimizer,
            criterion,
            batch_video_data,
            cfg,
            scaler=scaler,
            ema_state=ema_state,
            ema_decay=ema_decay,
        )

    videos = [video for video in videos if len(video.get("video_clips", [])) > 0]
    if not videos:
        return 0.0, []

    batch_size = len(videos)
    max_clips = max(len(video["video_clips"]) for video in videos)
    amp_enabled = bool(cfg.get("amp_enabled", False))
    amp_dtype = torch.float16
    if str(cfg.get("amp_dtype", "fp16")).lower() in {"bf16", "bfloat16"}:
        amp_dtype = torch.bfloat16

    outputs_for_loss = []
    labels_for_loss = []
    auxiliary_for_loss = []
    videos_for_loss = []
    records = []

    with torch.amp.autocast(device_type="cuda", enabled=amp_enabled, dtype=amp_dtype):
        state = model.init_state(batch_size, device=cfg["device"], dtype=torch.float32)
        for step in range(max_clips):
            clip_batch = []
            decode_mask = []
            active_indices = []
            for video_idx, video in enumerate(videos):
                video_clips = video["video_clips"]
                is_active = step < len(video_clips)
                if not is_active:
                    continue
                active_indices.append(video_idx)
                decode_mask.append(step + 1 == len(video_clips))
                clip_tensor = _prepare_clip_for_model(video_clips[step], video, cfg)
                clip_batch.append(clip_tensor)

            if not clip_batch:
                continue

            clip_batch = torch.cat(clip_batch, dim=0)
            active_index_tensor = torch.tensor(active_indices, dtype=torch.long, device=cfg["device"])
            active_state = _select_state(state, active_index_tensor)
            decode_mask_tensor = torch.tensor(decode_mask, dtype=torch.bool, device=cfg["device"])
            dn_targets = None
            if cfg.get("denoising_enabled", False) and any(decode_mask):
                dn_targets = [
                    videos[video_idx]["labels"].to(cfg["device"])
                    if decode_mask[local_idx]
                    else None
                    for local_idx, video_idx in enumerate(active_indices)
                ]

            output, active_new_state = model(
                clip_batch,
                state=active_state,
                decode_mask=decode_mask_tensor,
                detach_state=False,
                dn_targets=dn_targets,
            )
            memory_auxiliary = model.pop_memory_auxiliary()
            state = _scatter_state(state, active_index_tensor, active_new_state)

            if output is not None and output.numel() > 0:
                ending_indices = [i for i, flag in enumerate(decode_mask) if flag]
                for local_idx, active_pos in enumerate(ending_indices):
                    video_idx = active_indices[active_pos]
                    labels = videos[video_idx]["labels"].to(cfg["device"])
                    outputs_for_loss.append(output[local_idx])
                    labels_for_loss.append(labels)
                    auxiliary_for_loss.append(
                        {
                            name: value[local_idx]
                            for name, value in memory_auxiliary.items()
                        }
                        if memory_auxiliary is not None
                        else None
                    )
                    videos_for_loss.append(videos[video_idx])

        if not outputs_for_loss:
            return 0.0, []

        loss_terms = []
        for output, labels, auxiliary in zip(
            outputs_for_loss,
            labels_for_loss,
            auxiliary_for_loss,
        ):
            loss_term = criterion(output, labels)
            auxiliary_loss = _memory_auxiliary_loss(auxiliary, labels, cfg, criterion)
            if auxiliary_loss is not None:
                loss_term = loss_term + auxiliary_loss
            loss_terms.append(loss_term)
        loss = torch.stack(loss_terms).mean()

    if not torch.isfinite(loss):
        optimizer.zero_grad(set_to_none=True)
        return 0.0, [{"skipped_non_finite": True}]

    optimizer.zero_grad(set_to_none=True)
    if scaler is not None and scaler.is_enabled():
        scaler.scale(loss).backward()
        if cfg.get("grad_clip_norm", 0.0) > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg["grad_clip_norm"])
        scaler.step(optimizer)
        scaler.update()
        _update_ema_state(model, ema_state, ema_decay)
    else:
        loss.backward()
        if cfg.get("grad_clip_norm", 0.0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg["grad_clip_norm"])
        optimizer.step()
        _update_ema_state(model, ema_state, ema_decay)

    for video, output, labels in zip(videos_for_loss, outputs_for_loss, labels_for_loss):
        rec = _extract_pred_and_gt(
            output,
            labels,
            cfg["class_num"],
            conf_threshold=float(cfg.get("eval_conf_threshold", 0.0)),
            **_nms_kwargs_from_cfg(cfg),
        )
        rec["video_id"] = video.get("video_id", "")
        rec["skipped_non_finite"] = False
        records.append(rec)

    return float(loss.item()), records




def train_model(args):
    cfg = load_config(args.cfg)
    
    cfg = check_config(cfg, args.device)
    if cfg == False:
        return False
    _set_reproducibility(
        cfg["seed"],
        deterministic=cfg.get("deterministic_training", False),
    )
    
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
    cfg["config_source"] = str(Path(args.cfg).expanduser().resolve())
    resolved_config_path = Path(cfg["output_dir"]) / "resolved_config.yml"
    resolved_config = dict(cfg)
    resolved_config["device"] = str(cfg["device"])
    with resolved_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            resolved_config,
            handle,
            sort_keys=False,
            allow_unicode=True,
        )
    train_video_num = len(train_dataset.video_list)
    val_video_num = len(val_dataset.video_list)
    
    ly.box(f"{cfg['dataset_cache_dir'].split('/')[-1]}\nNumber of training videos: {train_video_num}, Number of training frames: {train_dataset.dataset_totle_frame}\nNumber of validation videos: {val_video_num}, Number of validation frames: {val_dataset.dataset_totle_frame}")

    first_epoch_first_video_idx = _find_video_dataset_index(
        train_dataset,
        cfg.get("first_epoch_first_video_id", ""),
    )
    first_epoch_largest_video_indices = _find_largest_video_indices(
        train_dataset,
        cfg.get("batch_size", 1),
    )

    def _build_train_dataloader(prioritize_first_batch=False, epoch_seed=0):
        generator = torch.Generator()
        generator.manual_seed(int(epoch_seed))
        if prioritize_first_batch and bool(cfg.get("first_epoch_largest_batch_enabled", True)):
            sampler = _PriorityIndicesSampler(
                dataset_size=len(train_dataset),
                first_indices=first_epoch_largest_video_indices,
                seed=epoch_seed,
            )
            return DataLoader(
                train_dataset,
                batch_size=cfg["batch_size"],
                shuffle=False,
                sampler=sampler,
                num_workers=cfg["dataloader_num_workers"],
                pin_memory=cfg["dataloader_pin_memory"],
                collate_fn=_video_collate,
                generator=generator,
            )

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
                collate_fn=_video_collate,
                generator=generator,
            )

        return DataLoader(
            train_dataset,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=cfg["dataloader_num_workers"],
            pin_memory=cfg["dataloader_pin_memory"],
            collate_fn=_video_collate,
            generator=generator,
        )

    dataloader_train = _build_train_dataloader(
        prioritize_first_batch=False,
        epoch_seed=cfg["seed"],
    )
    val_generator = torch.Generator()
    val_generator.manual_seed(cfg["seed"] + 1)
    dataloader_val = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["dataloader_num_workers"],
        pin_memory=cfg["dataloader_pin_memory"],
        collate_fn=_video_collate,
        generator=val_generator,
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
        force_gt_match=cfg.get("loss_force_gt_match", True),
        missed_gt_weight=cfg.get("loss_missed_gt_weight", 0.5),
        matcher=cfg.get("loss_matcher", "hungarian"),
        match_cost_class=cfg.get("loss_match_cost_class", 1.0),
        match_cost_l1=cfg.get("loss_match_cost_l1", 1.0),
        match_cost_iou=cfg.get("loss_match_cost_iou", 2.0),
        match_topk_per_gt=cfg.get("loss_match_topk_per_gt", 0),
        quality_focal_beta=cfg.get("loss_quality_focal_beta", 2.0),
        eps=cfg.get("loss_eps", cfg.get("eps", 1e-6))
    )
    optimizer_cls = torch.optim.__dict__[cfg["optimizer"]]
    optimizer_param_groups = _build_optimizer_param_groups(model, cfg)
    try:
        optimizer = optimizer_cls(optimizer_param_groups, lr=cfg["lr"])
    except TypeError:
        for group in optimizer_param_groups:
            group.pop("weight_decay", None)
        optimizer = optimizer_cls(optimizer_param_groups, lr=cfg["lr"])
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
            min_lr=_optimizer_min_lrs(optimizer, cfg["lr_plateau_min_lr"]),
            eps=cfg["lr_plateau_eps"],
        )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.get("amp_enabled", False))
    zero_frame = build_zero_frame(cfg)

    # checkpoint / resume support
    start_epoch = 0
    best_avg_map = 0.0
    resumed_ema_state = None
    if hasattr(args, "resume") and args.resume:
        resume_path = args.resume
        if os.path.isfile(resume_path):
            print(f"Loading checkpoint from {resume_path}")
            ckpt = torch.load(resume_path, map_location=cfg["device"])
            model.load_state_dict(ckpt.get("model", ckpt.get("state_dict", {})))
            resumed_ema_state = _load_ema_state_from_checkpoint(
                ckpt,
                cfg["device"],
            )
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
                    plateau_scheduler.min_lrs = _optimizer_min_lrs(
                        optimizer,
                        target_min_lr,
                    )
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

    ema_state = None
    if cfg.get("ema_enabled", False):
        ema_state = resumed_ema_state if resumed_ema_state is not None else _init_ema_state(model)
        print(
            f">>> EMA enabled: decay={cfg.get('ema_decay', 0.999):.5f} "
            f"| eval={cfg.get('ema_eval_enabled', True)} "
            f"| resumed={resumed_ema_state is not None}"
        )

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
    print(
        f">>> Reproducibility: seed={cfg.get('seed', 3407)} "
        f"| deterministic={cfg.get('deterministic_training', False)}"
    )
    print(
        ">>> Optimizer groups: "
        + " | ".join(
            f"{group.get('group_name', index)}: "
            f"lr_scale={group.get('lr_scale', 1.0):.3f}, "
            f"weight_decay={group.get('weight_decay', 0.0):.6g}"
            for index, group in enumerate(optimizer.param_groups)
        )
    )
    print(f">>> DataLoader batch size: train={dataloader_train.batch_size}, val={dataloader_val.batch_size}")
    print(f">>> Validation mAP scope: {cfg.get('map_eval_scope', 'global')}")
    print(f">>> Detection score mode: {cfg.get('detection_score_mode', 'sigmoid_class')}")
    print(
        f">>> Memory Transformer head: {cfg.get('memory_transformer_head_enabled', False)} "
        f"| dim={cfg.get('memory_transformer_dim', 256)} "
        f"| heads={cfg.get('memory_transformer_heads', 8)} "
        f"| enc={cfg.get('memory_transformer_encoder_layers', 2)} "
        f"| dec={cfg.get('memory_transformer_decoder_layers', 3)} "
        f"| ff={cfg.get('memory_transformer_ff_dim', 1024)} "
        f"| dropout={cfg.get('memory_transformer_dropout', 0.0):.3f} "
        f"| aux={cfg.get('memory_transformer_aux_loss_enabled', False)} "
        f"| aux_weight={cfg.get('memory_transformer_aux_loss_weight', 0.0):.3f} "
        f"| enc_prop={cfg.get('memory_transformer_encoder_proposal_enabled', False)} "
        f"| enc_prop_loss={cfg.get('memory_transformer_encoder_proposal_loss_enabled', False)} "
        f"| enc_prop_weight={cfg.get('memory_transformer_encoder_proposal_loss_weight', 0.0):.3f} "
        f"| query_mode={cfg.get('memory_transformer_query_mode', 'fixed')} "
        f"| hybrid_fixed={cfg.get('memory_transformer_hybrid_fixed_queries', 0)} "
        f"| hybrid_prop={cfg.get('memory_transformer_hybrid_proposal_queries', 0)} "
        f"| iter_refine={cfg.get('memory_transformer_iterative_refine_enabled', False)} "
        f"| iter_detach={cfg.get('memory_transformer_iterative_refine_detach', True)} "
        f"| deep_prior={cfg.get('memory_transformer_deep_prior_enabled', False)} "
        f"| deep_prior_w={cfg.get('memory_transformer_deep_prior_loss_weight', 0.0):.3f} "
        f"| deep_prior_scale={cfg.get('memory_transformer_deep_prior_context_scale', 0.0):.2f} "
        f"| lr_scale={cfg.get('memory_transformer_lr_scale', 1.0):.3f} "
        f"| wd_scale={cfg.get('memory_transformer_weight_decay_scale', 1.0):.3f}"
    )
    print(
        f">>> Denoising queries: {cfg.get('denoising_enabled', False)} "
        f"| groups={cfg.get('denoising_groups', 0)} "
        f"| max_queries={cfg.get('denoising_max_queries', 0)} "
        f"| label_noise={cfg.get('denoising_label_noise_ratio', 0.0):.3f} "
        f"| box_noise={cfg.get('denoising_box_noise_scale', 0.0):.3f} "
        f"| loss_weight={cfg.get('denoising_loss_weight', 0.0):.3f} "
        f"| aux_weight={cfg.get('denoising_aux_loss_weight', 0.0):.3f}"
    )
    print(
        f">>> Temporal references: {cfg.get('use_temporal_references', True)} "
        f"| widths={cfg.get('temporal_ref_widths', [cfg.get('temporal_ref_width', 0.03)])} "
        f"| query_counts={cfg.get('temporal_ref_query_counts', [cfg.get('max_detection_num')])}"
    )
    print(
        f">>> Reference attention: {cfg.get('reference_attention_enabled', False)} "
        f"| layers={cfg.get('reference_attention_layers', 3)} "
        f"| points={cfg.get('reference_attention_points', 7)} "
        f"| context_scale={cfg.get('reference_attention_context_scale', 1.5):.2f} "
        f"| context_dim={cfg.get('reference_attention_context_dim', 128)}"
    )
    print(
        f">>> Memory persistence: shallow_bias={cfg.get('shallow_gate_init_bias', 0.0):.3f} "
        f"| deep_bias={cfg.get('deep_gate_init_bias', -2.2):.3f} "
        f"| proposal_dropout={cfg.get('memory_proposal_dropout', 0.0):.3f} "
        f"| diagnostics={cfg.get('memory_diagnostics_enabled', False)} "
        f"| diag_stride={cfg.get('memory_diagnostics_stride', 4)}"
    )
    print(
        f">>> Joint memory detection: {cfg.get('joint_memory_detection_enabled', False)} "
        f"| dim={cfg.get('joint_memory_dim', 64)} "
        f"| dropout={cfg.get('joint_memory_dropout', 0.0):.3f} "
        f"| lr_scale={cfg.get('joint_memory_lr_scale', 1.0):.3f} "
        f"| wd_scale={cfg.get('joint_memory_weight_decay_scale', 1.0):.3f} "
        f"| points={cfg.get('joint_memory_points', 7)} "
        f"| context_scales={cfg.get('joint_memory_shallow_context_scale', 1.0):.2f}/"
        f"{cfg.get('joint_memory_deep_context_scale', 3.0):.2f} "
        f"| max_residual={cfg.get('joint_memory_max_residual_scale', 0.1):.3f}"
    )
    print(
        f">>> Memory proposal refinement: "
        f"{cfg.get('memory_proposal_refine_enabled', False)} "
        f"| dim={cfg.get('memory_proposal_refine_dim', 64)} "
        f"| points={cfg.get('memory_proposal_refine_points', 7)} "
        f"| context_scales="
        f"{cfg.get('memory_proposal_refine_shallow_context_scale', 1.0):.2f}/"
        f"{cfg.get('memory_proposal_refine_deep_context_scale', 3.0):.2f} "
        f"| max_shift="
        f"{cfg.get('memory_proposal_refine_max_boundary_shift', 0.25):.2f}/"
        f"{cfg.get('memory_proposal_refine_max_center_shift', 0.25):.2f} "
        f"| gate_bias={cfg.get('memory_proposal_refine_gate_bias', -6.0):.2f}"
    )
    print(
        f">>> Training-only memory auxiliary: "
        f"{cfg.get('memory_auxiliary_enabled', False)} "
        f"| shallow_boundary_weight={cfg.get('shallow_boundary_aux_weight', 0.0):.3f} "
        f"| deep_class_weight={cfg.get('deep_class_aux_weight', 0.0):.3f}"
    )
    print(
        f">>> Postprocess class selection: {cfg.get('postprocess_class_selection', 'query_max')} "
        f"| pre_nms_topk={cfg.get('postprocess_pre_nms_topk', None)} "
        f"| max_det={cfg.get('postprocess_nms_max_detections', None)}"
    )
    print(f">>> Loss force GT match: {cfg.get('loss_force_gt_match', True)}")
    print(
        f">>> Loss matcher: {cfg.get('loss_matcher', 'hungarian')} "
        f"| backend={getattr(criterion, 'hungarian_backend', 'n/a')} "
        f"| cls={cfg.get('loss_match_cost_class', 1.0):.3f} "
        f"| cls_warmup={cfg.get('loss_match_class_warmup_epochs', 0)} epochs "
        f"| l1={cfg.get('loss_match_cost_l1', 1.0):.3f} "
        f"| iou={cfg.get('loss_match_cost_iou', 2.0):.3f} "
        f"| topk_per_gt={cfg.get('loss_match_topk_per_gt', 0)}"
    )
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
        model.reset_memory_diagnostics()
        class_cost_warmup_epochs = cfg.get("loss_match_class_warmup_epochs", 0)
        criterion.match_cost_class = (
            0.0
            if epoch < class_cost_warmup_epochs
            else cfg.get("loss_match_cost_class", 1.0)
        )
        prioritize_first_batch = (
            epoch == start_epoch
            and (
                first_epoch_first_video_idx is not None
                or (
                    bool(cfg.get("first_epoch_largest_batch_enabled", True))
                    and bool(first_epoch_largest_video_indices)
                )
            )
        )
        dataloader_train_epoch = _build_train_dataloader(
            prioritize_first_batch=prioritize_first_batch,
            epoch_seed=cfg["seed"] + epoch,
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
        print(
            f">>> Epoch {epoch + 1}/{total_epochs} | LR: {current_lr:.8f} "
            f"| match_cls_cost: {criterion.match_cost_class:.3f}"
        )
        if prioritize_first_batch and bool(cfg.get("first_epoch_largest_batch_enabled", True)):
            selected = [
                train_dataset.video_list[idx]["video_id"]
                for idx in first_epoch_largest_video_indices
            ]
            print(
                ">>> Forcing largest videos into the first batch of this run: "
                + ", ".join(map(str, selected))
            )
        elif prioritize_first_batch:
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
                    loss_val, rec = train_video_batch(
                        model,
                        optimizer,
                        criterion,
                        current_video_data,
                        cfg,
                        scaler=scaler,
                        ema_state=ema_state,
                        ema_decay=cfg.get("ema_decay", 0.999),
                    )
                    rec_list = rec if isinstance(rec, list) else [rec]
                    if any(item.get("skipped_non_finite", False) for item in rec_list):
                        skipped_non_finite_batches += 1
                        pbar.update(max(len(rec_list), 1))
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
                    _, _, m03 = _compute_metrics_at_iou(rec_list, cfg["class_num"], 0.3, conf_threshold=eval_conf_thr)
                    _, _, m04 = _compute_metrics_at_iou(rec_list, cfg["class_num"], 0.4, conf_threshold=eval_conf_thr)
                    p, r, m05 = _compute_metrics_at_iou(rec_list, cfg["class_num"], 0.5, conf_threshold=eval_conf_thr)
                    _, _, m06 = _compute_metrics_at_iou(rec_list, cfg["class_num"], 0.6, conf_threshold=eval_conf_thr)
                    _, _, m07 = _compute_metrics_at_iou(rec_list, cfg["class_num"], 0.7, conf_threshold=eval_conf_thr)
                    _, _, m09 = _compute_metrics_at_iou(rec_list, cfg["class_num"], 0.9, conf_threshold=eval_conf_thr)
                    avgm = (m03 + m04 + m05 + m06 + m07 + m09) / 6.0
                    train_video_records.extend(rec_list)


                    # disable automatic sorting so keys appear in the order we specify
                    timing_bar.set_postfix_str(
                        f"[ read:{read:.1f}s  pre:{prep:.1f}s  train:{t_train:.1f}s  loss:{loss_val:.4f}  Prec:{p:.3f}  Rec:{r:.3f}  m30:{m03:.3f}  m40:{m04:.3f}  m50:{m05:.3f}  m60:{m06:.3f}  m70:{m07:.3f}  m90:{m09:.3f}  Avg_m:{avgm:.3f} ]"
                    )
                    batch_idx += 1
                    pbar.update(len(rec_list))

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
                loss_val, rec = train_video_batch(
                    model,
                    optimizer,
                    criterion,
                    video_data,
                    cfg,
                    scaler=scaler,
                    ema_state=ema_state,
                    ema_decay=cfg.get("ema_decay", 0.999),
                )
                rec_list = rec if isinstance(rec, list) else [rec]
                if any(item.get("skipped_non_finite", False) for item in rec_list):
                    skipped_non_finite_batches += 1
                    batch_idx += 1
                    pbar.update(max(len(rec_list), 1))
                    continue

                valid_train_batches += 1
                total_loss += float(loss_val)
                t_train = time.time() - t1

                read = video_data.get("read_time", t_data)
                prep = video_data.get("proc_time", 0.0)

                _, _, m03 = _compute_metrics_at_iou(rec_list, cfg["class_num"], 0.3, conf_threshold=eval_conf_thr)
                _, _, m04 = _compute_metrics_at_iou(rec_list, cfg["class_num"], 0.4, conf_threshold=eval_conf_thr)
                _, _, m05 = _compute_metrics_at_iou(rec_list, cfg["class_num"], 0.5, conf_threshold=eval_conf_thr)
                p, r, m06 = _compute_metrics_at_iou(rec_list, cfg["class_num"], 0.6, conf_threshold=eval_conf_thr)
                _, _, m07 = _compute_metrics_at_iou(rec_list, cfg["class_num"], 0.7, conf_threshold=eval_conf_thr)
                _, _, m09 = _compute_metrics_at_iou(rec_list, cfg["class_num"], 0.9, conf_threshold=eval_conf_thr)
                avgm = (m03 + m04 + m05 + m06 + m07 + m09) / 6.0
                train_video_records.extend(rec_list)

                # disable automatic sorting so keys appear in the order we specify
                timing_bar.set_postfix(
                    read=f"{read:.3f}s", prep=f"{prep:.3f}s", backward=f"{t_train:.3f}s",
                    train_loss=f"{loss_val:.4f}", precision=f"{p:.3f}", recall=f"{r:.3f}",
                    m03=f"{m03:.3f}", m04=f"{m04:.3f}", m05=f"{m05:.3f}", m06=f"{m06:.3f}", m07=f"{m07:.3f}", m09=f"{m09:.3f}", avgm=f"{avgm:.3f}"
                )
                batch_idx += 1
                pbar.update(len(rec_list))

        pbar.close()
        avg_loss = total_loss / max(valid_train_batches, 1)
        timing_bar.close()
        train_memory_diag = model.get_memory_diagnostics(reset=True)
        ema_eval_active = (
            ema_state is not None
            and cfg.get("ema_enabled", False)
            and cfg.get("ema_eval_enabled", True)
        )
        ema_backup = None
        if ema_eval_active:
            ema_backup = _swap_to_model_state(model, ema_state)
        try:
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
        finally:
            if ema_backup is not None:
                _restore_model_state(model, ema_backup)
        val_memory_diag = model.get_memory_diagnostics(reset=True)
        # Keep model selection, scheduling, and reporting on the evaluator's
        # configured Avg. mAP protocol.
        avg_map = float(val_metrics.get('avg_map', 0.0))

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
            avgm = (m03 + m04 + m05 + m06 + m07) / 5.0
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
        if cfg.get("memory_diagnostics_enabled", False):
            print(
                " => Memory diagnostics:\n"
                f"| Train shallow gate/change/retention: "
                f"{train_memory_diag['shallow_gate_mean']:.4f} / "
                f"{train_memory_diag['shallow_state_change_ratio']:.4f} / "
                f"{train_memory_diag['shallow_state_retention']:.4f} | "
                f"deep: {train_memory_diag['deep_gate_mean']:.4f} / "
                f"{train_memory_diag['deep_state_change_ratio']:.4f} / "
                f"{train_memory_diag['deep_state_retention']:.4f} | "
                f"S-D cosine: {train_memory_diag['shallow_deep_cosine']:.4f} |\n"
                f"| Val   shallow gate/change/retention: "
                f"{val_memory_diag['shallow_gate_mean']:.4f} / "
                f"{val_memory_diag['shallow_state_change_ratio']:.4f} / "
                f"{val_memory_diag['shallow_state_retention']:.4f} | "
                f"deep: {val_memory_diag['deep_gate_mean']:.4f} / "
                f"{val_memory_diag['deep_state_change_ratio']:.4f} / "
                f"{val_memory_diag['deep_state_retention']:.4f} | "
                f"S-D cosine: {val_memory_diag['shallow_deep_cosine']:.4f} |"
            )
            if cfg.get("joint_memory_detection_enabled", False):
                print(
                    "| Joint shallow/deep gates: "
                    f"{val_memory_diag['joint_shallow_gate_mean']:.4f} / "
                    f"{val_memory_diag['joint_deep_gate_mean']:.4f} | "
                    "loc/cls residual scales: "
                    f"{val_memory_diag['joint_loc_residual_scale']:.4f} / "
                    f"{val_memory_diag['joint_cls_residual_scale']:.4f} |"
                )
        
        record_metrics(Path(cfg["output_dir"]) / f"train_metrics.csv", 
                       [epoch + 1, avg_loss] + train_metrics + [val_metrics['loss'], val_metrics['precision'], val_metrics['recall'], 
                        val_metrics['map03'], val_metrics['map04'], val_metrics['map05'], val_metrics['map06'], val_metrics['map07'], val_metrics['map09'], avg_map])
        if cfg.get("memory_diagnostics_enabled", False):
            record_memory_diagnostics(
                Path(cfg["output_dir"]) / "memory_diagnostics.csv",
                epoch + 1,
                train_memory_diag,
                val_memory_diag,
            )
        # save checkpoints
        current_avg_map = float(avg_map)
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "plateau_scheduler": plateau_scheduler.state_dict() if plateau_scheduler is not None else None,
            "best_avg_map": best_avg_map,
        }
        if ema_state is not None:
            checkpoint["ema_model"] = _clone_state_dict(ema_state, device="cpu")
            checkpoint["ema_decay"] = float(cfg.get("ema_decay", 0.999))
        last_path = Path(cfg["output_dir"]) / "last_checkpoint.pt"
        torch.save(checkpoint, last_path)
        if current_avg_map > best_avg_map:
            best_avg_map = current_avg_map
            checkpoint["best_avg_map"] = best_avg_map
            best_checkpoint = dict(checkpoint)
            if (
                ema_state is not None
                and cfg.get("ema_enabled", False)
                and cfg.get("ema_eval_enabled", True)
            ):
                best_checkpoint["model"] = _clone_state_dict(
                    ema_state,
                    device="cpu",
                )
            best_path = Path(cfg["output_dir"]) / "best_checkpoint.pt"
            torch.save(best_checkpoint, best_path)
            print(
                f"{COLOR_GREEN}New best model "
                f"(avg_map={best_avg_map:.6f}) saved to {best_path}"
                f"{COLOR_RESET}"
            )
        print()

    return True
