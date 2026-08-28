"""Generate dataset cache using lz4-compressed payloads.

This variant writes each video's cache to a .lz4 file instead of a
plain .pt. The file contains a torch-saved dictionary that is run-length
compressed with lz4.frame.  Consumers must decompress before torch.load.
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import io
import lz4.frame
from PIL import Image
from torchvision.transforms import functional as TF
from tqdm import tqdm

# Make project-root imports work regardless of invocation cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_config
from utils.DataLoader import load_train_val_data


def normalize_cfg(cfg):
	cfg = dict(cfg or {})
	cfg.setdefault("view_frames_num", 30)
	cfg.setdefault("starting_frame_number", 1)
	cfg.setdefault("input_size", [720, 1280])
	cfg.setdefault("frames_dir", "")
	cfg.setdefault("annotations_json_path", "")
	cfg.setdefault("dataset_cache_dir", "./dataset_cache")
	cfg.setdefault("cache_pixel_dtype", "uint8")
	cfg.setdefault("cache_normalize", False)

	cfg["input_size"] = [int(cfg["input_size"][0]), int(cfg["input_size"][1])]
	cfg["view_frames_num"] = int(cfg["view_frames_num"])
	cfg["starting_frame_number"] = int(cfg["starting_frame_number"])
	cfg["cache_pixel_dtype"] = str(cfg["cache_pixel_dtype"]).lower()
	cfg["cache_normalize"] = bool(cfg["cache_normalize"])

	if cfg["cache_pixel_dtype"] not in {"int8", "uint8", "float32"}:
		raise ValueError(
			"cache_pixel_dtype must be one of: int8, uint8, float32"
		)

	return cfg


def build_zero_frame(cfg):
	h, w = cfg["input_size"]
	dtype = cfg["cache_pixel_dtype"]

	if dtype == "uint8":
		return torch.zeros(3, h, w, dtype=torch.uint8)
	if dtype == "int8":
		# int8 stores uint8 pixels with an offset: saved = pixel - 128.
		# Black pixel (0) is represented as -128.
		return torch.full((3, h, w), -128, dtype=torch.int8)
	# float32 (optionally normalized)
	return torch.zeros(3, h, w, dtype=torch.float32)


def convert_image_to_cache_tensor(img, cfg):
	"""Convert PIL image to cache tensor according to configured storage dtype."""
	h, w = cfg["input_size"]
	img = img.resize((w, h), resample=Image.BILINEAR)
	frame_u8 = TF.pil_to_tensor(img)  # [C, H, W], uint8 in [0, 255]

	dtype = cfg["cache_pixel_dtype"]
	if dtype == "uint8":
		return frame_u8.contiguous()
	if dtype == "int8":
		# Lossless uint8->int8 mapping by shifting range [0,255] to [-128,127].
		return (frame_u8.to(torch.int16) - 128).to(torch.int8).contiguous()

	frame_f32 = frame_u8.to(torch.float32)
	if cfg["cache_normalize"]:
		frame_f32 = frame_f32 / 255.0
	return frame_f32.contiguous()


def build_video_frames(video_folder, total_frames, cfg, zero_frame):
	"""Load every frame of one video, returning a single tensor [T,3,H,W] on CPU.

	The cache stores this full-frame tensor instead of pre-computed clips.
	Trainer will slice clips dynamically according to cfg.view_frames_num.
	"""
	frame_tensors = []

	for fid in range(total_frames):
		frame_path = Path(video_folder) / f"{fid + cfg['starting_frame_number']:06d}.jpg"
		try:
			with Image.open(frame_path) as img:
				frame_tensors.append(convert_image_to_cache_tensor(img.convert("RGB"), cfg))
		except (FileNotFoundError, OSError):
			frame_tensors.append(zero_frame)

	if frame_tensors:
		# stack into tensor [T,3,H,W]
		return torch.stack(frame_tensors, dim=0)
	# empty video -> return zero-length tensor
	return torch.empty((0, 3, cfg['input_size'][0], cfg['input_size'][1]))


from concurrent.futures import ThreadPoolExecutor, as_completed

def cache_one_split(dataset, split_name, cache_root, cfg, zero_frame, overwrite=False):
    split_dir = cache_root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    skipped = 0

    # helper for a single sample; returns (gen, skip)
    def process_sample(sample):
        video_id = sample["video_id"]
        video_folder = sample["video_folder"]
        total_frames = int(sample["total_frames"])
        fps = sample["fps"]
        labels = sample["labels"].cpu()

        cache_path = split_dir / f"{video_id}.lz4"
        if cache_path.exists() and not overwrite:
            return 0, 1

        # load every frame once and store as full tensor
        frames_tensor = build_video_frames(video_folder, total_frames, cfg, zero_frame)

        payload = {
            "video_id": video_id,
            "video_folder": video_folder,
            "fps": fps,
            "total_frames": total_frames,
            "labels": labels,
            "frames": frames_tensor,
            "view_frames_num": cfg["view_frames_num"],
            "input_size": cfg["input_size"],
            "starting_frame_number": cfg["starting_frame_number"],
            "cache_pixel_dtype": cfg["cache_pixel_dtype"],
            "cache_normalize": cfg["cache_normalize"],
            "cache_encoding": "int8_with_offset_128" if cfg["cache_pixel_dtype"] == "int8" else "native",
            "cache_structure": "frames",
        }
        # serialize and compress
        buf = io.BytesIO()
        torch.save(payload, buf)
        compressed = lz4.frame.compress(buf.getvalue())
        with open(cache_path, "wb") as f:
            f.write(compressed)
        return 1, 0

    # submit all samples to thread pool
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_sample, dataset[i]) for i in range(len(dataset))]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Caching {split_name}"):
            gen, skip = fut.result()
            generated += gen
            skipped += skip

    return generated, skipped


def main():
    parser = argparse.ArgumentParser(description="Generate per-video dataset cache for MEM-TAD")
    parser.add_argument("--cfg", type=str, required=True, help="Path to config yaml")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="",
        help="Override cache root directory (default: cfg.dataset_cache_dir)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cache files")
    parser.add_argument(
        "--pixel_dtype",
        type=str,
        default="",
        choices=["int8", "uint8", "float32"],
        help="Override cache pixel dtype (default from cfg.cache_pixel_dtype)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize to [0,1] only when --pixel_dtype float32",
    )
    args = parser.parse_args()

    cfg = normalize_cfg(load_config(args.cfg))

    if args.pixel_dtype:
        cfg["cache_pixel_dtype"] = args.pixel_dtype
    if args.normalize:
        cfg["cache_normalize"] = True

    if cfg["cache_pixel_dtype"] != "float32" and cfg["cache_normalize"]:
        print("Warning: cache_normalize only applies to float32 cache; ignored for int8/uint8.")

    if not cfg["frames_dir"] or not cfg["annotations_json_path"]:
        raise ValueError("Config must include frames_dir and annotations_json_path")

    # Cache generation itself runs on CPU to avoid occupying GPU memory.
    train_dataset, val_dataset = load_train_val_data(
        frames_dir=cfg["frames_dir"],
        json_path=cfg["annotations_json_path"],
        device="cpu",
    )

    cache_base = Path(args.cache_dir) if args.cache_dir else Path(cfg["dataset_cache_dir"])
    # view_frames_num no longer part of version tag since cache stores full-frame tensors
    # a single cache can serve multiple clip lengths at training time.
    version_tag = (
        f"lz4_h{cfg['input_size'][0]}_w{cfg['input_size'][1]}_"
        f"sf{cfg['starting_frame_number']}_"
        f"dtype{cfg['cache_pixel_dtype']}_"
        f"norm{int(cfg['cache_normalize'])}_"
        f"{Path(cfg['annotations_json_path']).stem.split('_')[-1]}"  # e.g. "low" or "high"
    )
    cache_root = cache_base / version_tag
    cache_root.mkdir(parents=True, exist_ok=True)

    zero_frame = build_zero_frame(cfg)

    print(f"Cache root: {cache_root}")
    print(f"Train videos: {len(train_dataset)}, Val videos: {len(val_dataset)}")

    train_gen, train_skip = cache_one_split(
        train_dataset,
        "train",
        cache_root,
        cfg,
        zero_frame,
        overwrite=args.overwrite,
    )
    val_gen, val_skip = cache_one_split(
        val_dataset,
        "val",
        cache_root,
        cfg,
        zero_frame,
        overwrite=args.overwrite,
    )

    metadata = {
        "frames_dir": cfg["frames_dir"],
        "annotations_json_path": cfg["annotations_json_path"],
        "view_frames_num": cfg["view_frames_num"],
        "input_size": cfg["input_size"],
        "starting_frame_number": cfg["starting_frame_number"],
        "cache_pixel_dtype": cfg["cache_pixel_dtype"],
        "cache_normalize": cfg["cache_normalize"],
        "cache_encoding": "int8_with_offset_128" if cfg["cache_pixel_dtype"] == "int8" else "native",
        "cache_structure": "frames",
        "cache_format": "lz4",
        "class_num": train_dataset.class_num,
        "train_videos": len(train_dataset),
        "val_videos": len(val_dataset),
    }
    with open(cache_root / "metadata.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)

    print(
        "Done. "
        f"train generated/skipped: {train_gen}/{train_skip}, "
        f"val generated/skipped: {val_gen}/{val_skip}"
    )
if __name__ == "__main__":
	main()
# Example:
# python -m tools.generate_dataset_cache_lz4 --cfg configs/tennisnet/slowfast_middle.yml --cache_dir ./cache --pixel_dtype uint8
