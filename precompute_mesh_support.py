from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import tqdm

from arguments import GroupParams
from mesh_renderer import MeshRenderer
from scene.dataset_readers import readColmapSceneInfo
from utils.camera_utils import cameraList_from_camInfos


def frame_stem(camera) -> str:
    return Path(camera.image_name).stem


def load_cameras(source_path: Path, images: str, resolution: int):
    info = readColmapSceneInfo(str(source_path), images)
    args = GroupParams()
    args.resolution = resolution
    args.data_device = "cpu"
    return cameraList_from_camInfos(info.train_cameras, 1.0, args)


def write_npz(path: Path, rgb: torch.Tensor, depth: torch.Tensor, mask: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb_np = (rgb.detach().cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    depth_np = depth.detach().cpu().numpy().astype(np.float32)
    mask_np = mask.detach().cpu().numpy().astype(bool)
    np.savez_compressed(path, rgb=rgb_np, depth=depth_np, mask=mask_np)


def save_rgb(path: Path, rgb: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (rgb.detach().cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    imageio.imwrite(path, arr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute mesh RGB/depth support for hybrid FastGS training")
    parser.add_argument("-s", "--source_path", required=True, help="COLMAP scene directory")
    parser.add_argument("--mesh", required=True, help="Textured OBJ mesh in the same coordinate system as COLMAP")
    parser.add_argument("-i", "--images", default="images", help="Image directory inside the scene")
    parser.add_argument("-r", "--resolution", type=int, default=-1, help="Same resolution rule as train.py")
    parser.add_argument("-o", "--output", default=None, help="Output directory, default: <source_path>/mesh_support")
    parser.add_argument("--preview", action="store_true", help="Also write PNG previews")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    source = Path(args.source_path)
    out_dir = Path(args.output) if args.output else source / "mesh_support"
    cameras = load_cameras(source, args.images, args.resolution)
    renderer = MeshRenderer(args.mesh, args.device)

    started = time.time()
    written = []
    for camera in tqdm.tqdm(cameras, desc="Mesh support"):
        rgb, depth, mask = renderer.render(camera)
        stem = frame_stem(camera)
        write_npz(out_dir / f"{stem}.npz", rgb, depth, mask)
        if args.preview:
            save_rgb(out_dir / "preview" / f"{stem}.png", rgb)
        written.append(stem)

    manifest_path = out_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({
        "source_path": str(source),
        "mesh": str(Path(args.mesh)),
        "images": args.images,
        "resolution": args.resolution,
        "frames": len(written),
        "seconds": round(time.time() - started, 3),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
