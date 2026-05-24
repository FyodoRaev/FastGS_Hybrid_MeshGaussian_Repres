import os
import sys
import uuid
from argparse import ArgumentParser, Namespace
from random import randint

import torch
from fused_ssim import fused_ssim as fast_ssim
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams
from gaussian_renderer import render_fastgs
from hybrid_math import annealed_beta
from mesh_support_cache import MeshSupportCache
from scene import GaussianModel, Scene
from utils.fast_utils import compute_gaussian_score_fastgs, sampling_cameras
from utils.loss_utils import l1_loss


def mesh(cache, cam):
    return cache.load_tensors(cam.image_name, "cuda", cam.image_width, cam.image_height)[:2]


def render(cam, gs, bg, opt, cache=None, beta=0.0, mode=None, **kw):
    rgb, depth = mesh(cache, cam)
    return render_fastgs(cam, gs, bg, opt.mult, mesh_rgb=rgb, mesh_depth=depth, hybrid_beta=beta, render_mode=mode or "hybrid", **kw)


def safe_name(name):
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


def label(im, text):
    draw = ImageDraw.Draw(im)
    size = max(14, im.width // 70)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        font = ImageFont.load_default()
    pad = max(4, size // 3)
    box = draw.textbbox((0, 0), text, font=font)
    draw.rectangle((0, 0, box[2] + 2 * pad, box[3] + 2 * pad), fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
    return im


def tensor_image(x):
    x = x.detach().clamp(0, 1)
    if x.ndim == 3 and x.shape[0] in (1, 3):
        x = x.permute(1, 2, 0)
    return Image.fromarray((x.contiguous().cpu().numpy() * 255).round().astype("uint8"))


def save_image(path, x, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    label(tensor_image(x), text).save(path)


def error_heatmap(rendered, gt, max_error):
    err = (rendered.detach() - gt.detach()).abs().mean(0)
    x = (err / max_error).clamp(0, 1)
    lut = x.new_tensor([
        [0.02, 0.02, 0.06],
        [0.05, 0.18, 0.55],
        [0.00, 0.58, 0.78],
        [0.20, 0.75, 0.32],
        [0.95, 0.80, 0.18],
        [0.88, 0.10, 0.06],
    ])
    y = x * (len(lut) - 1)
    i = y.floor().long().clamp(max=len(lut) - 2)
    t = (y - i.float()).unsqueeze(-1)
    return (lut[i] * (1 - t) + lut[i + 1] * t).permute(2, 0, 1)


def save_visualizations(root, iteration, cams, gs, bg, opt, cache, beta, max_error, num_of_gs):
    os.makedirs(root, exist_ok=True)
    with torch.no_grad():
        for cam in cams:
            stem = f"{iteration:06d}_{safe_name(cam.image_name)}"
            gt = cam.original_image.cuda()
            hybrid = render(cam, gs, bg, opt, cache, beta, "hybrid")["render"]
            gs_only = render_fastgs(cam, gs, bg, opt.mult, render_mode="gs")["render"]
            mesh_only = render(cam, gs, bg, opt, cache, beta, "mesh")["render"]
            heatmap = error_heatmap(hybrid, gt, max_error)
            common = f"iter {iteration}"
            save_image(os.path.join(root, f"{stem}_gt.png"), gt, f"{common}, gt")
            save_image(os.path.join(root, f"{stem}_hybrid.png"), hybrid, f"{common}, num of gaussians{num_of_gs} | hybrid")
            save_image(os.path.join(root, f"{stem}_gs_only.png"), gs_only, f"{common}, num of gaussians{num_of_gs} | gs only")
            save_image(os.path.join(root, f"{stem}_mesh_only.png"), mesh_only, f"{common} | mesh only")
            save_image(os.path.join(root, f"{stem}_heatmap.png"), heatmap, f"{common} | mean abs error 0..{max_error:g}")


def prepare(args):
    if not args.model_path: args.model_path = os.path.join("./output", os.getenv("OAR_JOB_ID") or str(uuid.uuid4()))
    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w", encoding="utf-8") as f:
        f.write(str(Namespace(**vars(args))))


def train(dataset, opt, args):
    prepare(dataset)
    gs = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gs)
    gs.training_setup(opt)
    first = 0
    cams = scene.getTrainCameras().copy()
    if not cams: raise RuntimeError("No training cameras loaded")

    cache = MeshSupportCache(dataset.source_path, opt.mesh_support_dir or None)
    if not cache.available: raise RuntimeError(f"No mesh support npz files found in {cache.support_dir}")

    stack, ema = cams.copy(), 0.0
    bar = tqdm(range(first, opt.iterations), desc="Training progress")
    for it in range(first + 1, opt.iterations + 1):
        gs.update_learning_rate(it)
        if it % 1000 == 0: gs.oneupSHdegree()
        if not stack: stack = cams.copy()

        cam = stack.pop(randint(0, len(stack) - 1))
        background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
        beta = annealed_beta(it, opt.hybrid_beta_min, opt.hybrid_beta_max, opt.hybrid_beta_warmup_iters)
        pkg = render(cam, gs, background, opt, cache, beta)
        image, gt = pkg["render"], cam.original_image.cuda()
        loss = (1 - opt.lambda_dssim) * l1_loss(image, gt) + opt.lambda_dssim * (1 - fast_ssim(image[None], gt[None]))
        loss.backward()

        with torch.no_grad():
            ema = 0.4 * loss.item() + 0.6 * ema
            if it % 10 == 0:
                bar.set_postfix({"Loss": f"{ema:.7f}"}); bar.update(10)
            if it == opt.iterations: bar.close()
            if it in args.save_iterations:
                print(f"\n[ITER {it}] Saving Gaussians"); scene.save(it)
            if args.vis_every > 0 and (it % args.vis_every == 0 or it == opt.iterations):
                pool = [c for c in cams if c is not cam]
                viewset = [cam]
                for _ in range(min(args.vis_count - 1, len(pool))):
                    viewset.append(pool.pop(randint(0, len(pool) - 1)))
                save_visualizations(
                    os.path.join(scene.model_path, "visualizations", f"iteration_{it}"),
                    it, viewset, gs, background, opt, cache, beta, args.heatmap_max_error, num_of_gs=gs.get_xyz.shape[0],)

            radii, visible = pkg["radii"], pkg["visibility_filter"]
            if it < opt.densify_until_iter:
                gs.max_radii2D[visible] = torch.max(gs.max_radii2D[visible], radii[visible])
                gs.add_densification_stats(pkg["viewspace_points"], visible)
                if it > opt.densify_from_iter and it % opt.densification_interval == 0:
                    imp, prune = compute_gaussian_score_fastgs(sampling_cameras(cams.copy()), gs, background, opt, DENSIFY=True, mesh_cache=cache, hybrid_beta=beta)
                    gs.densify_and_prune_fastgs(max_screen_size=20 if it > opt.opacity_reset_interval else None, min_opacity=0.005, extent=scene.cameras_extent, radii=radii, args=opt, importance_score=imp, pruning_score=prune)
                if it % opt.opacity_reset_interval == 0 or (dataset.white_background and it == opt.densify_from_iter): gs.reset_opacity()
            if it % 3000 == 0 and opt.densify_until_iter < it < opt.iterations:
                _, prune = compute_gaussian_score_fastgs(sampling_cameras(cams.copy()), gs, background, opt, mesh_cache=cache, hybrid_beta=beta)
                gs.final_prune_fastgs(min_opacity=0.1, pruning_score=prune)

            if it < opt.iterations:
                gs.optimizer_step(it)
    print(f"Gaussian number: {gs.get_xyz.shape[0]}")


if __name__ == "__main__":
    parser = ArgumentParser(description="FastGS hybrid training")
    lp, op = ModelParams(parser), OptimizationParams(parser)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--vis_every", type=int, default=1000)
    parser.add_argument("--vis_count", type=int, default=4)
    parser.add_argument("--heatmap_max_error", type=float, default=0.25)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    train(lp.extract(args), op.extract(args), args)
    print("Training complete.")
