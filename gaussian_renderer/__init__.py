#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from scene.gaussian_model import GaussianModel
from diff_gaussian_rasterization_fastgs import GaussianRasterizationSettings, GaussianRasterizer

def render_fastgs(
    viewpoint_camera,
    pc : GaussianModel,
    bg_color : torch.Tensor,
    mult,
    scaling_modifier = 1.0,
    override_color = None,
    get_flag=None,
    metric_map = None,
    mesh_rgb = None,
    mesh_depth = None,
    hybrid_beta = 0.0,
    render_mode = "hybrid",
):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points = torch.zeros((pc.get_xyz.shape[0], 4), dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if metric_map==None:
        metric_map=torch.zeros(int(viewpoint_camera.image_height)*int(viewpoint_camera.image_width), dtype=torch.int, device='cuda')
    use_mesh = render_mode in ("hybrid", "mesh") and mesh_rgb is not None and mesh_depth is not None
    if use_mesh:
        mesh_rgb = mesh_rgb.contiguous()
        mesh_depth = mesh_depth.contiguous()
    else:
        mesh_rgb = torch.empty(0, dtype=torch.float32, device=bg_color.device)
        mesh_depth = torch.empty(0, dtype=torch.float32, device=bg_color.device)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        mult = mult,
        prefiltered=False,
        debug=False,
        get_flag=get_flag,
        metric_map = metric_map,
        mesh_depth = mesh_depth,
        mesh_color = mesh_rgb,
        hybrid_beta = float(hybrid_beta) if render_mode == "hybrid" else 0.0,
        use_mesh = bool(use_mesh)
    )

    if render_mode == "mesh":
        return {"render": mesh_rgb.permute(2, 0, 1).contiguous(),
                "viewspace_points": screenspace_points,
                "visibility_filter" : torch.empty((0, 1), dtype=torch.long, device=bg_color.device),
                "radii": torch.zeros(pc.get_xyz.shape[0], dtype=torch.int32, device=bg_color.device),
                "accum_metric_counts" : torch.empty(0, dtype=torch.int32, device=bg_color.device),
                "participation_count": torch.zeros(pc.get_xyz.shape[0], dtype=torch.int32, device=bg_color.device)}

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    scales = pc.get_scaling
    rotations = pc.get_rotation
    cov3D_precomp = None

    if override_color is None:
        dc, shs = pc.get_features_dc, pc.get_features_rest
        colors_precomp = None
    else:
        dc, shs = None, None
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, accum_metric_counts, participation_count = rasterizer(
        means3D = means3D,
        means2D = means2D,
        dc = dc,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    visibility_filter = (participation_count > 0).nonzero() if use_mesh else (radii > 0).nonzero()
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : visibility_filter,
            "radii": radii,
            "accum_metric_counts" : accum_metric_counts,
            "participation_count": participation_count}
