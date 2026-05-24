from __future__ import annotations

from pathlib import Path

import nvdiffrast.torch as dr
from pytorch3d.io import load_obj
import torch


class MeshRenderer:
    def __init__(self, mesh_path: str | Path, device: str = "cuda"):
        self.device = torch.device(device)
        mesh_path = Path(mesh_path)
        verts, faces, aux = load_obj(str(mesh_path), load_textures=True, device=self.device)
        self.verts = verts.float()
        self.faces = faces.verts_idx.int().contiguous()
        self.ctx = dr.RasterizeCudaContext(device=self.device)

        self.uvs = None
        self.face_uvs = None
        self.texture = None
        if aux.verts_uvs is not None and faces.textures_idx is not None and aux.texture_images:
            self.uvs = aux.verts_uvs.to(self.device).float()
            self.face_uvs = faces.textures_idx.to(self.device).int().contiguous()
            tex = next(iter(aux.texture_images.values())).to(self.device).float()
            self.texture = tex.clamp(0, 1)

    def render(self, camera):
        height, width = int(camera.image_height), int(camera.image_width)
        ones = torch.ones((self.verts.shape[0], 1), dtype=self.verts.dtype, device=self.device)
        verts_h = torch.cat([self.verts, ones], dim=1)
        clip = verts_h @ camera.full_proj_transform.to(self.device)
        view = verts_h @ camera.world_view_transform.to(self.device)

        rast, _ = dr.rasterize(self.ctx, clip[None], self.faces, resolution=(height, width))
        mask = rast[0, :, :, 3] > 0
        depth, _ = dr.interpolate(view[None, :, 2:3].contiguous(), rast, self.faces)
        depth = depth[0, :, :, 0].contiguous()
        depth = torch.where(mask, depth, torch.full_like(depth, float("inf")))

        if self.texture is not None:
            uv, _ = dr.interpolate(self.uvs[None], rast, self.face_uvs)
            rgb = dr.texture(self.texture[None], uv, filter_mode="linear")[0, :, :, :3]
        else:
            rgb = torch.full((height, width, 3), 0.5, dtype=torch.float32, device=self.device)
        rgb = torch.where(mask[..., None], rgb, torch.zeros_like(rgb)).contiguous()

        return rgb.clamp(0, 1), depth, mask
