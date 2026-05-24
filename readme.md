Minimal FastGS hybrid pipeline.

1. Precompute mesh support:

```bash
python precompute_mesh_support.py \
  -s /path/to/colmap_scene \
  -i images \
  --mesh /path/to/mesh.obj \
  --preview
```

This writes `/path/to/colmap_scene/mesh_support/*.npz` with `rgb`, `depth`, and `mask`.

2. Train:

```bash
python train.py \
  -s /path/to/colmap_scene \
  -i images \
  -m output/scene_hybrid \
  --mesh_support_dir /path/to/colmap_scene/mesh_support \
  --densification_interval 500 \
  --save_iterations 30000
```
