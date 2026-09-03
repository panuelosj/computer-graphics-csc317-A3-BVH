# Public test data: A3-BVH

These are the exact arrays the public unit checks compare your
functions against — the inputs they feed in and the outputs they
expect back. Everything here is generated from
`a3_golden.npz`; nothing here is secret, and you
are meant to look at it when a check fails.

* `arrays/<name>.txt` — the values, in readable form.
* `images/<name>.png` — the same array as a viewable image, for
  the entries that are pictures.

To load an array yourself:

```python
import numpy as np
g = np.load('tests/golden/a3_golden.npz')
print(g.files)          # every available name
img = g['<name>']       # one array
```

Regenerate this folder with `python export_expected.py <assignment>` from the repository root.

| entry | what it is | values | picture |
|---|---|---|---|
| `bbi_A_max` | shape (4, 3), dtype float64, range [1, 1] | [bbi_A_max.txt](arrays/bbi_A_max.txt) | — |
| `bbi_A_min` | shape (4, 3), dtype float64, range [0, 0] | [bbi_A_min.txt](arrays/bbi_A_min.txt) | — |
| `bbi_B_max` | shape (4, 3), dtype float64, range [1, 3] | [bbi_B_max.txt](arrays/bbi_B_max.txt) | — |
| `bbi_B_min` | shape (4, 3), dtype float64, range [0, 2] | [bbi_B_min.txt](arrays/bbi_B_min.txt) | — |
| `bbi_expected` | shape (4,), dtype bool | [bbi_expected.txt](arrays/bbi_expected.txt) | — |
| `ibb_A_max` | shape (3,), dtype float64, range [1, 1] | [ibb_A_max.txt](arrays/ibb_A_max.txt) | — |
| `ibb_A_min` | shape (3,), dtype float64, range [0, 0] | [ibb_A_min.txt](arrays/ibb_A_min.txt) | — |
| `ibb_B_max` | shape (3,), dtype float64, range [0.5, 2] | [ibb_B_max.txt](arrays/ibb_B_max.txt) | — |
| `ibb_B_min` | shape (3,), dtype float64, range [-1, 0.5] | [ibb_B_min.txt](arrays/ibb_B_min.txt) | — |
| `ibb_empty_out_max` | shape (3,), dtype float64, range [1, 1] | [ibb_empty_out_max.txt](arrays/ibb_empty_out_max.txt) | — |
| `ibb_empty_out_min` | shape (3,), dtype float64, range [0, 0] | [ibb_empty_out_min.txt](arrays/ibb_empty_out_min.txt) | — |
| `ibb_out_max` | shape (3,), dtype float64, range [1, 2] | [ibb_out_max.txt](arrays/ibb_out_max.txt) | — |
| `ibb_out_min` | shape (3,), dtype float64, range [-1, 0] | [ibb_out_min.txt](arrays/ibb_out_min.txt) | — |
| `itb_a` | shape (3,), dtype float64, range [1, 3] | [itb_a.txt](arrays/itb_a.txt) | — |
| `itb_b` | shape (3,), dtype float64, range [-1, 4] | [itb_b.txt](arrays/itb_b.txt) | — |
| `itb_c` | shape (3,), dtype float64, range [-2, 1] | [itb_c.txt](arrays/itb_c.txt) | — |
| `itb_empty_max` | shape (3,), dtype float64, range [1, 4] | [itb_empty_max.txt](arrays/itb_empty_max.txt) | — |
| `itb_empty_min` | shape (3,), dtype float64, range [-2, 1] | [itb_empty_min.txt](arrays/itb_empty_min.txt) | — |
| `itb_out_max` | shape (3,), dtype float64, range [1, 5] | [itb_out_max.txt](arrays/itb_out_max.txt) | — |
| `itb_out_min` | shape (3,), dtype float64, range [-2, 0] | [itb_out_min.txt](arrays/itb_out_min.txt) | — |
| `itb_pre_max` | shape (3,), dtype float64, range [0.5, 5] | [itb_pre_max.txt](arrays/itb_pre_max.txt) | — |
| `itb_pre_min` | shape (3,), dtype float64, range [0, 0] | [itb_pre_min.txt](arrays/itb_pre_min.txt) | — |
| `mesh_F` | shape (668, 3), dtype int64, range [0, 333] | [mesh_F.txt](arrays/mesh_F.txt) | — |
| `mesh_V` | shape (334, 3), dtype float64, range [-0.960488, 0.960488] | [mesh_V.txt](arrays/mesh_V.txt) | — |
| `nn_I` | shape (200,), dtype int64, range [2, 1987] | [nn_I.txt](arrays/nn_I.txt) | — |
| `nn_sqrD` | shape (200,), dtype float64, range [0.000474865, 0.0297889] | [nn_sqrD.txt](arrays/nn_sqrD.txt) | — |
| `pair_FA` | shape (48, 3), dtype int64, range [0, 23] | [pair_FA.txt](arrays/pair_FA.txt) | — |
| `pair_FB` | shape (48, 3), dtype int64, range [0, 23] | [pair_FB.txt](arrays/pair_FB.txt) | — |
| `pair_VA` | shape (24, 3), dtype float64, range [-4, 4] | [pair_VA.txt](arrays/pair_VA.txt) | — |
| `pair_VB` | shape (24, 3), dtype float64, range [-3.65, 4.35] | [pair_VB.txt](arrays/pair_VB.txt) | — |
| `pair_expected` | shape (539, 2), dtype int64, range [0, 47] | [pair_expected.txt](arrays/pair_expected.txt) | — |
| `pbd_box_max` | shape (3,), dtype float64, range [1, 1] | [pbd_box_max.txt](arrays/pbd_box_max.txt) | — |
| `pbd_box_min` | shape (3,), dtype float64, range [-1, -1] | [pbd_box_min.txt](arrays/pbd_box_min.txt) | — |
| `pbd_expected` | shape (5,), dtype float64, range [0, 4] | [pbd_expected.txt](arrays/pbd_expected.txt) | — |
| `pbd_queries` | shape (5, 3), dtype float64, range [-3, 2] | [pbd_queries.txt](arrays/pbd_queries.txt) | — |
| `points` | _(txt truncated)_ | [points.txt](arrays/points.txt) | — |
| `queries` | shape (200, 3), dtype float64, range [-0.998057, 0.999774] | [queries.txt](arrays/queries.txt) | — |
| `ray_bf_f` | shape (500,), dtype int64, range [-1, 657] | [ray_bf_f.txt](arrays/ray_bf_f.txt) | — |
| `ray_bf_hit` | shape (500,), dtype bool | [ray_bf_hit.txt](arrays/ray_bf_hit.txt) | — |
| `ray_bf_t` | shape (500,), dtype float64, range [0.00327921, 3.78465] | [ray_bf_t.txt](arrays/ray_bf_t.txt) | — |
| `ray_d` | shape (500, 3), dtype float64, range [-0.999847, 0.999263] | [ray_d.txt](arrays/ray_d.txt) | — |
| `ray_o` | shape (500, 3), dtype float64, range [-0.999872, 0.996127] | [ray_o.txt](arrays/ray_o.txt) | — |
| `rib_box_max` | shape (3,), dtype float64, range [1, 1] | [rib_box_max.txt](arrays/rib_box_max.txt) | — |
| `rib_box_min` | shape (3,), dtype float64, range [-1, -1] | [rib_box_min.txt](arrays/rib_box_min.txt) | — |
| `rib_dirs` | shape (5, 3), dtype float64, range [-1, 1] | [rib_dirs.txt](arrays/rib_dirs.txt) | — |
| `rib_expected` | shape (5,), dtype bool | [rib_expected.txt](arrays/rib_expected.txt) | — |
| `rib_origins` | shape (5, 3), dtype float64, range [-5, 5] | [rib_origins.txt](arrays/rib_origins.txt) | — |
| `rit_A` | shape (3,), dtype float64, range [0, 0] | [rit_A.txt](arrays/rit_A.txt) | — |
| `rit_B` | shape (3,), dtype float64, range [0, 1] | [rit_B.txt](arrays/rit_B.txt) | — |
| `rit_C` | shape (3,), dtype float64, range [0, 1] | [rit_C.txt](arrays/rit_C.txt) | — |
| `rit_dirs` | shape (4, 3), dtype float64, range [0, 1] | [rit_dirs.txt](arrays/rit_dirs.txt) | — |
| `rit_hit` | shape (4,), dtype bool | [rit_hit.txt](arrays/rit_hit.txt) | — |
| `rit_origins` | shape (4, 3), dtype float64, range [-2, 2] | [rit_origins.txt](arrays/rit_origins.txt) | — |
| `rit_t` | shape (4,), dtype float64, range [-1, 2] | [rit_t.txt](arrays/rit_t.txt) | — |
| `tree_num_leaves` | shape (), dtype int64, range [668, 668] | [tree_num_leaves.txt](arrays/tree_num_leaves.txt) | — |
| `tree_root_max` | shape (3,), dtype float64, range [0.521779, 0.960488] | [tree_root_max.txt](arrays/tree_root_max.txt) | — |
| `tree_root_min` | shape (3,), dtype float64, range [-0.960488, -0.521779] | [tree_root_min.txt](arrays/tree_root_min.txt) | — |
