#!/usr/bin/env python3
"""Assignment 3 demo backend: bounding volume hierarchies (AABB trees).

Three demos compare a brute-force algorithm against the AABB-tree-accelerated
version your ``src/`` functions implement, report any disagreements as a
WARNING, and print a timing table:

    python main.py --mode rays        # ray-mesh intersection (default)
    python main.py --mode distances   # nearest-neighbour point queries
    python main.py --mode pairs       # broad-phase mesh-mesh box overlaps

Add ``--headless`` to skip the interactive polyscope window (the computation and
timings still run). Without a display the window is skipped automatically.
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# `cgcommon` (shared helpers) normally sits next to this file. If several
# assignments are checked out together it lives two levels up instead.
if not os.path.isdir(os.path.join(HERE, "cgcommon")):
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

from cgcommon.objio import read_obj, triangulate

from backend.types import Ray, MeshTriangle, CloudPoint

from src.ray_intersect_triangle_mesh_brute_force import (
    ray_intersect_triangle_mesh_brute_force,
)
from src.AABBTree import AABBTree
from src.AABBTree_ray_intersect import AABBTree_ray_intersect
from src.nearest_neighbor_brute_force import nearest_neighbor_brute_force
from src.point_AABBTree_squared_distance import point_AABBTree_squared_distance
from src.box_box_intersect import box_box_intersect
from src.find_all_intersecting_pairs_using_AABBTrees import (
    find_all_intersecting_pairs_using_AABBTrees,
)


def _timing_table(rows):
    print("  | Method      | Time in seconds |")
    print("  |:------------|----------------:|")
    for name, secs in rows:
        print(f"  | {name:<11} | {secs:15.9f} |")
    print()


def load_mesh(path):
    V, Fraw = read_obj(path)
    F = triangulate(Fraw)
    return V, F


# --------------------------------------------------------------------------- #
# Mode: rays
# --------------------------------------------------------------------------- #

def run_rays(args):
    path = args.mesh or os.path.join(HERE, "data", "bob.obj")
    V, F = load_mesh(path)
    print("# Ray Triangle Mesh Intersection")
    print(f"  |V| {V.shape[0]}")
    print(f"  |F| {F.shape[0]}\n")

    rng = np.random.default_rng(args.seed)
    n = args.n
    origins = rng.uniform(-1.0, 1.0, size=(n, 3))
    dirs = rng.uniform(-1.0, 1.0, size=(n, 3))
    print(f"  Firing {n} rays...\n")

    min_t, max_t = 0.0, np.inf

    t0 = time.perf_counter()
    bf = [
        ray_intersect_triangle_mesh_brute_force(Ray(origins[r], dirs[r]), V, F, min_t, max_t)
        for r in range(n)
    ]
    t_bf = time.perf_counter() - t0

    t0 = time.perf_counter()
    root = AABBTree([MeshTriangle(V, F, f) for f in range(F.shape[0])])
    t_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    tree = []
    for r in range(n):
        hit, t, desc = AABBTree_ray_intersect(root, Ray(origins[r], dirs[r]), min_t, max_t)
        tree.append((hit, t, desc.f if (hit and desc is not None) else -1))
    t_tree = time.perf_counter() - t0

    # Agreement check.
    mismatches = 0
    hit_pts = []
    for r in range(n):
        bh, bt, bff = bf[r]
        th, tt, tf = tree[r]
        if bool(bh) != bool(th):
            mismatches += 1
        elif bh:
            if bff != tf or not np.isclose(bt, tt):
                mismatches += 1
            else:
                hit_pts.append(origins[r] + tt * dirs[r])
    if mismatches:
        print(f"  WARNING: brute force and tree disagree on {mismatches}/{n} rays")
    n_hit = sum(1 for bh, _, _ in bf if bh)
    print(f"  {n_hit}/{n} rays hit the mesh; brute force and tree agree on "
          f"{n - mismatches}/{n} rays.\n")

    _timing_table([("brute force", t_bf), ("build tree", t_build), ("use tree", t_tree)])

    if not args.headless:
        _visualize_rays(V, F, np.array(hit_pts) if hit_pts else np.zeros((0, 3)), root)


def _register_aabb_tree(ps, root, max_depth=5):
    """Draw the AABB tree's boxes as one curve network named "aabb tree".

    Draws the 12 edges of every tree
    node's box, down to ``max_depth`` levels below the root.
    """
    verts = []
    edges = []
    # Corner k has bits (bx, by, bz) with k = bx + 2*by + 4*bz; edges pair the
    # corners differing in exactly one bit.
    box_edges = [
        (0, 1), (2, 3), (4, 5), (6, 7),   # x-aligned edges
        (0, 2), (1, 3), (4, 6), (5, 7),   # y-aligned edges
        (0, 4), (1, 5), (2, 6), (3, 7),   # z-aligned edges
    ]

    def add_box(box):
        mn, mx = box.min_corner, box.max_corner
        if not (np.all(np.isfinite(mn)) and np.all(np.isfinite(mx))):
            return
        base = len(verts)
        for z in (mn[2], mx[2]):
            for y in (mn[1], mx[1]):
                for x in (mn[0], mx[0]):
                    verts.append((x, y, z))
        edges.extend((base + a, base + b) for a, b in box_edges)

    def walk(node, depth):
        if not isinstance(node, AABBTree) or depth > max_depth:
            return
        add_box(node.box)
        walk(node.left, depth + 1)
        walk(node.right, depth + 1)

    walk(root, 0)
    if verts:
        ps.register_curve_network(
            "aabb tree",
            np.asarray(verts, dtype=float),
            np.asarray(edges, dtype=int),
            radius=0.0015,
        )


_MAX_FRAMES = None  # debug: when set, tick N frames instead of blocking


def _show(viewer):
    if _MAX_FRAMES is None:
        viewer.show()
    else:
        import polyscope as ps
        for _ in range(_MAX_FRAMES):
            ps.frame_tick()


def _visualize_rays(V, F, hit_pts, root=None):
    try:
        from cgcommon import viewer
        viewer.ensure_init()
        import polyscope as ps
        ps.register_surface_mesh("mesh", V, F)
        if len(hit_pts):
            ps.register_point_cloud("ray hits", hit_pts)
        if root is not None:
            _register_aabb_tree(ps, root)
        _show(viewer)
    except Exception as exc:  # noqa: BLE001
        print(f"  (visualization unavailable: {exc})")


# --------------------------------------------------------------------------- #
# Mode: distances
# --------------------------------------------------------------------------- #

def run_distances(args):
    print("# Point Cloud Distance Queries")
    rng = np.random.default_rng(args.seed)
    n_pts = args.points
    n_q = args.queries
    points = rng.uniform(-1.0, 1.0, size=(n_pts, 3))
    queries = rng.uniform(-1.0, 1.0, size=(n_q, 3))
    print(f"    |points|: {n_pts}")
    print(f"  |queries|: {n_q}\n")

    t0 = time.perf_counter()
    bf = [nearest_neighbor_brute_force(points, queries[i]) for i in range(n_q)]
    t_bf = time.perf_counter() - t0

    t0 = time.perf_counter()
    root = AABBTree([CloudPoint(points, i) for i in range(n_pts)])
    t_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    tree = []
    for i in range(n_q):
        found, d, desc = point_AABBTree_squared_distance(queries[i], root, 0.0, np.inf)
        tree.append((found, d, desc.i if (found and desc is not None) else -1))
    t_tree = time.perf_counter() - t0

    mismatches = 0
    for i in range(n_q):
        (bI, bd) = bf[i]
        (found, td, tI) = tree[i]
        if not found or bI != tI or not np.isclose(bd, td):
            mismatches += 1
    if mismatches:
        print(f"  WARNING: brute force and tree disagree on {mismatches}/{n_q} queries")
    print(f"  brute force and tree agree on {n_q - mismatches}/{n_q} "
          "nearest-neighbour queries.\n")

    _timing_table([("brute force", t_bf), ("build tree", t_build), ("use tree", t_tree)])

    # Amortization: the tree only pays off once its per-query savings cover the
    # up-front build cost.
    brute_per_query = t_bf / n_q
    tree_per_query = t_tree / n_q
    tiny = 1e-12
    if tree_per_query < brute_per_query:
        queries_to_amortize = t_build / max(brute_per_query - tree_per_query, tiny)
        print(f"  tree build amortizes after ~{int(np.ceil(queries_to_amortize))} queries.\n")
    else:
        print("  tree build never amortizes at this size "
              "(tree per-query is not faster than brute force).\n")

    if not args.headless:
        _visualize_points(points, queries, root)


def _visualize_points(points, queries, root=None):
    try:
        from cgcommon import viewer
        viewer.ensure_init()
        import polyscope as ps
        ps.register_point_cloud("points", points)
        ps.register_point_cloud("queries", queries)
        if root is not None:
            _register_aabb_tree(ps, root)
        _show(viewer)
    except Exception as exc:  # noqa: BLE001
        print(f"  (visualization unavailable: {exc})")


# --------------------------------------------------------------------------- #
# Mode: pairs
# --------------------------------------------------------------------------- #

def run_pairs(args):
    print("# Triangle Mesh Intersection Detection (broad phase)")
    pathA = args.mesh or os.path.join(HERE, "data", "bob.obj")
    VA, FA = load_mesh(pathA)
    if args.mesh_b:
        VB, FB = load_mesh(args.mesh_b)
    else:
        # Overlap a second copy of the same mesh by a small translation.
        VB = VA.copy() + np.array([0.35, 0.15, 0.1])
        FB = FA.copy()
    print(f"  |FA| {FA.shape[0]}    |FB| {FB.shape[0]}\n")

    trisA = [MeshTriangle(VA, FA, f) for f in range(FA.shape[0])]
    trisB = [MeshTriangle(VB, FB, f) for f in range(FB.shape[0])]

    # Brute-force box-box pairs.
    t0 = time.perf_counter()
    bf_pairs = set()
    for a in trisA:
        for b in trisB:
            if box_box_intersect(a.box, b.box):
                bf_pairs.add((a.f, b.f))
    t_bf = time.perf_counter() - t0

    t0 = time.perf_counter()
    rootA = AABBTree(trisA)
    rootB = AABBTree(trisB)
    t_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    leaf_pairs = find_all_intersecting_pairs_using_AABBTrees(rootA, rootB)
    tree_pairs = {(a.f, b.f) for a, b in leaf_pairs}
    t_tree = time.perf_counter() - t0

    n_mismatch = len(tree_pairs ^ bf_pairs)
    if n_mismatch:
        print(f"  WARNING: tree and brute force disagree on {n_mismatch} pairs "
              f"(tree found {len(tree_pairs)}, brute force {len(bf_pairs)})")
    print(f"  {len(tree_pairs)} overlapping leaf-box pairs; tree and brute force "
          f"agree on {len(tree_pairs & bf_pairs)}.\n")

    _timing_table([("brute force", t_bf), ("build trees", t_build), ("use trees", t_tree)])

    if not args.headless:
        _visualize_pairs(VA, FA, VB, FB, rootA)


def _visualize_pairs(VA, FA, VB, FB, root=None):
    try:
        from cgcommon import viewer
        viewer.ensure_init()
        import polyscope as ps
        ps.register_surface_mesh("mesh A", VA, FA)
        ps.register_surface_mesh("mesh B", VB, FB)
        if root is not None:
            _register_aabb_tree(ps, root)
        _show(viewer)
    except Exception as exc:  # noqa: BLE001
        print(f"  (visualization unavailable: {exc})")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["rays", "distances", "pairs"], default="rays")
    parser.add_argument("--headless", action="store_true", help="skip the interactive viewer")
    parser.add_argument("--mesh", default=None, help="mesh OBJ for rays/pairs (mesh A)")
    parser.add_argument("--mesh-b", dest="mesh_b", default=None, help="second mesh OBJ for pairs")
    parser.add_argument("--n", type=int, default=1000, help="number of rays (rays mode)")
    parser.add_argument("--points", type=int, default=100000, help="point-cloud size (distances)")
    parser.add_argument("--queries", type=int, default=2000, help="number of queries (distances)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--frames", type=int, default=None,
                        help=argparse.SUPPRESS)  # debug: tick N frames and exit
    args = parser.parse_args(argv)

    global _MAX_FRAMES
    _MAX_FRAMES = args.frames

    if args.mode == "rays":
        run_rays(args)
    elif args.mode == "distances":
        run_distances(args)
    else:
        run_pairs(args)


if __name__ == "__main__":
    main()
