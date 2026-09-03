"""Unit checks for Assignment 3 (BVH / AABB Tree).

Each check exercises one function in ``src/`` against the golden fixtures in
``tests/golden/a3_golden.npz``. The same checks are used by ``run_tests.py`` (for
students) and by the grader. Marks per check mirror the marking scheme.

The leaf-level functions are checked against baked reference outputs. The
hierarchical functions (tree build / traversal / nearest-neighbour / pair
finding) build a structure with the student's ``AABBTree`` and assert the result
*agrees with the reference brute-force answer* baked into the fixture -- robust
to reasonable implementation variation.
"""

import os

import numpy as np

from cgcommon.testkit import Check, fixture_path
from backend.types import BoundingBox, Ray, MeshTriangle, CloudPoint

_GOLDEN = None


def golden():
    global _GOLDEN
    if _GOLDEN is None:
        path = fixture_path(
            os.path.join(os.path.dirname(__file__), "golden", "a3_golden.npz")
        )
        _GOLDEN = np.load(path)
    return _GOLDEN


def _box(mn, mx):
    return BoundingBox(np.asarray(mn, float).copy(), np.asarray(mx, float).copy())


# --- leaf-level checks ------------------------------------------------------

def check_insert_box_into_box():
    from src.insert_box_into_box import insert_box_into_box
    g = golden()
    B = _box(g["ibb_B_min"], g["ibb_B_max"])
    insert_box_into_box(_box(g["ibb_A_min"], g["ibb_A_max"]), B)
    assert np.allclose(B.min_corner, g["ibb_out_min"]), "insert_box_into_box: wrong min_corner"
    assert np.allclose(B.max_corner, g["ibb_out_max"]), "insert_box_into_box: wrong max_corner"
    # growing an empty box yields exactly the inserted box
    Be = BoundingBox()
    insert_box_into_box(_box(g["ibb_A_min"], g["ibb_A_max"]), Be)
    assert np.allclose(Be.min_corner, g["ibb_empty_out_min"]), "insert_box_into_box: empty-box min"
    assert np.allclose(Be.max_corner, g["ibb_empty_out_max"]), "insert_box_into_box: empty-box max"


def check_insert_triangle_into_box():
    from src.insert_triangle_into_box import insert_triangle_into_box
    g = golden()
    Be = BoundingBox()
    insert_triangle_into_box(g["itb_a"], g["itb_b"], g["itb_c"], Be)
    assert np.allclose(Be.min_corner, g["itb_empty_min"]), "insert_triangle_into_box: empty min"
    assert np.allclose(Be.max_corner, g["itb_empty_max"]), "insert_triangle_into_box: empty max"
    B = _box(g["itb_pre_min"], g["itb_pre_max"])
    insert_triangle_into_box(g["itb_a"], g["itb_b"], g["itb_c"], B)
    assert np.allclose(B.min_corner, g["itb_out_min"]), "insert_triangle_into_box: grown min"
    assert np.allclose(B.max_corner, g["itb_out_max"]), "insert_triangle_into_box: grown max"


def check_ray_intersect_box():
    from src.ray_intersect_box import ray_intersect_box
    g = golden()
    box = _box(g["rib_box_min"], g["rib_box_max"])
    for i in range(len(g["rib_origins"])):
        got = ray_intersect_box(Ray(g["rib_origins"][i], g["rib_dirs"][i]), box, 0.0, np.inf)
        assert bool(got) == bool(g["rib_expected"][i]), (
            f"ray_intersect_box: ray {i} expected {bool(g['rib_expected'][i])}, got {bool(got)}"
        )
    # Bounded-interval cases (hand-computed): unit box [0,1]^3, ray from
    # (-1, .5, .5) along +x enters the box at t=1 and leaves at t=2.
    unit = _box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    ray = Ray([-1.0, 0.5, 0.5], [1.0, 0.0, 0.0])
    assert not ray_intersect_box(ray, unit, 0.0, 0.5), (
        "ray_intersect_box: must MISS with max_t=0.5 (box entry is at t=1)"
    )
    assert ray_intersect_box(ray, unit, 0.0, 1.5), (
        "ray_intersect_box: must HIT with max_t=1.5 (box interval [1,2] overlaps [0,1.5])"
    )
    assert ray_intersect_box(ray, unit, 1.5, np.inf), (
        "ray_intersect_box: must HIT with min_t=1.5 (ray is inside the box until t=2)"
    )


def check_ray_intersect_triangle():
    from src.ray_intersect_triangle import ray_intersect_triangle
    g = golden()
    A, B, C = g["rit_A"], g["rit_B"], g["rit_C"]
    for i in range(len(g["rit_origins"])):
        hit, t = ray_intersect_triangle(
            Ray(g["rit_origins"][i], g["rit_dirs"][i]), A, B, C, 0.0, np.inf
        )
        assert bool(hit) == bool(g["rit_hit"][i]), (
            f"ray_intersect_triangle: ray {i} hit expected {bool(g['rit_hit'][i])}, got {bool(hit)}"
        )
        if g["rit_hit"][i]:
            assert np.isclose(t, g["rit_t"][i]), (
                f"ray_intersect_triangle: ray {i} t expected {g['rit_t'][i]}, got {t}"
            )
    # Bounded-interval case (hand-computed): triangle in the z=0 plane, ray from
    # (0.2, 0.2, -1) along +z hits it at t=1 (beta=gamma=0.2).
    Ah = np.array([0.0, 0.0, 0.0])
    Bh = np.array([1.0, 0.0, 0.0])
    Ch = np.array([0.0, 1.0, 0.0])
    ray = Ray([0.2, 0.2, -1.0], [0.0, 0.0, 1.0])
    hit, t = ray_intersect_triangle(ray, Ah, Bh, Ch, 0.0, np.inf)
    assert hit and np.isclose(t, 1.0), (
        "ray_intersect_triangle: hand case must HIT at t=1 with [0, inf)"
    )
    hit, _ = ray_intersect_triangle(ray, Ah, Bh, Ch, 0.0, 0.5)
    assert not hit, "ray_intersect_triangle: max_t=0.5 must exclude the hit at t=1"
    hit, _ = ray_intersect_triangle(ray, Ah, Bh, Ch, 1.5, np.inf)
    assert not hit, "ray_intersect_triangle: min_t=1.5 must exclude the hit at t=1"


def check_ray_intersect_triangle_mesh_brute_force():
    from src.ray_intersect_triangle_mesh_brute_force import (
        ray_intersect_triangle_mesh_brute_force,
    )
    g = golden()
    V, F = g["mesh_V"], g["mesh_F"]
    for r in range(len(g["ray_o"])):
        hit, t, f = ray_intersect_triangle_mesh_brute_force(
            Ray(g["ray_o"][r], g["ray_d"][r]), V, F, 0.0, np.inf
        )
        assert bool(hit) == bool(g["ray_bf_hit"][r]), (
            f"brute force: ray {r} hit expected {bool(g['ray_bf_hit'][r])}, got {bool(hit)}"
        )
        if g["ray_bf_hit"][r]:
            assert int(f) == int(g["ray_bf_f"][r]), (
                f"brute force: ray {r} face expected {int(g['ray_bf_f'][r])}, got {int(f)}"
            )
            assert np.isclose(t, g["ray_bf_t"][r]), (
                f"brute force: ray {r} t expected {g['ray_bf_t'][r]}, got {t}"
            )


def check_nearest_neighbor_brute_force():
    from src.nearest_neighbor_brute_force import nearest_neighbor_brute_force
    g = golden()
    points, queries = g["points"], g["queries"]
    for i in range(len(queries)):
        I, d = nearest_neighbor_brute_force(points, queries[i])
        assert int(I) == int(g["nn_I"][i]), (
            f"nearest_neighbor_brute_force: query {i} index expected {int(g['nn_I'][i])}, got {int(I)}"
        )
        assert np.isclose(d, g["nn_sqrD"][i]), (
            f"nearest_neighbor_brute_force: query {i} sqrD expected {g['nn_sqrD'][i]}, got {d}"
        )


def check_point_box_squared_distance():
    from src.point_box_squared_distance import point_box_squared_distance
    g = golden()
    box = _box(g["pbd_box_min"], g["pbd_box_max"])
    for i in range(len(g["pbd_queries"])):
        d = point_box_squared_distance(g["pbd_queries"][i], box)
        assert np.isclose(d, g["pbd_expected"][i]), (
            f"point_box_squared_distance: query {i} expected {g['pbd_expected'][i]}, got {d}"
        )


def check_box_box_intersect():
    from src.box_box_intersect import box_box_intersect
    g = golden()
    for i in range(len(g["bbi_A_min"])):
        got = box_box_intersect(
            _box(g["bbi_A_min"][i], g["bbi_A_max"][i]),
            _box(g["bbi_B_min"][i], g["bbi_B_max"][i]),
        )
        assert bool(got) == bool(g["bbi_expected"][i]), (
            f"box_box_intersect: pair {i} expected {bool(g['bbi_expected'][i])}, got {bool(got)}"
        )


# --- hierarchical checks ----------------------------------------------------

def _collect_leaves(node, AABBTree, out):
    """Walk a tree, appending every leaf object (non-AABBTree) to ``out``."""
    if node is None:
        return
    if isinstance(node, AABBTree):
        _collect_leaves(node.left, AABBTree, out)
        _collect_leaves(node.right, AABBTree, out)
    else:
        out.append(node)


def check_AABBTree():
    from src.AABBTree import AABBTree
    g = golden()
    V, F = g["mesh_V"], g["mesh_F"]
    m = F.shape[0]
    objs = [MeshTriangle(V, F, f) for f in range(m)]
    root = AABBTree(objs)
    # num_leaves and enclosing box.
    assert int(root.num_leaves) == m, (
        f"AABBTree: num_leaves expected {m}, got {root.num_leaves}"
    )
    assert np.allclose(root.box.min_corner, g["tree_root_min"]), "AABBTree: wrong enclosing min"
    assert np.allclose(root.box.max_corner, g["tree_root_max"]), "AABBTree: wrong enclosing max"
    # Every input triangle must appear exactly once among the leaves.
    leaves = []
    _collect_leaves(root, AABBTree, leaves)
    faces = sorted(leaf.f for leaf in leaves)
    assert faces == list(range(m)), (
        "AABBTree: leaves must contain each input object exactly once"
    )

    # Structural invariants: every internal node's num_leaves must equal the sum
    # over its children (a non-AABBTree child counts as 1 leaf), and the tree
    # must be reasonably balanced for this fixture.
    def _leaves_of(node):
        return int(node.num_leaves) if isinstance(node, AABBTree) else 1

    def _walk(node, depth):
        """Check the num_leaves invariant; return the max depth below ``node``."""
        if not isinstance(node, AABBTree):
            return depth
        children = [c for c in (node.left, node.right) if c is not None]
        assert children, "AABBTree: internal node with no children"
        total = sum(_leaves_of(c) for c in children)
        assert total == int(node.num_leaves), (
            f"AABBTree: node at depth {depth} has num_leaves={int(node.num_leaves)} "
            f"but its children account for {total}"
        )
        return max(_walk(c, depth + 1) for c in children)

    max_depth = _walk(root, 0)
    depth_cap = 4.0 * np.log2(m) + 8.0
    assert max_depth <= depth_cap, (
        f"AABBTree: tree depth {max_depth} exceeds {depth_cap:.1f} "
        f"(4*log2({m})+8) -- the midpoint split should stay roughly balanced"
    )


def check_AABBTree_ray_intersect():
    from src.AABBTree import AABBTree
    from src.AABBTree_ray_intersect import AABBTree_ray_intersect
    g = golden()
    V, F = g["mesh_V"], g["mesh_F"]
    root = AABBTree([MeshTriangle(V, F, f) for f in range(F.shape[0])])
    for r in range(len(g["ray_o"])):
        hit, t, desc = AABBTree_ray_intersect(
            root, Ray(g["ray_o"][r], g["ray_d"][r]), 0.0, np.inf
        )
        assert bool(hit) == bool(g["ray_bf_hit"][r]), (
            f"AABBTree_ray_intersect: ray {r} hit expected {bool(g['ray_bf_hit'][r])}, got {bool(hit)}"
        )
        if g["ray_bf_hit"][r]:
            assert desc is not None, f"AABBTree_ray_intersect: ray {r} hit but no descendant"
            assert int(desc.f) == int(g["ray_bf_f"][r]), (
                f"AABBTree_ray_intersect: ray {r} face expected {int(g['ray_bf_f'][r])}, got {int(desc.f)}"
            )
            assert np.isclose(t, g["ray_bf_t"][r]), (
                f"AABBTree_ray_intersect: ray {r} t expected {g['ray_bf_t'][r]}, got {t}"
            )


def check_point_AABBTree_squared_distance():
    from src.AABBTree import AABBTree
    from src.point_AABBTree_squared_distance import point_AABBTree_squared_distance
    g = golden()
    points, queries = g["points"], g["queries"]
    root = AABBTree([CloudPoint(points, i) for i in range(points.shape[0])])
    for i in range(len(queries)):
        found, d, desc = point_AABBTree_squared_distance(queries[i], root, 0.0, np.inf)
        assert found, f"point_AABBTree_squared_distance: query {i} found nothing"
        assert int(desc.i) == int(g["nn_I"][i]), (
            f"point_AABBTree_squared_distance: query {i} index expected {int(g['nn_I'][i])}, got {int(desc.i)}"
        )
        assert np.isclose(d, g["nn_sqrD"][i]), (
            f"point_AABBTree_squared_distance: query {i} sqrD expected {g['nn_sqrD'][i]}, got {d}"
        )
    # min_sqrd > 0 must skip nearer points: only distances in
    # [min_sqrd, max_sqrd) are accepted. The
    # expected answer is the second-nearest point, computed here by brute force.
    q = queries[0]
    d_all = np.sum((points - q) ** 2, axis=1)
    order = np.argsort(d_all)
    lo = 0.5 * (d_all[order[0]] + d_all[order[1]])  # between 1st- and 2nd-nearest
    found, d, desc = point_AABBTree_squared_distance(q, root, lo, np.inf)
    assert found, "point_AABBTree_squared_distance: min_sqrd>0 query found nothing"
    assert int(desc.i) == int(order[1]), (
        f"point_AABBTree_squared_distance: min_sqrd={lo} must skip the nearest point "
        f"{int(order[0])} and return {int(order[1])}, got {int(desc.i)}"
    )
    assert np.isclose(d, d_all[order[1]]), (
        f"point_AABBTree_squared_distance: min_sqrd>0 sqrD expected {d_all[order[1]]}, got {d}"
    )


def check_find_all_intersecting_pairs_using_AABBTrees():
    from src.AABBTree import AABBTree
    from src.find_all_intersecting_pairs_using_AABBTrees import (
        find_all_intersecting_pairs_using_AABBTrees,
    )
    g = golden()
    VA, FA, VB, FB = g["pair_VA"], g["pair_FA"], g["pair_VB"], g["pair_FB"]
    rootA = AABBTree([MeshTriangle(VA, FA, f) for f in range(FA.shape[0])])
    rootB = AABBTree([MeshTriangle(VB, FB, f) for f in range(FB.shape[0])])
    pairs = find_all_intersecting_pairs_using_AABBTrees(rootA, rootB)
    got = sorted((int(a.f), int(b.f)) for a, b in pairs)
    expected = sorted((int(a), int(b)) for a, b in g["pair_expected"])
    assert got == expected, (
        f"find_all_intersecting_pairs: expected {len(expected)} leaf-box pairs, got {len(got)} "
        f"(sets {'match' if set(got) == set(expected) else 'differ'})"
    )


def get_checks():
    """Return the ordered list of checks for this assignment."""
    return [
        Check("AABBTree", 15, check_AABBTree, "builds the bounding volume hierarchy"),
        Check("AABBTree_ray_intersect", 10, check_AABBTree_ray_intersect, "ray-mesh query via the tree"),
        Check("ray_intersect_triangle", 10, check_ray_intersect_triangle, "Marschner-Shirley ray-triangle"),
        Check("ray_intersect_box", 10, check_ray_intersect_box, "slab ray-box test"),
        Check("ray_intersect_triangle_mesh_brute_force", 10, check_ray_intersect_triangle_mesh_brute_force, "O(n) ray-mesh"),
        Check("insert_box_into_box", 10, check_insert_box_into_box, "grow a box by a box"),
        Check("insert_triangle_into_box", 10, check_insert_triangle_into_box, "grow a box by a triangle"),
        Check("point_AABBTree_squared_distance", 8, check_point_AABBTree_squared_distance, "nearest neighbour via the tree"),
        Check("nearest_neighbor_brute_force", 6, check_nearest_neighbor_brute_force, "O(n) nearest neighbour"),
        Check("point_box_squared_distance", 6, check_point_box_squared_distance, "point-to-box squared distance"),
        Check("find_all_intersecting_pairs_using_AABBTrees", 3, check_find_all_intersecting_pairs_using_AABBTrees, "broad-phase overlapping pairs"),
        Check("box_box_intersect", 2, check_box_box_intersect, "box-box overlap test"),
    ]
