from backend.types import MeshTriangle
from src.AABBTree import AABBTree
from src.ray_intersect_box import ray_intersect_box
from src.ray_intersect_triangle import ray_intersect_triangle


def AABBTree_ray_intersect(root, ray, min_t, max_t):
    """Find the closest ray-object hit stored under ``root`` (an ``AABBTree``).

    Recursive depth-first traversal that prunes whole subtrees the ray misses:

    * If the ray misses ``root.box`` (use :func:`ray_intersect_box`), return no
      hit immediately.
    * Otherwise recurse into the children. A child may be another ``AABBTree``
      (recurse) or a leaf (e.g. a ``MeshTriangle`` -- test it directly with
      :func:`ray_intersect_triangle`). Keep the closest hit, and tighten the
      ``max_t`` bound to the current best before descending into the second
      child.

    Return the leaf object that was actually hit as ``descendant`` (so callers
    can recover, e.g., the triangle's face index).

    Parameters
    ----------
    root : AABBTree
    ray : Ray
    min_t, max_t : float

    Returns
    -------
    (hit, t, descendant) : (bool, float, Object or None)
    """
    # TODO: prune on root.box; recurse left/right; keep the closest hit.
    raise NotImplementedError("Implement AABBTree_ray_intersect")
