import numpy as np

from src.ray_intersect_triangle import ray_intersect_triangle


def ray_intersect_triangle_mesh_brute_force(ray, V, F, min_t, max_t):
    """Shoot ``ray`` at every triangle of ``(V, F)`` and keep the closest hit.

    This is the O(n) reference implementation: loop over all ``m`` faces and test
    each triangle with :func:`ray_intersect_triangle`, using the running closest
    distance as the upper bound so only nearer hits replace the current best.

    Parameters
    ----------
    ray : Ray
    V : (n, 3) float ndarray
    F : (m, 3) int ndarray
    min_t, max_t : float

    Returns
    -------
    (hit, hit_t, hit_f) : (bool, float, int)
        ``hit`` is ``True`` iff any triangle was hit; ``hit_t`` is the parametric
        distance of the closest hit and ``hit_f`` its face index (``-1`` if no
        hit).

    Note: you may initialize the closest-hit tracker
    to infinity -- the incoming ``max_t`` bound is not exercised by the checks.
    """
    # TODO: track the running closest hit over all faces.
    raise NotImplementedError("Implement ray_intersect_triangle_mesh_brute_force")
