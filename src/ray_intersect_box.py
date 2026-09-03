import numpy as np


def ray_intersect_box(ray, box, min_t, max_t):
    """Test whether ``ray`` intersects the *solid* box within ``[min_t, max_t]``.

    Use the classic "slab" method: for each axis the ray enters the slab
    ``[min_corner[c], max_corner[c]]`` at ``tmin[c]`` and leaves at ``tmax[c]``
    (watch the sign of ``1 / direction[c]`` -- a negative direction swaps which
    corner is entered first). The ray hits the box iff the intersection of all
    three per-axis intervals with ``[min_t, max_t]`` is non-empty.

    Because the box is solid, a ray whose origin (or ``min_t`` point) is *inside*
    the box still counts as a hit.

    Parameters
    ----------
    ray : Ray
    box : BoundingBox
    min_t, max_t : float

    Returns
    -------
    bool
        ``True`` iff the ray intersects the box within the range.
    """
    # TODO: compute per-axis [tmin, tmax]; hit iff the running interval is valid.
    raise NotImplementedError("Implement ray_intersect_box")
