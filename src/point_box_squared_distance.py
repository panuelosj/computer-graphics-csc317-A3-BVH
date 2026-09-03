import numpy as np


def point_box_squared_distance(query, box):
    """Squared distance from ``query`` to the closest point of the solid ``box``.

    The closest point is ``query`` clamped into the box. Per axis the signed
    overshoot is ``max(0, min_corner - query, query - max_corner)``; the squared
    distance is the squared length of that vector. A query *inside* the box has
    distance ``0``.

    Parameters
    ----------
    query : (3,) float array
    box : BoundingBox

    Returns
    -------
    float
        The squared distance.
    """
    # TODO: clamp the per-axis overshoot at 0 on both sides; return its sq. norm.
    raise NotImplementedError("Implement point_box_squared_distance")
