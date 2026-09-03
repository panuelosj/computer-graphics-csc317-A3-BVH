import numpy as np


def nearest_neighbor_brute_force(points, query):
    """Find the nearest point in ``points`` to ``query`` (O(n) reference).

    Parameters
    ----------
    points : (n, 3) float ndarray
    query : (3,) float array

    Returns
    -------
    (I, sqrD) : (int, float)
        Index of the closest point and its squared distance to ``query``. Ties
        should resolve to the lowest index.
    """
    # TODO: squared distances to every point; return argmin and that distance.
    raise NotImplementedError("Implement nearest_neighbor_brute_force")
