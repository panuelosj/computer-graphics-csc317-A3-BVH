import numpy as np


def ray_intersect_triangle(ray, A, B, C, min_t, max_t):
    """Intersect ``ray`` with the triangle ``A, B, C`` (Marschner-Shirley).

    Solve the 3x3 system for the barycentric coordinates ``(beta, gamma)`` and
    the ray parameter ``t`` using Cramer's rule, as in *Fundamentals of Computer
    Graphics*. The hit is valid only when ``min_t <= t < max_t``,
    ``0 <= gamma <= 1`` and ``0 <= beta <= 1 - gamma``.

    Parameters
    ----------
    ray : Ray
    A, B, C : (3,) float arrays
        Triangle corners.
    min_t, max_t : float

    Returns
    -------
    (hit, t) : (bool, float)
        ``hit`` is ``True`` iff the ray hits the triangle within the range; when
        ``hit`` is ``False`` the returned ``t`` need not be meaningful.

    Hint: degenerate triangles make Cramer's denominator zero; wrapping the
    divisions in ``with np.errstate(divide="ignore", invalid="ignore"):``
    silences the resulting RuntimeWarnings without changing the result.
    """
    # TODO: Cramer's rule for t, gamma, beta; range-check each.
    raise NotImplementedError("Implement ray_intersect_triangle")
