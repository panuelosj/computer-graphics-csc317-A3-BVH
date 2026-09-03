import numpy as np


def insert_triangle_into_box(a, b, c, box):
    """Grow ``box`` so that it encloses the triangle with corners ``a, b, c``.

    Each corner is a ``(3,)`` array. After the call::

        box.min_corner = min(box.min_corner, a, b, c)   (per component)
        box.max_corner = max(box.max_corner, a, b, c)   (per component)

    ``box`` should be mutated in place and also returned.

    Parameters
    ----------
    a, b, c : (3,) float arrays
        Triangle corners.
    box : BoundingBox

    Returns
    -------
    BoundingBox
        ``box`` (grown in place).
    """
    # TODO: fold each of the three corners into the box's min/max corners.
    raise NotImplementedError("Implement insert_triangle_into_box")
