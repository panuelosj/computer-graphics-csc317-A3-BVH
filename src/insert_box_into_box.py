import numpy as np


def insert_box_into_box(A_box, B_box):
    """Grow box ``B_box`` so that it also encloses box ``A_box``.

    The result is the smallest axis-aligned box containing both inputs::

        B.min_corner = min(B.min_corner, A.min_corner)   (per component)
        B.max_corner = max(B.max_corner, A.max_corner)   (per component)

    ``B_box`` should be mutated in place and also returned for convenience.

    Parameters
    ----------
    A_box, B_box : BoundingBox
        Boxes with ``.min_corner`` / ``.max_corner`` ``(3,)`` float arrays.

    Returns
    -------
    BoundingBox
        ``B_box`` (grown in place).
    """
    # TODO: per-component min/max of the two corners (np.minimum / np.maximum).
    raise NotImplementedError("Implement insert_box_into_box")
