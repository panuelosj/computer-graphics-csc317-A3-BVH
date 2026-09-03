import numpy as np

from backend.types import Object, BoundingBox
from src.insert_box_into_box import insert_box_into_box


class AABBTree(Object):
    """An axis-aligned bounding-box tree (a simple bounding volume hierarchy).

    Build it by *object partitioning*:

    1. Compute the enclosing box of all input objects by repeatedly growing an
       empty :class:`~backend.types.BoundingBox` with each object's ``.box``
       (use :func:`insert_box_into_box`). Store it in ``self.box``.
    2. If there are 1 or 2 objects, store them directly as the ``left`` (and
       ``right``) leaves -- recursion stops.
    3. Otherwise find the *longest axis* of the enclosing box and split the
       objects by whether their box center is below the midpoint along that
       axis. Recurse on each side to build ``self.left`` / ``self.right`` as
       child ``AABBTree`` nodes (remember to pass ``depth + 1``).

    Watch out for the degenerate case where every object lands on one side: move
    one object across so neither side is empty.

    Attributes you must set: ``self.box``, ``self.left``, ``self.right``,
    ``self.depth``, ``self.num_leaves``. ``left`` / ``right`` are either another
    ``AABBTree`` or a leaf ``Object``; ``right`` may be ``None``.

    Parameters
    ----------
    objects : list[Object]
        Boxable objects to store (non-empty).
    depth : int, optional
        Depth of this node (root is 0).
    """

    def __init__(self, objects, depth=0):
        super().__init__()
        # TODO: set depth/num_leaves, compute the enclosing box, then split.
        raise NotImplementedError("Implement AABBTree.__init__")
