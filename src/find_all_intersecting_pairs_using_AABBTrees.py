from collections import deque

from src.AABBTree import AABBTree
from src.box_box_intersect import box_box_intersect


def find_all_intersecting_pairs_using_AABBTrees(rootA, rootB):
    """Broad-phase: all overlapping leaf-box pairs between two ``AABBTree``s.

    Descend both hierarchies simultaneously using a simple (non-prioritised)
    queue. Only enqueue a pair of nodes when their boxes overlap
    (:func:`box_box_intersect`) -- this prunes non-overlapping subtrees wholesale
    so the work scales with the number of *actual* box overlaps rather than the
    quadratic number of triangle pairs. When both popped nodes are leaves, record
    that pair.

    Handle every combination: both internal, one leaf / one internal, both
    leaves. ``left`` / ``right`` children may be ``None``.

    Parameters
    ----------
    rootA, rootB : AABBTree

    Returns
    -------
    list[tuple[Object, Object]]
        Each tuple is a pair of overlapping leaf objects ``(leafA, leafB)``.
    """
    # TODO: queue-driven simultaneous descent; record overlapping leaf pairs.
    raise NotImplementedError("Implement find_all_intersecting_pairs_using_AABBTrees")
