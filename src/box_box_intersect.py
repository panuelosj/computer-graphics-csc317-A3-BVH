def box_box_intersect(A_box, B_box):
    """Test whether two axis-aligned boxes overlap (separating-axis test).

    Two AABBs overlap iff they overlap on *every* axis. Equivalently they are
    separated (no overlap) if on some axis one box ends before the other begins.

    Parameters
    ----------
    A_box, B_box : BoundingBox

    Returns
    -------
    bool
        ``True`` iff the boxes overlap (touching counts as overlapping).
    """
    # TODO: return False as soon as any axis separates the two boxes.
    raise NotImplementedError("Implement box_box_intersect")
