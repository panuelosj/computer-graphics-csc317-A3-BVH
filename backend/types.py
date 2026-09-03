"""Geometric primitives and "boxable" objects for the BVH assignment.

This is **non-student infrastructure**: the building blocks that the graded
functions in ``src/`` consume and produce: ``BoundingBox``, ``Ray``,
``Object``, ``MeshTriangle`` and ``CloudPoint``.

* :class:`BoundingBox` -- an axis-aligned bounding box stored as two ``(3,)``
  float arrays ``min_corner`` and ``max_corner``. The *default* box is "empty":
  ``min_corner = +inf`` and ``max_corner = -inf``. Growing an empty box by a
  point/box/triangle therefore yields the tightest box around that geometry.
* :class:`Ray` -- an origin and a (not necessarily unit-length) direction.
* :class:`Object` -- the common base class. Every object carries a ``.box``.
  The two concrete leaf objects are :class:`MeshTriangle` (one triangle of a
  mesh) and :class:`CloudPoint` (one point of a point cloud). The student's
  ``AABBTree`` (in ``src/AABBTree.py``) is *also* an ``Object`` so that internal
  nodes and leaves can be mixed freely in a hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_INF = float("inf")


def _empty_min():
    return np.full(3, _INF, dtype=float)


def _empty_max():
    return np.full(3, -_INF, dtype=float)


@dataclass
class BoundingBox:
    """An axis-aligned bounding box defined by two opposite corners.

    Attributes
    ----------
    min_corner : (3,) float ndarray
        Per-axis minimum. Defaults to ``+inf`` (an empty box).
    max_corner : (3,) float ndarray
        Per-axis maximum. Defaults to ``-inf`` (an empty box).
    """

    min_corner: np.ndarray = field(default_factory=_empty_min)
    max_corner: np.ndarray = field(default_factory=_empty_max)

    def center(self) -> np.ndarray:
        """Return the midpoint ``0.5 * (max_corner + min_corner)``."""
        return 0.5 * (self.max_corner + self.min_corner)


class Ray:
    """A parametric ray ``origin + t * direction``.

    ``direction`` need not be unit length (it is sometimes convenient for the
    point ``origin + 1 * direction`` to be meaningful).
    """

    def __init__(self, origin, direction):
        self.origin = np.asarray(origin, dtype=float)
        self.direction = np.asarray(direction, dtype=float)


class Object:
    """Base class for anything that can be stored in an ``AABBTree``.

    Every object carries an axis-aligned bounding ``box``. Subclasses set it in
    their constructor.
    """

    def __init__(self):
        self.box = BoundingBox()


class MeshTriangle(Object):
    """One triangle of a mesh ``(V, F)`` plus its precomputed bounding box.

    Parameters
    ----------
    V : (n, 3) float ndarray
        Mesh vertex positions.
    F : (m, 3) int ndarray
        Triangle vertex indices.
    f : int
        Index of the triangle this object references.
    """

    def __init__(self, V, F, f):
        super().__init__()
        self.V = V
        self.F = F
        self.f = int(f)
        self.a = np.asarray(V[F[f, 0]], dtype=float)
        self.b = np.asarray(V[F[f, 1]], dtype=float)
        self.c = np.asarray(V[F[f, 2]], dtype=float)
        # The leaf box is the tightest axis-aligned box around the 3 corners.
        mn = np.minimum(np.minimum(self.a, self.b), self.c)
        mx = np.maximum(np.maximum(self.a, self.b), self.c)
        self.box = BoundingBox(mn.copy(), mx.copy())


class CloudPoint(Object):
    """One point of a point cloud, with a degenerate (zero-volume) box.

    Parameters
    ----------
    points : (n, 3) float ndarray
        All point-cloud positions.
    i : int
        Index of the point this object references.
    """

    def __init__(self, points, i):
        super().__init__()
        self.points = points
        self.i = int(i)
        p = np.asarray(points[i], dtype=float)
        self.box = BoundingBox(p.copy(), p.copy())
