"""Lightweight 2D spatial geometry primitives for newspaper layout analysis."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class BBox:
    """Immutable 2D bounding box (x0, y0, x1, y1) in pixel coordinates."""

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        # Guarantee canonical coordinate ordering (x0 <= x1, y0 <= y1)
        if self.x0 > self.x1 or self.y0 > self.y1:
            object.__setattr__(self, "x0", min(self.x0, self.x1))
            object.__setattr__(self, "y0", min(self.y0, self.y1))
            object.__setattr__(self, "x1", max(self.x0, self.x1))
            object.__setattr__(self, "y1", max(self.y0, self.y1))

    @classmethod
    def from_tuple(cls, bbox: tuple[float, float, float, float] | list[float]) -> BBox:
        """Create a BBox instance from a 4-element sequence."""
        if len(bbox) >= 4:
            return cls(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        return cls(0.0, 0.0, 0.0, 0.0)

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        h = self.height
        return self.width / h if h > 0.0 else 0.0

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2.0

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Export as standard 4-tuple (x0, y0, x1, y1)."""
        return (self.x0, self.y0, self.x1, self.y1)

    def horizontal_overlap(self, other: BBox) -> float:
        """Calculate the 1D horizontal span overlap in pixels."""
        return max(0.0, min(self.x1, other.x1) - max(self.x0, other.x0))

    def vertical_overlap(self, other: BBox) -> float:
        """Calculate the 1D vertical span overlap in pixels."""
        return max(0.0, min(self.y1, other.y1) - max(self.y0, other.y0))

    def intersection_area(self, other: BBox) -> float:
        """Calculate the 2D intersection area."""
        dx = self.horizontal_overlap(other)
        dy = self.vertical_overlap(other)
        return dx * dy

    def iou(self, other: BBox) -> float:
        """Compute Intersection-over-Union (IoU) with another bounding box."""
        inter = self.intersection_area(other)
        if inter <= 0.0:
            return 0.0
        union = self.area + other.area - inter
        return inter / union if union > 0.0 else 0.0

    def column_track_overlap_ratio(self, other: BBox) -> float:
        """Calculate symmetric horizontal column track overlap ratio.
        
        Evaluated against max(width_a, width_b) to prevent narrow elements inside wide banner
        headlines from triggering false 100% track matches.
        """
        max_w = max(self.width, other.width)
        if max_w <= 0.0:
            return 0.0
        return self.horizontal_overlap(other) / max_w

    def union(self, other: BBox) -> BBox:
        """Return the minimal bounding box enclosing both boxes."""
        return BBox(
            min(self.x0, other.x0),
            min(self.y0, other.y0),
            max(self.x1, other.x1),
            max(self.y1, other.y1),
        )

    def contains(self, other: BBox) -> bool:
        """Check if this box completely contains another box."""
        return (
            self.x0 <= other.x0
            and self.y0 <= other.y0
            and self.x1 >= other.x1
            and self.y1 >= other.y1
        )
