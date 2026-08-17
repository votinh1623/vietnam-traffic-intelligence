"""Pure geometry helpers for normalized ROI and line-crossing analytics."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence


Point = tuple[float, float]


def to_pixels(points: Sequence[Point], width: int, height: int) -> tuple[Point, ...]:
    return tuple((x * width, y * height) for x, y in points)


def polygon_area(polygon: Sequence[Point]) -> float:
    if len(polygon) < 3:
        return 0.0
    return abs(
        sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1])
        )
    ) / 2.0


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        if ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / (y2 - y1) + x1
        ):
            inside = not inside
        previous = current
    return inside


def signed_line_distance(point: Point, start: Point, end: Point) -> float:
    x, y = point
    x1, y1 = start
    x2, y2 = end
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        raise ValueError("line endpoints must differ")
    return ((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)) / length


def stable_line_side(
    point: Point, start: Point, end: Point, tolerance_px: float
) -> int:
    distance = signed_line_distance(point, start, end)
    if abs(distance) <= tolerance_px:
        return 0
    return 1 if distance > 0 else -1


def segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    def orientation(p: Point, q: Point, r: Point) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (
            r[0] - p[0]
        )

    ab_c = orientation(a, b, c)
    ab_d = orientation(a, b, d)
    cd_a = orientation(c, d, a)
    cd_b = orientation(c, d, b)
    return ab_c * ab_d <= 0 and cd_a * cd_b <= 0


def _clip_edge(
    polygon: list[Point],
    inside: Callable[[Point], bool],
    intersection: Callable[[Point, Point], Point],
) -> list[Point]:
    if not polygon:
        return []
    output: list[Point] = []
    previous = polygon[-1]
    previous_inside = inside(previous)
    for current in polygon:
        current_inside = inside(current)
        if current_inside:
            if not previous_inside:
                output.append(intersection(previous, current))
            output.append(current)
        elif previous_inside:
            output.append(intersection(previous, current))
        previous = current
        previous_inside = current_inside
    return output


def polygon_rectangle_intersection_area(
    polygon: Sequence[Point], rectangle: tuple[float, float, float, float]
) -> float:
    """Clip a polygon to an axis-aligned rectangle and return its area."""
    x1, y1, x2, y2 = rectangle
    if x2 <= x1 or y2 <= y1:
        return 0.0
    clipped = list(polygon)

    def vertical_intersection(p: Point, q: Point, x: float) -> Point:
        ratio = 0.0 if q[0] == p[0] else (x - p[0]) / (q[0] - p[0])
        return (x, p[1] + ratio * (q[1] - p[1]))

    def horizontal_intersection(p: Point, q: Point, y: float) -> Point:
        ratio = 0.0 if q[1] == p[1] else (y - p[1]) / (q[1] - p[1])
        return (p[0] + ratio * (q[0] - p[0]), y)

    clipped = _clip_edge(
        clipped, lambda p: p[0] >= x1, lambda p, q: vertical_intersection(p, q, x1)
    )
    clipped = _clip_edge(
        clipped, lambda p: p[0] <= x2, lambda p, q: vertical_intersection(p, q, x2)
    )
    clipped = _clip_edge(
        clipped,
        lambda p: p[1] >= y1,
        lambda p, q: horizontal_intersection(p, q, y1),
    )
    clipped = _clip_edge(
        clipped,
        lambda p: p[1] <= y2,
        lambda p, q: horizontal_intersection(p, q, y2),
    )
    return polygon_area(clipped)
