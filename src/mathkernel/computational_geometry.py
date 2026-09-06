# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
"""Typed computational geometry with exact and ambiguity-aware predicates.

Public text is parsed by :mod:`mathkernel.geometry_adapter`.  This module only
accepts finite, already-parsed SymPy scalars.
"""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from itertools import combinations
import sympy as sp
from pydantic import model_validator

from .engineering import (EngineeringModel, cap_trust, checked_result,
                          validate_scalars)


PointTuple = tuple[sp.Expr, ...]


def _fraction_point(point):
    if not all(isinstance(value, sp.Rational) for value in point):
        return None
    return tuple(Fraction(int(value.p), int(value.q)) for value in point)


def _fraction_orientation(a, b, c):
    value=(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    return (value > 0) - (value < 0)


def _validate_point(point: PointTuple, dimension: int) -> None:
    if len(point) != dimension:
        raise ValueError("point coordinate count must equal the geometry dimension")
    validate_scalars(point, real=True)
    if any(value.free_symbols for value in point):
        raise ValueError("computational-geometry coordinates must be concrete")


class Point(EngineeringModel):
    coordinates: PointTuple
    dimension: int
    input_trust: str = "exact"

    @model_validator(mode="after")
    def validate_point(self):
        if self.dimension < 1:
            raise ValueError("point dimension must be positive")
        _validate_point(self.coordinates, self.dimension)
        cap_trust(self.input_trust)
        return self


class PointSet(EngineeringModel):
    dimension: int
    points: tuple[PointTuple, ...]
    input_trust: str = "exact"

    @model_validator(mode="after")
    def validate_points(self):
        if self.dimension < 1 or not self.points:
            raise ValueError("point set needs a positive dimension and at least one point")
        for point in self.points:
            _validate_point(point, self.dimension)
        if len(set(self.points)) != len(self.points):
            raise ValueError("point-set coordinates must be unique")
        cap_trust(self.input_trust)
        return self


class Polygon(EngineeringModel):
    vertices: tuple[PointTuple, ...]
    input_trust: str = "exact"

    @model_validator(mode="after")
    def validate_polygon(self):
        if len(self.vertices) < 3:
            raise ValueError("polygon needs at least three vertices")
        for point in self.vertices:
            _validate_point(point, 2)
        if len(set(self.vertices)) != len(self.vertices):
            raise ValueError("polygon vertices must be unique")
        cap_trust(self.input_trust)
        return self


class HalfSpace(EngineeringModel):
    normal: PointTuple
    bound: sp.Expr


class Polytope(EngineeringModel):
    dimension: int
    halfspaces: tuple[HalfSpace, ...]
    vertices: tuple[PointTuple, ...] = ()
    input_trust: str = "exact"

    @model_validator(mode="after")
    def validate_polytope(self):
        if self.dimension < 1 or not self.halfspaces:
            raise ValueError("polytope requires a positive dimension and halfspaces")
        for halfspace in self.halfspaces:
            _validate_point(halfspace.normal, self.dimension)
            validate_scalars((halfspace.bound,), real=True)
            if halfspace.bound.free_symbols:
                raise ValueError("polytope bounds must be concrete")
            if all(value.is_zero is True for value in halfspace.normal):
                raise ValueError("halfspace normal must be nonzero")
        for vertex in self.vertices:
            _validate_point(vertex, self.dimension)
        cap_trust(self.input_trust)
        return self


class Triangulation(EngineeringModel):
    points: tuple[PointTuple, ...]
    triangles: tuple[tuple[int, int, int], ...]
    input_trust: str = "exact"

    @model_validator(mode="after")
    def validate_triangulation(self):
        if len(self.points) < 3 or not self.triangles:
            raise ValueError("triangulation requires points and triangles")
        for point in self.points:
            _validate_point(point, 2)
        for triangle in self.triangles:
            if len(triangle) != 3 or len(set(triangle)) != 3:
                raise ValueError("triangles require three distinct indices")
            if any(isinstance(i, bool) or not isinstance(i, int)
                   or i < 0 or i >= len(self.points) for i in triangle):
                raise ValueError("triangle index is outside the point range")
        if len({frozenset(triangle) for triangle in self.triangles}) != len(self.triangles):
            raise ValueError("triangulation cannot contain duplicate triangles")
        cap_trust(self.input_trust)
        return self


class VoronoiRay(EngineeringModel):
    vertex: int
    direction: PointTuple


class VoronoiDiagram(EngineeringModel):
    sites: tuple[PointTuple, ...]
    vertices: tuple[PointTuple, ...]
    edges: tuple[tuple[int, int], ...]
    rays: tuple[VoronoiRay, ...]
    delaunay_triangles: tuple[tuple[int, int, int], ...]
    input_trust: str = "exact"


def _contains_float(values) -> bool:
    return any(value.has(sp.Float) for value in values)


def _sign(value: sp.Expr, scale: sp.Expr | None = None) -> int | None:
    value = sp.sympify(value)
    if not value.has(sp.Float):
        if value.is_zero is True:
            return 0
        if value.is_positive is True:
            return 1
        if value.is_negative is True:
            return -1
        simplified = sp.simplify(value)
        if simplified.is_zero is True: return 0
        if simplified.is_positive is True: return 1
        if simplified.is_negative is True: return -1
        return None
    numeric = float(sp.N(value, 17))
    magnitude = max(1.0, abs(float(sp.N(scale if scale is not None else value, 17))))
    # Filter only: a near-zero approximate determinant cannot decide topology.
    if abs(numeric) <= 64 * 2.220446049250313e-16 * magnitude:
        return None
    return 1 if numeric > 0 else -1


def _comparison(left, right) -> int | None:
    difference = left - right
    return (_sign(difference) if not difference.has(sp.Float) else
            _sign(difference, sp.Abs(left) + sp.Abs(right)))


@lru_cache(maxsize=262_144)
def _orient_value(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])


@lru_cache(maxsize=262_144)
def _orient(a, b, c):
    value = _orient_value(a, b, c)
    if not value.has(sp.Float):
        return _sign(value)
    scale = (sp.Abs((b[0]-a[0])*(c[1]-a[1]))
             + sp.Abs((b[1]-a[1])*(c[0]-a[0])))
    return _sign(value, scale)


def _orientation_nd(points):
    dimension = len(points[0])
    if len(points) != dimension + 1:
        raise ValueError("orientation requires dimension+1 point indices")
    matrix = sp.Matrix([[points[row + 1][column] - points[0][column]
                         for column in range(dimension)]
                        for row in range(dimension)])
    determinant = matrix.det()
    scale = sum(abs(term) for term in sp.Add.make_args(determinant))
    return determinant, _sign(determinant, scale)


@lru_cache(maxsize=262_144)
def _incircle_value(a, b, c, d):
    ax, ay = a[0]-d[0], a[1]-d[1]
    bx, by = b[0]-d[0], b[1]-d[1]
    cx, cy = c[0]-d[0], c[1]-d[1]
    az, bz, cz = ax*ax+ay*ay, bx*bx+by*by, cx*cx+cy*cy
    return ax*(by*cz-bz*cy) - ay*(bx*cz-bz*cx) + az*(bx*cy-by*cx)


@lru_cache(maxsize=262_144)
def _incircle(a, b, c, d):
    orientation = _orient(a, b, c)
    if orientation in (None, 0):
        return None
    value = _incircle_value(a, b, c, d) * orientation
    if not value.has(sp.Float):
        return _sign(value)
    return _sign(value, sum(abs(term) for term in sp.Add.make_args(value)))


def _trust(*objects):
    levels = [obj.input_trust for obj in objects]
    return cap_trust(*levels)


def orientation(point_set, indices):
    selected = tuple(point_set.points[i] for i in indices)
    determinant, sign = _orientation_nd(selected)
    classification = {1: "positive", -1: "negative", 0: "degenerate", None: "ambiguous"}[sign]
    checks = {"predicate_decided": sign is not None if sign is not None else None}
    return checked_result("orientation", {"sign": sign, "classification": classification,
        "determinant": determinant}, method="adaptive_determinant_filter",
        trust=point_set.input_trust, checks=checks,
        details={"indices": tuple(indices), "predicate": "orientation"})


def incircle(point_set, indices):
    if point_set.dimension != 2 or len(indices) != 4:
        raise ValueError("incircle requires four indices in a 2D point set")
    a, b, c, d = (point_set.points[i] for i in indices)
    sign = _incircle(a, b, c, d)
    value = _incircle_value(a, b, c, d)
    classification = {1: "inside", -1: "outside", 0: "cocircular", None: "ambiguous"}[sign]
    return checked_result("incircle", {"sign": sign, "classification": classification,
        "determinant": value}, method="adaptive_incircle_filter",
        trust=point_set.input_trust,
        checks={"predicate_decided": sign is not None if sign is not None else None},
        details={"indices": tuple(indices), "orientation_normalized": True})


def _on_segment(a, b, p):
    if _orient(a, b, p) != 0:
        return False
    comparisons = (_comparison(min(a[0], b[0]), p[0]),
                   _comparison(p[0], max(a[0], b[0])),
                   _comparison(min(a[1], b[1]), p[1]),
                   _comparison(p[1], max(a[1], b[1])))
    if any(value is None for value in comparisons):
        return None
    return all(value <= 0 for value in comparisons)


def _segment_classification(a, b, c, d):
    signs = (_orient(a, b, c), _orient(a, b, d),
             _orient(c, d, a), _orient(c, d, b))
    if any(sign is None for sign in signs):
        return "ambiguous"
    if signs[0] * signs[1] < 0 and signs[2] * signs[3] < 0:
        return "proper"
    for sign, point, left, right in ((signs[0], c, a, b), (signs[1], d, a, b),
                                     (signs[2], a, c, d), (signs[3], b, c, d)):
        if sign == 0:
            on = _on_segment(left, right, point)
            if on is None:
                return "ambiguous"
            if on:
                return "touching"
    return "disjoint"


def segment_intersection(point_set, indices):
    if point_set.dimension != 2 or len(indices) != 4:
        raise ValueError("segment_intersection requires four indices in 2D")
    a, b, c, d = (point_set.points[i] for i in indices)
    classification = _segment_classification(a, b, c, d)
    decided = None if classification == "ambiguous" else True
    return checked_result("segment_intersection", {"classification": classification,
        "intersects": None if decided is None else classification != "disjoint"},
        method="adaptive_orientation_intersection", trust=point_set.input_trust,
        checks={"predicate_decided": decided}, details={"indices": tuple(indices)})


def _polygon_area2(vertices):
    return sum(vertices[i][0]*vertices[(i+1) % len(vertices)][1]
                         - vertices[(i+1) % len(vertices)][0]*vertices[i][1]
                         for i in range(len(vertices)))


def _polygon_properties(vertices):
    n = len(vertices)
    simple = True
    ambiguous = False
    for i in range(n):
        for j in range(i + 1, n):
            if j in {i, (i+1) % n} or i == (j+1) % n:
                continue
            state = _segment_classification(vertices[i], vertices[(i+1) % n],
                                            vertices[j], vertices[(j+1) % n])
            ambiguous |= state == "ambiguous"
            if state != "disjoint" and state != "ambiguous":
                simple = False
    turns = [_orient(vertices[i-1], vertices[i], vertices[(i+1) % n]) for i in range(n)]
    ambiguous |= any(turn is None for turn in turns)
    nonzero = {turn for turn in turns if turn not in (None, 0)}
    convex = None if ambiguous else simple and len(nonzero) <= 1
    area2 = _polygon_area2(vertices)
    orientation_sign = _sign(area2, sum(abs(term) for term in sp.Add.make_args(area2)))
    return {"simple": None if ambiguous and simple else simple, "convex": convex,
            "orientation": orientation_sign, "signed_double_area": area2}


def verify_polygon(polygon):
    properties = _polygon_properties(polygon.vertices)
    checks = {"predicate_decided": None if properties["simple"] is None else True,
              "simple": properties["simple"],
              "nonzero_area": (None if properties["orientation"] is None
                               else properties["orientation"] != 0)}
    return checked_result("verify", properties, method="edge_intersection_and_turn_checks",
        trust=polygon.input_trust, checks=checks, details={"identity_checks": checks})


def _point_in_polygon(vertices, point):
    winding = 0
    for a, b in zip(vertices, vertices[1:] + vertices[:1]):
        orient = _orient(a, b, point)
        if orient is None:
            return "ambiguous"
        if orient == 0:
            on = _on_segment(a, b, point)
            if on is None:
                return "ambiguous"
            if on:
                return "boundary"
        low = _comparison(a[1], point[1]); high = _comparison(b[1], point[1])
        if low is None or high is None:
            return "ambiguous"
        if low <= 0 < high and orient > 0:
            winding += 1
        elif high <= 0 < low and orient < 0:
            winding -= 1
    return "inside" if winding else "outside"


def _point_in_triangle(point, a, b, c):
    """Classify a point against a triangle without assuming its orientation."""
    triangle_sign = _orient(a, b, c)
    signs = (_orient(a, b, point), _orient(b, c, point), _orient(c, a, point))
    if triangle_sign is None or any(sign is None for sign in signs):
        return "ambiguous"
    if triangle_sign == 0:
        return "degenerate"
    normalized = tuple(sign * triangle_sign for sign in signs)
    if all(sign >= 0 for sign in normalized):
        return "boundary" if any(sign == 0 for sign in normalized) else "inside"
    return "outside"


def contains_polygon(polygon, point):
    _validate_point(point, 2)
    properties = _polygon_properties(polygon.vertices)
    if properties["simple"] is None:
        return _ambiguous_result("contains", polygon.input_trust,
                                 "polygon simplicity is numerically ambiguous")
    if properties["simple"] is False:
        raise ValueError("containment is undefined for a self-intersecting polygon")
    classification = _point_in_polygon(polygon.vertices, point)
    decided = None if classification == "ambiguous" else True
    trust = cap_trust(polygon.input_trust,
        "numeric" if _contains_float(point) else "exact")
    return checked_result("contains", {"classification": classification,
        "contains": None if decided is None else classification in {"inside", "boundary"}},
        method="winding_number_with_boundary", trust=trust,
        checks={"predicate_decided": decided}, details={"boundary_included": True})


def _cross_for_hull(a, b, c):
    return _orient(a, b, c)


def convex_hull(point_set):
    if point_set.dimension != 2:
        raise NotImplementedError("convex_hull currently supports exact/filtered 2D point sets")
    fraction_points = [_fraction_point(point) for point in point_set.points]
    rational_path = all(point is not None for point in fraction_points)
    paired = (sorted(zip(fraction_points, point_set.points), key=lambda pair: pair[0])
              if rational_path else
              [(None, point) for point in sorted(point_set.points,
                                                  key=lambda p: (float(p[0]), float(p[1])))])
    points = [point for _, point in paired]
    if len(points) < 3:
        raise ValueError("convex hull requires at least three points")
    lower = []
    for fraction, point in paired:
        while len(lower) >= 2:
            turn = (_fraction_orientation(lower[-2][0], lower[-1][0], fraction)
                    if rational_path else _cross_for_hull(lower[-2][1], lower[-1][1], point))
            if turn is None:
                return _ambiguous_result("convex_hull", point_set.input_trust,
                                         "orientation filter could not decide hull topology")
            if turn > 0:
                break
            lower.pop()
        lower.append((fraction, point))
    upper = []
    for fraction, point in reversed(paired):
        while len(upper) >= 2:
            turn = (_fraction_orientation(upper[-2][0], upper[-1][0], fraction)
                    if rational_path else _cross_for_hull(upper[-2][1], upper[-1][1], point))
            if turn is None:
                return _ambiguous_result("convex_hull", point_set.input_trust,
                                         "orientation filter could not decide hull topology")
            if turn > 0:
                break
            upper.pop()
        upper.append((fraction, point))
    hull_pairs = lower[:-1] + upper[:-1]
    vertices = tuple(point for _, point in hull_pairs)
    if len(vertices) < 3:
        return _ambiguous_result("convex_hull", point_set.input_trust,
                                 "points are collinear or numerically ambiguous")
    polygon = Polygon(vertices=vertices, input_trust=point_set.input_trust)
    if rational_path:
        fraction_hull = [point for point, _ in hull_pairs]
        contained = all(all(_fraction_orientation(fraction_hull[i],
            fraction_hull[(i+1) % len(fraction_hull)], point) >= 0
            for i in range(len(fraction_hull))) for point in fraction_points)
        containment_check = contained
    else:
        containment = [_point_in_polygon(vertices, p) for p in point_set.points]
        containment_check = (None if "ambiguous" in containment else
                             all(state in {"inside", "boundary"} for state in containment))
    checks = {"counterclockwise": _sign(_polygon_area2(vertices)) == 1,
              "all_points_contained": containment_check}
    result = checked_result("convex_hull", vertices, method="monotone_chain",
        trust=point_set.input_trust, checks=checks, witness={"vertices": vertices},
        details={"identity_checks": checks, "input_count": len(points),
                 "hull_count": len(vertices)})
    return result, polygon


def nearest_neighbor(point_set, query):
    _validate_point(query, point_set.dimension)
    fraction_points = [_fraction_point(point) for point in point_set.points]
    fraction_query = _fraction_point(query)
    if fraction_query is not None and all(point is not None for point in fraction_points):
        exact_distances = [sum((a-b)*(a-b) for a,b in zip(point,fraction_query))
                           for point in fraction_points]
        distances = [sp.Rational(value.numerator, value.denominator)
                     for value in exact_distances]
        comparisons = exact_distances
    else:
        distances = [sum((a-b)*(a-b) for a, b in zip(point, query))
                     for point in point_set.points]
        comparisons = distances
    best = 0
    ties = []
    for index in range(1, len(distances)):
        comparison = ((_comparison(comparisons[index], comparisons[best]))
                      if isinstance(comparisons[index], sp.Basic) else
                      (comparisons[index] > comparisons[best]) - (comparisons[index] < comparisons[best]))
        if comparison is None:
            return _ambiguous_result("nearest_neighbor",
                cap_trust(point_set.input_trust, "numeric"),
                "distance filter could not establish a unique ordering")
        if comparison < 0:
            best, ties = index, []
        elif comparison == 0:
            ties.append(index)
    unique = not ties
    trust = cap_trust(point_set.input_trust,
        "numeric" if _contains_float(query) else "exact")
    return checked_result("nearest_neighbor", {"index": best if unique else None,
        "indices": [best, *ties], "point": point_set.points[best] if unique else None,
        "squared_distance": distances[best]}, method="exact_squared_distance_scan",
        trust=trust, checks={"minimum_verified": True, "unique": unique},
        witness={"distances": distances})


def distance_to(point, other, other_id):
    if point.dimension != other.dimension:
        raise ValueError("points must have the same dimension")
    squared = sum((a-b)**2 for a, b in zip(point.coordinates, other.coordinates))
    trust = _trust(point, other)
    return checked_result("distance_to", {"squared_distance": squared,
        "distance": sp.sqrt(squared)}, method="euclidean_distance_identity", trust=trust,
        checks={"squared_distance_nonnegative": squared.is_nonnegative is True},
        details={"other_id": other_id})


def contains_polytope(polytope, point):
    _validate_point(point, polytope.dimension)
    residuals = [sum(a*x for a, x in zip(h.normal, point)) - h.bound
                 for h in polytope.halfspaces]
    signs = [(_sign(value) if not value.has(sp.Float) else
              _sign(value, sum(abs(a*x) for a, x in zip(h.normal, point)) + abs(h.bound)))
             for value, h in zip(residuals, polytope.halfspaces)]
    if any(sign is None for sign in signs):
        return _ambiguous_result("contains", cap_trust(polytope.input_trust, "numeric"),
                                 "halfspace predicate is numerically ambiguous")
    classification = "outside" if any(sign > 0 for sign in signs) else (
        "boundary" if any(sign == 0 for sign in signs) else "inside")
    return checked_result("contains", {"classification": classification,
        "contains": classification != "outside", "residuals": residuals},
        method="halfspace_feasibility", trust=polytope.input_trust,
        checks={"all_halfspaces_checked": True}, details={"boundary_included": True})


def verify_polytope(polytope):
    checks = {"nonzero_normals": all(any(v.is_zero is not True for v in h.normal)
                                     for h in polytope.halfspaces)}
    if polytope.vertices:
        outcomes = [contains_polytope(polytope, vertex) for vertex in polytope.vertices]
        checks["listed_vertices_feasible"] = all(
            outcome.value.get("classification") in {"inside", "boundary"}
            for outcome in outcomes)
    return checked_result("verify", {"halfspace_count": len(polytope.halfspaces),
        "vertex_count": len(polytope.vertices)}, method="halfspace_model_checks",
        trust=polytope.input_trust, checks=checks, details={"identity_checks": checks})


def _line_intersection(a, b, c, d):
    r = (b[0]-a[0], b[1]-a[1]); s = (d[0]-c[0], d[1]-c[1])
    denominator = r[0]*s[1] - r[1]*s[0]
    if _sign(denominator) in (None, 0):
        return None
    t = ((c[0]-a[0])*s[1] - (c[1]-a[1])*s[0]) / denominator
    return (sp.factor(a[0]+t*r[0]), sp.factor(a[1]+t*r[1]))


def polygon_intersection(left, left_id, right, right_id):
    left_props, right_props = _polygon_properties(left.vertices), _polygon_properties(right.vertices)
    if left_props["convex"] is not True or right_props["convex"] is not True:
        raise NotImplementedError("polygon intersection requires verified convex polygons")
    clip = right.vertices if right_props["orientation"] == 1 else tuple(reversed(right.vertices))
    output = list(left.vertices)
    for c, d in zip(clip, clip[1:] + clip[:1]):
        source, output = output, []
        if not source:
            break
        for a, b in zip(source, source[1:] + source[:1]):
            sa, sb = _orient(c, d, a), _orient(c, d, b)
            if sa is None or sb is None:
                return _ambiguous_result("intersection", _trust(left, right),
                                         "clip-edge orientation is ambiguous")
            inside_a, inside_b = sa >= 0, sb >= 0
            if inside_a and inside_b:
                output.append(b)
            elif inside_a and not inside_b:
                point = _line_intersection(a, b, c, d)
                if point is not None: output.append(point)
            elif not inside_a and inside_b:
                point = _line_intersection(a, b, c, d)
                if point is not None: output.append(point)
                output.append(b)
    unique = []
    for point in output:
        if point not in unique:
            unique.append(point)
    trust = _trust(left, right)
    if len(unique) < 3:
        return checked_result("intersection", {"classification": "empty_or_lower_dimensional",
            "vertices": tuple(unique)}, method="sutherland_hodgman_exact", trust=trust,
            checks={"convex_inputs": True})
    polygon = Polygon(vertices=tuple(unique), input_trust=trust)
    checks = {"convex_inputs": True,
              "vertices_in_both": all(_point_in_polygon(left.vertices, p) != "outside"
                                      and _point_in_polygon(right.vertices, p) != "outside"
                                      for p in unique)}
    return checked_result("intersection", tuple(unique), method="sutherland_hodgman_exact",
        trust=trust, checks=checks, witness={"vertices": tuple(unique)},
        details={"identity_checks": checks, "left_id": left_id, "right_id": right_id}), polygon


def triangulate_polygon(polygon, polygon_id):
    props = _polygon_properties(polygon.vertices)
    if props["simple"] is not True:
        return _ambiguous_result("triangulate", polygon.input_trust,
                                 "polygon simplicity is false or ambiguous")
    vertices = polygon.vertices
    order = list(range(len(vertices)))
    if props["orientation"] == -1:
        order.reverse()
    triangles = []
    guard = 0
    while len(order) > 3 and guard < len(vertices)**2:
        guard += 1; clipped = False
        for position, current in enumerate(order):
            previous, following = order[position-1], order[(position+1) % len(order)]
            turn = _orient(vertices[previous], vertices[current], vertices[following])
            if turn is None:
                return _ambiguous_result("triangulate", polygon.input_trust,
                                         "ear orientation is ambiguous")
            if turn <= 0:
                continue
            blocked = False
            triangle_vertices = (vertices[previous], vertices[current], vertices[following])
            for candidate in order:
                if candidate in {previous, current, following}: continue
                states = [_orient(triangle_vertices[i], triangle_vertices[(i+1)%3], vertices[candidate])
                          for i in range(3)]
                if any(state is None for state in states):
                    return _ambiguous_result("triangulate", polygon.input_trust,
                                             "ear containment is ambiguous")
                if all(state >= 0 for state in states): blocked = True; break
            if not blocked:
                triangles.append((previous, current, following)); order.pop(position)
                clipped = True; break
        if not clipped:
            return _ambiguous_result("triangulate", polygon.input_trust,
                                     "no certified ear was available")
    triangles.append(tuple(order))
    triangulation = Triangulation(points=vertices, triangles=tuple(triangles),
                                  input_trust=polygon.input_trust)
    checks = {"triangle_count": len(triangles) == len(vertices)-2,
              "area_partition": sp.simplify(sum(abs(_orient_value(vertices[a], vertices[b], vertices[c]))
                                                 for a,b,c in triangles) - abs(_polygon_area2(vertices))) == 0}
    return checked_result("triangulate", triangles, method="certified_ear_clipping",
        trust=polygon.input_trust, checks=checks, witness={"triangles": triangles},
        details={"identity_checks": checks, "polygon_id": polygon_id}), triangulation


def _delaunay_triangles(points):
    triangles = []
    for triple in combinations(range(len(points)), 3):
        a, b, c = (points[i] for i in triple)
        orient = _orient(a, b, c)
        if orient is None:
            return None, "orientation"
        if orient == 0:
            continue
        cocircular = False; empty = True
        for index, point in enumerate(points):
            if index in triple: continue
            state = _incircle(a, b, c, point)
            if state is None:
                return None, "incircle"
            if state == 0:
                cocircular = True; break
            if state > 0:
                empty = False; break
        if cocircular:
            return None, "cocircular"
        if empty:
            triangles.append(triple if orient > 0 else (triple[0], triple[2], triple[1]))
    return tuple(triangles), None


def delaunay(point_set):
    if point_set.dimension != 2 or len(point_set.points) < 3:
        raise ValueError("delaunay requires at least three 2D points")
    triangles, ambiguity = _delaunay_triangles(point_set.points)
    if ambiguity:
        return _ambiguous_result("delaunay", point_set.input_trust,
            "Delaunay topology is ambiguous due to " + ambiguity)
    hull_result = convex_hull(point_set)
    if not isinstance(hull_result, tuple):
        return _ambiguous_result("delaunay", point_set.input_trust,
                                 "convex hull is ambiguous")
    hull_count = len(hull_result[1].vertices)
    expected = 2*len(point_set.points)-2-hull_count
    checks = {"empty_circumcircle": True, "planar_triangle_count": len(triangles) == expected}
    if not all(checks.values()):
        return _ambiguous_result("delaunay", point_set.input_trust,
                                 "empty-circle faces did not form one triangulation")
    output = Triangulation(points=point_set.points, triangles=triangles,
                           input_trust=point_set.input_trust)
    return checked_result("delaunay", triangles, method="exact_empty_circumcircle",
        trust=point_set.input_trust, checks=checks, witness={"triangles": triangles},
        details={"identity_checks": checks}), output


def _circumcenter(a, b, c):
    denominator = 2*(a[0]*(b[1]-c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1]))
    aa, bb, cc = a[0]**2+a[1]**2, b[0]**2+b[1]**2, c[0]**2+c[1]**2
    return (sp.factor((aa*(b[1]-c[1])+bb*(c[1]-a[1])+cc*(a[1]-b[1]))/denominator),
            sp.factor((aa*(c[0]-b[0])+bb*(a[0]-c[0])+cc*(b[0]-a[0]))/denominator))


def _voronoi_from_triangulation(triangulation):
    centers = tuple(_circumcenter(*(triangulation.points[i] for i in triangle))
                    for triangle in triangulation.triangles)
    edge_owners = {}
    for triangle_index, triangle in enumerate(triangulation.triangles):
        for a, b in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
            edge_owners.setdefault(tuple(sorted((a,b))), []).append(triangle_index)
    edges, rays = [], []
    for (a,b), owners in edge_owners.items():
        if len(owners) == 2:
            edges.append(tuple(sorted(owners)))
        else:
            triangle = triangulation.triangles[owners[0]]
            third = next(i for i in triangle if i not in {a,b})
            pa,pb,pc = triangulation.points[a],triangulation.points[b],triangulation.points[third]
            direction = (pb[1]-pa[1], -(pb[0]-pa[0]))
            if _orient(pa,pb,pc) < 0:
                direction = (-direction[0], -direction[1])
            rays.append(VoronoiRay(vertex=owners[0], direction=direction))
    return VoronoiDiagram(sites=triangulation.points, vertices=centers,
        edges=tuple(sorted(set(edges))), rays=tuple(rays),
        delaunay_triangles=triangulation.triangles,
        input_trust=triangulation.input_trust)


def voronoi(point_set):
    outcome = delaunay(point_set)
    if not isinstance(outcome, tuple):
        return _ambiguous_result("voronoi", point_set.input_trust,
                                 "Voronoi dual is ambiguous because Delaunay is ambiguous")
    diagram = _voronoi_from_triangulation(outcome[1])
    checks = {"delaunay_dual": True,
              "finite_vertex_count": len(diagram.vertices) == len(diagram.delaunay_triangles)}
    return checked_result("voronoi", {"vertices": diagram.vertices,
        "edges": diagram.edges, "rays": diagram.rays}, method="delaunay_dual",
        trust=point_set.input_trust, checks=checks,
        details={"identity_checks": checks}), diagram


def verify_triangulation(triangulation):
    orientations = [_orient(*(triangulation.points[i] for i in triangle))
                    for triangle in triangulation.triangles]
    decided = not any(value is None for value in orientations)
    checks = {"predicates_decided": True if decided else None,
              "nondegenerate_ccw": all(value == 1 for value in orientations) if decided else None}
    edges = []
    directed_edges = []
    for triangle in triangulation.triangles:
        triangle_edges = ((triangle[0], triangle[1]), (triangle[1], triangle[2]),
                          (triangle[2], triangle[0]))
        directed_edges.append(triangle_edges)
        edges.extend(tuple(sorted(edge)) for edge in triangle_edges)
    checks["manifold_edge_incidence"] = all(edges.count(edge) <= 2 for edge in set(edges))
    nonoverlap = True
    for left_index, right_index in combinations(range(len(triangulation.triangles)), 2):
        left = triangulation.triangles[left_index]
        right = triangulation.triangles[right_index]
        shared = set(left) & set(right)
        if len(shared) == 2:
            shared_edge = frozenset(shared)
            left_edge = next(edge for edge in directed_edges[left_index]
                             if frozenset(edge) == shared_edge)
            right_edge = next(edge for edge in directed_edges[right_index]
                              if frozenset(edge) == shared_edge)
            if left_edge == right_edge:
                nonoverlap = False
        for left_edge in directed_edges[left_index]:
            for right_edge in directed_edges[right_index]:
                if set(left_edge) & set(right_edge):
                    continue
                state = _segment_classification(
                    triangulation.points[left_edge[0]], triangulation.points[left_edge[1]],
                    triangulation.points[right_edge[0]], triangulation.points[right_edge[1]])
                if state == "ambiguous" and nonoverlap is True:
                    nonoverlap = None
                elif state != "disjoint":
                    nonoverlap = False
        for vertex in set(left) - shared:
            state = _point_in_triangle(triangulation.points[vertex],
                                       *(triangulation.points[i] for i in right))
            if state == "ambiguous" and nonoverlap is True:
                nonoverlap = None
            elif state in {"inside", "boundary"}:
                nonoverlap = False
        for vertex in set(right) - shared:
            state = _point_in_triangle(triangulation.points[vertex],
                                       *(triangulation.points[i] for i in left))
            if state == "ambiguous" and nonoverlap is True:
                nonoverlap = None
            elif state in {"inside", "boundary"}:
                nonoverlap = False
    checks["nonoverlapping_interiors"] = nonoverlap
    return checked_result("verify", {"triangle_count": len(triangulation.triangles),
        "edge_count": len(set(edges))}, method="triangulation_topology_and_orientation",
        trust=triangulation.input_trust, checks=checks, details={"identity_checks": checks})


def voronoi_from_triangulation(triangulation):
    verified = verify_triangulation(triangulation)
    if not all(value is True for value in verified.verification.values()):
        return _ambiguous_result("voronoi", triangulation.input_trust,
                                 "triangulation is not certified")
    diagram = _voronoi_from_triangulation(triangulation)
    return checked_result("voronoi", {"vertices": diagram.vertices,
        "edges": diagram.edges, "rays": diagram.rays}, method="triangulation_dual",
        trust=triangulation.input_trust, checks={"triangulation_verified": True}), diagram


def _ambiguous_result(operation, trust, diagnostic):
    return checked_result(operation, {"classification": "ambiguous"},
        method="adaptive_predicate_filter", trust=trust,
        checks={"predicate_decided": None}, details={"ambiguity": diagnostic},
        candidate=True)


def cache_info():
    return {"orientation_values": _orient_value.cache_info()._asdict(),
            "orientation_signs": _orient.cache_info()._asdict(),
            "incircle_values": _incircle_value.cache_info()._asdict(),
            "incircle_signs": _incircle.cache_info()._asdict()}


def clear_caches():
    _orient_value.cache_clear(); _orient.cache_clear()
    _incircle_value.cache_clear(); _incircle.cache_clear()
