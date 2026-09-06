# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
"""Restricted public boundary for Phase E geometry objects and operations."""
from __future__ import annotations

import sympy as sp

from .engineering import cap_trust
from .models import TrustLevel


TYPES = {
    "manifold": "Manifold",
    "chart": "Chart", "coordinatechart": "Chart", "coordinate_chart": "Chart",
    "metric": "Metric", "metrictensor": "Metric", "metric_tensor": "Metric",
    "coordinatemap": "CoordinateMap", "coordinate_map": "CoordinateMap",
    "tensorfield": "TensorField", "tensor_field": "TensorField",
    "differentialform": "DifferentialForm", "differential_form": "DifferentialForm",
    "point": "Point", "geometrypoint": "Point", "geometry_point": "Point",
    "pointset": "PointSet", "point_set": "PointSet",
    "polygon": "Polygon", "polytope": "Polytope",
    "triangulation": "Triangulation",
}


def _rebind_coordinates(value, coordinates):
    """Bind parser-created symbols to this chart's canonical real symbols."""
    if isinstance(value, tuple):
        return tuple(_rebind_coordinates(item, coordinates) for item in value)
    if hasattr(value, "free_symbols"):
        by_name = {coordinate.name: coordinate for coordinate in coordinates}
        return value.xreplace({symbol: by_name[symbol.name]
                               for symbol in value.free_symbols
                               if symbol.name in by_name})
    return value

OPERATIONS = {
    "Metric": {
        "inverse_metric": {},
        "christoffel": {},
        "riemann": {},
        "ricci": {},
        "scalar_curvature": {},
        "einstein": {},
        "geodesic_equations": {},
    },
    "CoordinateMap": {"jacobian": {}, "verify": {}},
    "TensorField": {
        "covariant_derivative": {"metric_id": "stored Metric object id"},
        "lie_derivative": {"vector_field_id": "stored contravariant rank-one TensorField object id"},
    },
    "DifferentialForm": {
        "wedge": {"other_id": "stored DifferentialForm object id"},
        "exterior_derivative": {},
        "interior_product": {"vector_field_id": "stored contravariant rank-one TensorField object id"},
        "pullback": {"map_id": "stored CoordinateMap object id"},
        "hodge_star": {"metric_id": "stored Metric object id", "orientation": "1|-1 (defaults to chart orientation)"},
    },
    "Point": {"distance_to": {"other_id": "stored Point object id"}},
    "PointSet": {
        "orientation": {"indices": "dimension+1 point indices"},
        "incircle": {"indices": "four point indices"},
        "segment_intersection": {"indices": "four endpoint indices"},
        "convex_hull": {},
        "nearest_neighbor": {"point": "MathIR coordinate sequence"},
        "delaunay": {},
        "voronoi": {},
    },
    "Polygon": {
        "verify": {}, "contains": {"point": "MathIR coordinate sequence"},
        "intersection": {"other_id": "stored Polygon object id"},
        "triangulate": {},
    },
    "Polytope": {
        "verify": {}, "contains": {"point": "MathIR coordinate sequence"},
    },
    "Triangulation": {"verify": {}, "to_simplicial_complex": {}},
}

DERIVED_OUTPUTS = {
    "inverse_metric": "GeometryTensor",
    "christoffel": "Connection",
    "riemann": "GeometryTensor",
    "ricci": "GeometryTensor",
    "einstein": "GeometryTensor",
    "geodesic_equations": "GeodesicSystem",
    "jacobian": "JacobianMap",
    "covariant_derivative": "TensorField",
    "lie_derivative": "TensorField",
    "wedge": "DifferentialForm",
    "exterior_derivative": "DifferentialForm",
    "interior_product": "DifferentialForm",
    "pullback": "DifferentialForm",
    "hodge_star": "DifferentialForm",
    "convex_hull": "Polygon",
    "intersection": "Polygon",
    "triangulate": "Triangulation",
    "delaunay": "Triangulation",
    "voronoi": "VoronoiDiagram",
    "to_simplicial_complex": "SimplicialComplex",
}


def _record(kernel, object_id, expected):
    record = kernel._get_math_object_record(str(object_id))
    if record is None or record["object_type"] != expected:
        raise ValueError(f"{expected} reference is unavailable")
    return record


def construct(kernel, kind, definition):
    from .differential_geometry import (Chart, CoordinateMap, DifferentialForm,
        FormTerm, Manifold, Metric, TensorField)
    from .engineering_adapter import _parse_tree

    typ = TYPES[kind]
    data = dict(definition)
    context_id = data.pop("context_id", None)
    settings = kernel.settings
    if typ in {"Point", "PointSet", "Polygon", "Polytope", "Triangulation"}:
        from .computational_geometry import (HalfSpace, Point, PointSet, Polygon,
            Polytope, Triangulation)
        def parsed_points(raw, dimension=None):
            if not isinstance(raw, (list, tuple)):
                raise ValueError("points must be a sequence")
            if len(raw) > settings.max_geometry_points:
                raise ValueError("point count exceeds max_geometry_points")
            parsed, trust = _parse_tree(kernel, raw, context_id)
            points = tuple(tuple(point) for point in parsed)
            if dimension is not None and any(len(point) != dimension for point in points):
                raise ValueError("point coordinate count must equal dimension")
            return points, trust
        if typ == "Point":
            allowed = {"coordinates"}; unknown = set(data)-allowed
            if unknown: raise ValueError(f"unknown Point field: {sorted(unknown)[0]}")
            raw = data.get("coordinates", ())
            if not isinstance(raw, (list, tuple)) or not 1 <= len(raw) <= settings.max_geometry_dimension:
                raise ValueError("point dimension is invalid")
            coordinates, trust = _parse_tree(kernel, raw, context_id)
            return typ, Point(coordinates=tuple(coordinates), dimension=len(coordinates),
                              input_trust=trust.value), trust, []
        if typ == "PointSet":
            allowed = {"dimension", "points"}; unknown = set(data)-allowed
            if unknown: raise ValueError(f"unknown PointSet field: {sorted(unknown)[0]}")
            dimension = data.get("dimension")
            if isinstance(dimension, bool) or not isinstance(dimension, int) or not 1 <= dimension <= settings.max_geometry_dimension:
                raise ValueError("point-set dimension is invalid")
            points, trust = parsed_points(data.get("points", ()), dimension)
            return typ, PointSet(dimension=dimension, points=points,
                                 input_trust=trust.value), trust, []
        if typ == "Polygon":
            allowed = {"vertices"}; unknown = set(data)-allowed
            if unknown: raise ValueError(f"unknown Polygon field: {sorted(unknown)[0]}")
            vertices, trust = parsed_points(data.get("vertices", ()), 2)
            return typ, Polygon(vertices=vertices, input_trust=trust.value), trust, []
        if typ == "Polytope":
            allowed = {"dimension", "halfspaces", "vertices"}; unknown = set(data)-allowed
            if unknown: raise ValueError(f"unknown Polytope field: {sorted(unknown)[0]}")
            dimension = data.get("dimension")
            if isinstance(dimension, bool) or not isinstance(dimension, int) or not 1 <= dimension <= settings.max_geometry_dimension:
                raise ValueError("polytope dimension is invalid")
            raw_halfspaces = data.get("halfspaces", ())
            if not isinstance(raw_halfspaces, (list, tuple)) or not raw_halfspaces or len(raw_halfspaces) > settings.max_geometry_simplices:
                raise ValueError("polytope halfspaces must be a bounded nonempty sequence")
            halfspaces=[]; trusts=[]
            for raw in raw_halfspaces:
                if not isinstance(raw, dict) or set(raw) != {"normal", "bound"}:
                    raise ValueError("each halfspace requires only normal and bound")
                normal, nt = _parse_tree(kernel, raw["normal"], context_id)
                bound, bt = _parse_tree(kernel, raw["bound"], context_id)
                halfspaces.append(HalfSpace(normal=tuple(normal), bound=bound)); trusts += [nt.value, bt.value]
            vertices, vt = parsed_points(data.get("vertices", ()), dimension)
            trusts.append(vt.value)
            trust = TrustLevel(cap_trust(*trusts))
            return typ, Polytope(dimension=dimension, halfspaces=tuple(halfspaces),
                vertices=vertices, input_trust=trust.value), trust, []
        allowed = {"points", "triangles"}; unknown = set(data)-allowed
        if unknown: raise ValueError(f"unknown Triangulation field: {sorted(unknown)[0]}")
        points, trust = parsed_points(data.get("points", ()), 2)
        triangles = data.get("triangles", ())
        if not isinstance(triangles, (list, tuple)) or len(triangles) > settings.max_geometry_simplices:
            raise ValueError("triangles must be a bounded sequence")
        for triangle in triangles:
            if not isinstance(triangle, (list, tuple)) or any(isinstance(i, bool) or not isinstance(i, int) for i in triangle):
                raise ValueError("triangle indices must be integers")
        return typ, Triangulation(points=points,
            triangles=tuple(tuple(i for i in triangle) for triangle in triangles),
            input_trust=trust.value), trust, []
    if typ == "Manifold":
        allowed = {"name", "dimension", "orientable"}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown Manifold field: {sorted(unknown)[0]}")
        dimension = data.get("dimension")
        if (isinstance(dimension, bool) or not isinstance(dimension, int)
                or not 1 <= dimension <= settings.max_geometry_dimension):
            raise ValueError("dimension must be a positive integer not exceeding "
                             f"max_geometry_dimension={settings.max_geometry_dimension}")
        manifold = Manifold(name=data.get("name", ""), dimension=dimension,
                            orientable=data.get("orientable"))
        return typ, manifold, TrustLevel.EXACT, []
    if typ == "Chart":
        allowed = {"name", "manifold_id", "coordinates", "domain", "orientation"}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown Chart field: {sorted(unknown)[0]}")
        source = _record(kernel, data.get("manifold_id"), "Manifold")
        manifold = source["value"]
        raw_coordinates = data.get("coordinates", ())
        if len(raw_coordinates) != manifold.dimension:
            raise ValueError("coordinate count must equal manifold dimension")
        parsed_coordinates, coordinate_trust = _parse_tree(kernel, raw_coordinates, context_id)
        coordinates = tuple(sp.Symbol(item.name, real=True)
                            if hasattr(item, "name") else item
                            for item in parsed_coordinates)
        raw_domain = data.get("domain", ())
        domain, domain_trust = _parse_tree(kernel, raw_domain, context_id) if raw_domain else ((), TrustLevel.EXACT)
        domain = _rebind_coordinates(domain, coordinates)
        orientation = data.get("orientation")
        if isinstance(orientation, bool) or orientation not in {None, -1, 1}:
            raise ValueError("orientation must be 1, -1, or omitted")
        trust = TrustLevel(cap_trust(source["input_trust"].value,
                                     coordinate_trust.value, domain_trust.value))
        chart = Chart(name=data.get("name", ""), manifold_id=str(data["manifold_id"]),
            manifold_name=manifold.name, coordinates=coordinates, domain=domain,
            orientation=orientation, input_trust=trust.value)
        return typ, chart, trust, [str(data["manifold_id"])]
    if typ == "CoordinateMap":
        allowed = {"name", "source_chart_id", "target_chart_id", "forward", "inverse"}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown CoordinateMap field: {sorted(unknown)[0]}")
        source_record = _record(kernel, data.get("source_chart_id"), "Chart")
        target_record = _record(kernel, data.get("target_chart_id"), "Chart")
        source, target = source_record["value"], target_record["value"]
        if source.manifold_id != target.manifold_id:
            raise ValueError("coordinate-map charts must belong to the same manifold")
        if len(source.coordinates) != len(target.coordinates):
            raise ValueError("coordinate-map charts must have the same dimension")
        raw_forward = data.get("forward", ())
        raw_inverse = data.get("inverse")
        if len(raw_forward) != len(target.coordinates):
            raise ValueError("forward map must define every target coordinate")
        forward, forward_trust = _parse_tree(kernel, raw_forward, context_id)
        forward = _rebind_coordinates(forward, source.coordinates)
        inverse = None; inverse_trust = TrustLevel.EXACT
        if raw_inverse is not None:
            if len(raw_inverse) != len(source.coordinates):
                raise ValueError("inverse map must define every source coordinate")
            inverse, inverse_trust = _parse_tree(kernel, raw_inverse, context_id)
            inverse = _rebind_coordinates(inverse, target.coordinates)
        trust = TrustLevel(cap_trust(source_record["input_trust"].value,
            target_record["input_trust"].value, forward_trust.value, inverse_trust.value))
        mapping = CoordinateMap(name=data.get("name", "F"),
            source_chart_id=str(data["source_chart_id"]), target_chart_id=str(data["target_chart_id"]),
            manifold_id=source.manifold_id, source_coordinates=source.coordinates,
            target_coordinates=target.coordinates, forward=forward, inverse=inverse,
            source_domain=source.domain, target_domain=target.domain, input_trust=trust.value)
        return typ, mapping, trust, [str(data["source_chart_id"]), str(data["target_chart_id"])]
    if typ == "TensorField":
        allowed = {"name", "chart_id", "variance", "components"}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown TensorField field: {sorted(unknown)[0]}")
        source = _record(kernel, data.get("chart_id"), "Chart")
        chart = source["value"]
        variance = tuple(data.get("variance", ()))
        if not variance or any(item not in {"up", "down"} for item in variance):
            raise ValueError("variance must be a non-empty sequence of up/down entries")
        if len(variance) > settings.max_geometry_rank:
            raise ValueError("tensor rank exceeds max_geometry_rank")
        if len(chart.coordinates) ** len(variance) > settings.max_geometry_work:
            raise ValueError("tensor field exceeds max_geometry_work")
        components, parsed_trust = _parse_tree(kernel, data.get("components"), context_id)
        components = _rebind_coordinates(components, chart.coordinates)
        trust = TrustLevel(cap_trust(source["input_trust"].value, parsed_trust.value))
        tensor = TensorField(name=data.get("name", "T"), chart_id=str(data["chart_id"]),
            manifold_id=chart.manifold_id, coordinates=chart.coordinates, domain=chart.domain,
            variance=variance, components=components, input_trust=trust.value)
        return typ, tensor, trust, [str(data["chart_id"])]
    if typ == "DifferentialForm":
        allowed = {"name", "chart_id", "degree", "terms"}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown DifferentialForm field: {sorted(unknown)[0]}")
        source = _record(kernel, data.get("chart_id"), "Chart")
        chart = source["value"]
        degree = data.get("degree")
        if isinstance(degree, bool) or not isinstance(degree, int) or not 0 <= degree <= len(chart.coordinates):
            raise ValueError("degree must be an integer within the chart dimension")
        raw_terms = data.get("terms", ())
        if not isinstance(raw_terms, (list, tuple)) or len(raw_terms) > settings.max_geometry_work:
            raise ValueError("form terms must be a bounded sequence")
        terms = []
        trusts = [source["input_trust"].value]
        for raw_term in raw_terms:
            if not isinstance(raw_term, dict) or set(raw_term) != {"indices", "coefficient"}:
                raise ValueError("each form term requires only indices and coefficient")
            indices = raw_term["indices"]
            if not isinstance(indices, (list, tuple)) or any(isinstance(i, bool) or not isinstance(i, int) for i in indices):
                raise ValueError("form indices must be integer sequences")
            coefficient, parsed_trust = _parse_tree(kernel, raw_term["coefficient"], context_id)
            coefficient = _rebind_coordinates(coefficient, chart.coordinates)
            trusts.append(parsed_trust.value)
            terms.append(FormTerm(indices=tuple(indices), coefficient=coefficient))
        trust = TrustLevel(cap_trust(*trusts))
        form = DifferentialForm(name=data.get("name", "omega"), chart_id=str(data["chart_id"]),
            manifold_id=chart.manifold_id, coordinates=chart.coordinates, domain=chart.domain,
            degree=degree, terms=tuple(terms), input_trust=trust.value)
        return typ, form, trust, [str(data["chart_id"])]
    allowed = {"name", "chart_id", "components", "signature"}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown Metric field: {sorted(unknown)[0]}")
    source = _record(kernel, data.get("chart_id"), "Chart")
    chart = source["value"]
    raw = data.get("components", ())
    n = len(chart.coordinates)
    if len(raw) != n or any(not isinstance(row, (list, tuple)) or len(row) != n for row in raw):
        raise ValueError("metric must be square with the chart dimension")
    if n**5 > settings.max_geometry_work:
        raise ValueError("metric exceeds max_geometry_work")
    components, parsed_trust = _parse_tree(kernel, raw, context_id)
    components = _rebind_coordinates(components, chart.coordinates)
    trust = TrustLevel(cap_trust(source["input_trust"].value, parsed_trust.value))
    metric = Metric(name=data.get("name", "g"), chart_id=str(data["chart_id"]),
        manifold_id=chart.manifold_id, coordinates=chart.coordinates,
        domain=chart.domain, components=components, signature=data.get("signature"),
        input_trust=trust.value)
    return typ, metric, trust, [str(data["chart_id"])]


def apply(kernel, value, operation, parameters, object_id):
    from . import differential_geometry as geometry
    computational_types = {"Point", "PointSet", "Polygon", "Polytope", "Triangulation"}
    if type(value).__name__ in computational_types:
        from . import computational_geometry as cg
        from .engineering_adapter import _parse_tree
        count = max(1, len(getattr(value, "points", ())),
                    len(getattr(value, "vertices", ())),
                    len(getattr(value, "triangles", ())),
                    len(getattr(value, "halfspaces", ())))
        work = {"delaunay": count**4, "voronoi": count**4,
                "convex_hull": count**2, "triangulate": count**3,
                "intersection": count**2, "verify": count**2}.get(operation, max(1, count))
        if work > kernel.settings.max_geometry_work:
            raise ValueError(f"{operation} exceeds max_geometry_work")
        def indices(expected=None, *, distinct=True):
            raw = parameters.get("indices")
            if not isinstance(raw, (list, tuple)) or any(isinstance(i, bool) or not isinstance(i, int) for i in raw):
                raise ValueError("indices must be an integer sequence")
            if expected is not None and len(raw) != expected:
                raise ValueError(f"operation requires {expected} indices")
            if any(i < 0 or i >= len(value.points) for i in raw) or (distinct and len(set(raw)) != len(raw)):
                raise ValueError("indices must be within the point set and distinct where required")
            return tuple(raw)
        def point_parameter(dimension):
            raw = parameters.get("point")
            if not isinstance(raw, (list, tuple)) or len(raw) != dimension:
                raise ValueError("point parameter must match the geometry dimension")
            parsed, _ = _parse_tree(kernel, raw, parameters.get("context_id"))
            return tuple(parsed)
        def reference(name, expected):
            record = _record(kernel, parameters.get(name), expected)
            return record["value"], str(parameters[name])
        if operation == "distance_to":
            other, other_id = reference("other_id", "Point")
            return cg.distance_to(value, other, other_id)
        if operation == "orientation": return cg.orientation(value, indices(value.dimension+1))
        if operation == "incircle": return cg.incircle(value, indices(4))
        if operation == "segment_intersection":
            selected = indices(4, distinct=False)
            if selected[0] == selected[1] or selected[2] == selected[3]:
                raise ValueError("segment endpoints must be distinct")
            return cg.segment_intersection(value, selected)
        if operation == "convex_hull": return cg.convex_hull(value)
        if operation == "nearest_neighbor": return cg.nearest_neighbor(value, point_parameter(value.dimension))
        if operation == "delaunay": return cg.delaunay(value)
        if operation == "voronoi" and type(value).__name__ == "PointSet": return cg.voronoi(value)
        if operation == "verify" and type(value).__name__ == "Polygon": return cg.verify_polygon(value)
        if operation == "contains" and type(value).__name__ == "Polygon": return cg.contains_polygon(value, point_parameter(2))
        if operation == "intersection":
            other, other_id = reference("other_id", "Polygon")
            return cg.polygon_intersection(value, object_id, other, other_id)
        if operation == "triangulate": return cg.triangulate_polygon(value, object_id)
        if operation == "verify" and type(value).__name__ == "Polytope": return cg.verify_polytope(value)
        if operation == "contains" and type(value).__name__ == "Polytope": return cg.contains_polytope(value, point_parameter(value.dimension))
        if operation == "verify" and type(value).__name__ == "Triangulation": return cg.verify_triangulation(value)
        if operation == "to_simplicial_complex":
            from .algebraic_topology import triangulation_to_simplicial_complex
            return triangulation_to_simplicial_complex(
                value, max_cells=kernel.settings.max_topology_cells,
                max_matrix_entries=kernel.settings.max_topology_matrix_entries,
                max_work=kernel.settings.max_topology_work)
        raise NotImplementedError(f"unsupported computational geometry operation {operation}")
    n = len(getattr(value, "coordinates", getattr(value, "source_coordinates", ())))
    rank = len(getattr(value, "variance", ()))
    powers = {"inverse_metric": 3, "christoffel": 4, "geodesic_equations": 4,
              "jacobian": 2, "verify": 2, "wedge": 3,
              "exterior_derivative": 2, "interior_product": rank + 1,
              "pullback": 4, "hodge_star": 4,
              "covariant_derivative": rank + 2, "lie_derivative": rank + 2}
    required = n ** powers.get(operation, 5)
    if required > kernel.settings.max_geometry_work:
        raise ValueError(f"{operation} exceeds max_geometry_work")
    if operation == "inverse_metric":
        return geometry.inverse_metric(value, object_id)
    if operation == "christoffel":
        return geometry.christoffel(value, object_id)
    if operation in {"riemann", "ricci", "scalar_curvature", "einstein"}:
        return geometry.curvature(value, object_id, operation)
    if operation == "geodesic_equations":
        return geometry.geodesic_equations(value, object_id)
    if operation == "jacobian":
        return geometry.coordinate_jacobian(value, object_id)
    if operation == "verify":
        return geometry.verify_coordinate_map(value)
    def referenced(name, expected):
        record = _record(kernel, parameters.get(name), expected)
        return record["value"], str(parameters[name])
    if operation == "covariant_derivative":
        metric, metric_id = referenced("metric_id", "Metric")
        if metric.chart_id != value.chart_id:
            raise ValueError("metric and tensor field must use the same chart")
        return geometry.covariant_derivative(value, object_id, metric, metric_id)
    if operation == "lie_derivative":
        vector, vector_id = referenced("vector_field_id", "TensorField")
        if vector.chart_id != value.chart_id or vector.variance != ("up",):
            raise ValueError("vector field must be contravariant rank one on the same chart")
        return geometry.lie_derivative(value, object_id, vector, vector_id)
    if operation == "wedge":
        other, other_id = referenced("other_id", "DifferentialForm")
        if other.chart_id != value.chart_id:
            raise ValueError("forms must use the same chart")
        return geometry.wedge(value, object_id, other, other_id)
    if operation == "exterior_derivative":
        return geometry.exterior_derivative(value, object_id)
    if operation == "interior_product":
        vector, vector_id = referenced("vector_field_id", "TensorField")
        if vector.chart_id != value.chart_id or vector.variance != ("up",):
            raise ValueError("vector field must be contravariant rank one on the same chart")
        return geometry.interior_product(value, object_id, vector, vector_id)
    if operation == "pullback":
        mapping, map_id = referenced("map_id", "CoordinateMap")
        if mapping.target_chart_id != value.chart_id:
            raise ValueError("form chart must be the coordinate-map target")
        return geometry.pullback_form(value, object_id, mapping, map_id)
    if operation == "hodge_star":
        metric, metric_id = referenced("metric_id", "Metric")
        if metric.chart_id != value.chart_id:
            raise ValueError("metric and form must use the same chart")
        orientation = parameters.get("orientation")
        if orientation is None:
            chart = _record(kernel, value.chart_id, "Chart")["value"]
            orientation = chart.orientation
        if isinstance(orientation, bool) or orientation not in {-1, 1}:
            raise ValueError("hodge_star requires orientation=1|-1 or an oriented chart")
        return geometry.hodge_star(value, object_id, metric, metric_id, orientation)
    raise NotImplementedError(f"unsupported geometry operation {operation}")
