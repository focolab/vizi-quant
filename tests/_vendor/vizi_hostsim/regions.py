"""Generic regions and region types a plugin shapes:
the host's half of `control_plane_protocol.md` §8.8's two subsections
*What a contributed region may be* and *Region types a plugin shapes*.

Pure functions over `pois_math.v1.Roi` messages, so a driver can call them
too:

- :func:`contribution_refusal` -- why the host refuses a contributed region
  under §8.8's rules (the three rules, `vizi.`, `SHELL`/`POINTS`, a
  `DirectionHandle` on an asked region, and the checks on a handle's form);
- :func:`closure_problem` (rule 1) and :func:`star_problem` (rule 2);
- :func:`surface_problem` and :func:`shape_result_problem` -- an asked
  shape the host refuses, including one with no inside;
- :func:`project_onto_dof`, :func:`constrain` and :func:`propagate` -- a
  handle drag as the ROI scheme's §4 and §5 run it:
  the host enacts a DECLARED region's drag with all three, and for an ASKED
  one sends the projected point and leaves the rest to the plugin.

These follow the spec's text; where vizi's host checks something itself,
they follow vizi.
"""

from __future__ import annotations

import numpy as np

from vizi_plugin_sdk import roi_pb2

#: `Roi.type_name`s the host draws itself (`manifest_schema.md` §3.8).
BUILTIN_TYPES = frozenset({"box", "ellipsoid", "cylinder", "surfq6"})
RESERVED_PREFIX = "vizi."

_ROLE_ENCLOSING = (roi_pb2.ELEMENT_BOTH, roi_pb2.ELEMENT_ENCLOSING)
_EPS = 1e-9
_PRIMARY, _SECONDARY = roi_pb2.PRIMARY, roi_pb2.SECONDARY
_AXIS_RADIAL = roi_pb2.AXIS_RADIAL
_OK, _CLAMPED = roi_pb2.OK, roi_pb2.CLAMPED


def _vec(v) -> np.ndarray:
    return np.array([v.z, v.y, v.x], dtype=np.float64)


def positions(roi) -> np.ndarray:
    """Every vertex's position, ZYX, as an (n, 3) array."""
    return np.array([_vec(v.position) for v in roi.vertices], dtype=np.float64).reshape(-1, 3)


def display_positions(roi) -> np.ndarray:
    """The display vertices: the compile operator's rows over its controls
    when the region has one, else the stored positions.  Edges, facets and
    patches index these."""
    stored = positions(roi)
    if not roi.HasField("compile"):
        return stored
    out = np.zeros((int(roi.compile.num_display), 3), dtype=np.float64)
    controls = list(roi.compile.controls)
    for t in roi.compile.triplets:
        out[t.display_vertex] += t.coeff * stored[controls[t.control]]
    return out


def set_positions(roi, points) -> None:
    for vertex, p in zip(roi.vertices, np.asarray(points, dtype=np.float64)):
        vertex.position.z, vertex.position.y, vertex.position.x = (float(c) for c in p)


def _enclosing_surface_declared(roi) -> bool:
    return any(f.role in _ROLE_ENCLOSING for f in roi.facets) or any(p.role in _ROLE_ENCLOSING for p in roi.patches)


def exposing_rows(roi) -> dict[int, int]:
    """Rule 0: each control's EXPOSING display row -- the first
    row of the compile operator `S` whose only nonzero is exactly 1 in that
    control's column.  With no operator every control is its own row.
    Controls no row exposes are absent."""
    if not roi.HasField("compile"):
        return {i: i for i in range(len(roi.vertices))}
    rows: dict[int, dict[int, float]] = {}
    for t in roi.compile.triplets:
        cells = rows.setdefault(int(t.display_vertex), {})
        cells[int(t.control)] = cells.get(int(t.control), 0.0) + float(t.coeff)
    controls = list(roi.compile.controls)
    out: dict[int, int] = {}
    for row in sorted(rows):
        nonzero = [(col, v) for col, v in rows[row].items() if v != 0.0]
        if len(nonzero) == 1 and nonzero[0][1] == 1.0:
            control = controls[nonzero[0][0]]
            out.setdefault(control, row)
    return out


def rule0_problem(roi) -> str | None:
    """Rule 0's refusal: a control marked `enclosing` that no display row
    exposes (in every inclusion mode)."""
    exposed = exposing_rows(roi)
    for i, v in enumerate(roi.vertices):
        if v.enclosing and i not in exposed:
            return f"vertex {i} is marked enclosing, and no row of the compile operator exposes it (rule 0)"
    return None


def enclosing_vertices(roi) -> list[int]:
    """The ROI scheme's §8.2 point set, in DISPLAY indices (rule 0): the
    display vertex exposing each `enclosing` control, ENCLOSING/BOTH edge
    endpoints and facet corners.  Sorted, unique."""
    exposed = exposing_rows(roi)
    out = {exposed[i] for i, v in enumerate(roi.vertices) if v.enclosing and i in exposed}
    for e in roi.edges:
        if e.role in _ROLE_ENCLOSING:
            out.update((e.source, e.target))
    for f in roi.facets:
        if f.role in _ROLE_ENCLOSING:
            out.update(f.vertices)
    return sorted(out)


# -- rule 1: closure ---------------------------------------------------------


def closure_problem(roi) -> str | None:
    """Rule 1: ``None`` when the ENCLOSING/BOTH facets and patches form a
    closed surface -- every facet side and patch boundary row occurring
    exactly twice, as one index sequence read either way -- else a sentence
    naming one that does not.  A patch row whose controls are all one index
    (a collapsed corner) contributes nothing."""
    counts: dict[tuple[int, ...], int] = {}

    def add(seq) -> None:
        seq = tuple(int(i) for i in seq)
        if len(set(seq)) == 1:
            return
        key = min(seq, seq[::-1])
        counts[key] = counts.get(key, 0) + 1

    for facet in roi.facets:
        if facet.role not in _ROLE_ENCLOSING:
            continue
        cycle = list(facet.vertices)
        for a, b in zip(cycle, cycle[1:] + cycle[:1]):
            add((a, b))
    for patch in roi.patches:
        if patch.role not in _ROLE_ENCLOSING:
            continue
        grid = np.asarray(patch.controls, dtype=np.int64).reshape(patch.nu, patch.nv)
        for row in (grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]):
            add(row)
    for key, n in sorted(counts.items()):
        if n != 2:
            what = "side" if len(key) == 2 else "boundary row"
            return f"the enclosing surface is not closed: {what} {list(key)} occurs {n} time{'s' if n != 1 else ''}, not twice"
    return None


# -- rule 2: STAR ------------------------------------------------------------


def _hull_faces(points: np.ndarray) -> list[tuple[np.ndarray, float, list[int]]]:
    """The supporting planes of the convex hull of ``points`` (brute force,
    as the host computes it: fine for the few points a STAR region has):
    ``(outward unit normal, offset, members)`` per face."""
    n = len(points)
    centroid = points.mean(axis=0)
    faces: list[tuple[np.ndarray, float, list[int]]] = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                normal = np.cross(points[j] - points[i], points[k] - points[i])
                length = float(np.linalg.norm(normal))
                if length < _EPS:
                    continue
                u = normal / length
                if np.dot(u, points[i] - centroid) < 0:
                    u = -u
                proj = points @ u
                offset = float(points[i] @ u)
                if np.any(proj > offset + _EPS):
                    continue
                if any(np.dot(u, seen) > 1 - _EPS for seen, _, _ in faces):
                    continue
                faces.append((u, offset, [m for m in range(n) if proj[m] >= offset - _EPS]))
    return faces


def star_cloud(roi) -> tuple[np.ndarray, list[int]]:
    """Rule 2's center and cloud: the `root` vertex's position, and every
    OTHER enclosing vertex."""
    root = int(roi.root)
    shown = exposing_rows(roi).get(root)
    return positions(roi)[root], [i for i in enclosing_vertices(roi) if i != shown]


def star_problem(roi) -> str | None:
    """Rule 2's refusals, or ``None``: fewer than 4 points, one at the
    center, two on one ray, or directions that do not STRICTLY surround the
    center (the origin strictly inside the hull of the unit directions), with
    absolute tolerances: at the center within 1e-12, one ray when
    1 - d.d' < 1e-12, strictly surrounding when every face's offset > 1e-12
    (and :func:`_hull_faces` takes a direction as on a face within 1e-9)."""
    c, cloud = star_cloud(roi)
    if len(cloud) < 4:
        return f"STAR needs at least 4 points about the root, and the region has {len(cloud)}"
    pts = display_positions(roi)
    dirs = []
    for i in cloud:
        offset = pts[i] - c
        length = float(np.linalg.norm(offset))
        if length <= 1e-12:
            return f"STAR: vertex {i} coincides with the root (vertex {int(roi.root)})"
        dirs.append(offset / length)
    d = np.array(dirs)
    for a in range(len(d)):
        for b in range(a + 1, len(d)):
            if 1.0 - float(d[a] @ d[b]) < 1e-12:
                return f"STAR: vertices {cloud[a]} and {cloud[b]} lie on one ray from the root"
    if np.linalg.matrix_rank(d - d[0], tol=1e-9) < 3:
        return "STAR: the points' directions from the root do not surround it (they lie on one plane)"
    faces = _hull_faces(d)
    if not faces or any(offset <= 1e-12 for _, offset, _ in faces):
        return "STAR: the points' directions from the root do not strictly surround it"
    return None


def star_triangles(roi) -> list[tuple[int, int, int]]:
    """Rule 2's surface as vertex-index triangles facing away from the root:
    the faces of the hull of the DIRECTIONS, each at the points' own
    positions, a face with more than three corners fanned from its
    lowest-indexed vertex.  Call only when :func:`star_problem` is ``None``."""
    c, cloud = star_cloud(roi)
    pts = display_positions(roi)
    d = np.array([(pts[i] - c) / np.linalg.norm(pts[i] - c) for i in cloud])
    out: list[tuple[int, int, int]] = []
    for u, _, members in _hull_faces(d):
        ring = [cloud[m] for m in members]
        centre = d[members].mean(axis=0)
        ref = d[members[0]] - centre
        ref /= np.linalg.norm(ref)
        ortho = np.cross(u, ref)
        order = sorted(range(len(ring)), key=lambda m: np.arctan2(np.dot(d[members[m]] - centre, ortho), np.dot(d[members[m]] - centre, ref)))
        ring = [ring[m] for m in order]
        start = ring.index(min(ring))
        ring = ring[start:] + ring[:start]
        for t in range(1, len(ring) - 1):
            a, b, e = ring[0], ring[t], ring[t + 1]
            if np.dot(np.cross(pts[b] - pts[a], pts[e] - pts[a]), pts[a] - c) < 0:
                b, e = e, b
            out.append((a, b, e))
    return out


# -- what a contributed region may be ----------------------------------------


def contribution_refusal(roi, asked_types: "frozenset[str] | set[str]" = frozenset()) -> str | None:
    """§8.8 *What a contributed region may be*: why the host refuses ``roi``
    (`INVALID_ARGUMENT`, the whole batch), or ``None``.  ``asked_types`` is
    the contributing plugin's `region_types`.  Non-finite numbers and the
    pose domain are the caller's (`Host._roi_refusal`)."""
    name = roi.type_name
    if name.startswith(RESERVED_PREFIX):
        return f"type_name {name!r}: names beginning {RESERVED_PREFIX!r} are reserved for the host's built-ins"
    if roi.views:
        return "ROI viewpoints are not accepted from plugins"
    if roi.direction_specs:
        return "ROI direction_specs are not supported"
    n = len(roi.vertices)
    # Edges, facets and patches index the DISPLAY vertices: the compile operator's rows when there is one.
    shown = int(roi.compile.num_display) if roi.HasField("compile") else n

    def bad(i) -> bool:
        return not 0 <= int(i) < n

    def bad_shown(i) -> bool:
        return not 0 <= int(i) < shown

    for i, vertex in enumerate(roi.vertices):
        if vertex.HasField("handle"):
            if len(vertex.handle.frame) != 9:
                return f"vertex {i}'s handle frame must contain 9 values"
            if bad(vertex.handle.extent_anchor):
                return f"vertex {i}'s handle extent_anchor {vertex.handle.extent_anchor} is out of range"
        if name in asked_types and vertex.HasField("direction_handle"):
            return f"vertex {i} has a DirectionHandle, which an asked region may not carry (§8.8)"
    if n and bad(roi.root):
        return f"root {roi.root} is out of range"
    for k, e in enumerate(roi.edges):
        if bad_shown(e.source) or bad_shown(e.target) or (e.control >= 0 and bad_shown(e.control)):
            return f"edge {k} names a vertex out of range"
    for k, f in enumerate(roi.facets):
        if len(f.vertices) < 3 or any(bad_shown(i) for i in f.vertices):
            return f"facet {k} needs at least 3 vertices, all in range"
    for k, p in enumerate(roi.patches):
        if p.nu < 2 or p.nv < 2 or len(p.controls) != p.nu * p.nv or any(bad_shown(i) for i in p.controls):
            return f"patch {k} needs nu*nv controls (nu, nv >= 2), all in range"
    for k, b in enumerate(roi.bindings):
        if bad(b.moved) or bad(b.target) or len(b.m) != 9:
            return f"binding {k} needs moved and target in range and 9 values of m"
    why = rule0_problem(roi)
    if why is not None:
        return why
    mode = roi.inclusion.mode
    if mode in (roi_pb2.INCLUDE_SHELL, roi_pb2.INCLUDE_POINTS):
        return f"inclusion mode {roi_pb2.InclusionMode.Name(mode)} is refused from a plugin: its thickness or radius would decide what is inside"
    if mode == roi_pb2.INCLUDE_ENCLOSED and name not in BUILTIN_TYPES:
        if _enclosing_surface_declared(roi):
            return closure_problem(roi)
        if roi.auto_surface == roi_pb2.AUTO_SURFACE_STAR:
            return star_problem(roi)
    return None


# -- a handle drag (the ROI scheme's §4 and §5) -------------------------------


def handle_axis(roi, handle: int) -> np.ndarray:
    """The handle's live PRIMARY axis: `AXIS_RADIAL` from `extent_anchor` to
    the handle, else the frame's first column (row-major, so elements 0, 3,
    6)."""
    spec = roi.vertices[handle].handle
    if spec.axis_source == _AXIS_RADIAL:
        radial = _vec(roi.vertices[handle].position) - _vec(roi.vertices[spec.extent_anchor].position)
        norm = float(np.linalg.norm(radial))
        if norm > 1e-12:
            return radial / norm
    frame = np.asarray(spec.frame, dtype=np.float64).reshape(3, 3)
    return frame[:, 0]


def project_onto_dof(mode: int, p0, axis, p1) -> np.ndarray:
    """The host's half (the ROI scheme's §3): the pointer onto the handle's
    DoF -- PRIMARY onto the axis line through ``p0``, SECONDARY onto the
    plane through ``p0`` orthogonal to it, ALL unchanged."""
    p0, p1, axis = (np.asarray(v, dtype=np.float64) for v in (p0, p1, axis))
    u = axis / np.linalg.norm(axis)
    if mode == _PRIMARY:
        return p0 + np.dot(p1 - p0, u) * u
    if mode == _SECONDARY:
        offset = p1 - p0
        return p0 + (offset - np.dot(offset, u) * u)
    return p1


def constrain(roi, handle: int, p1) -> tuple[np.ndarray, int]:
    """§4's order for a DECLARED region: project, then clamp a PRIMARY
    handle's signed extent from `extent_anchor` into its bounds.  Returns
    ``(p_actual, resolution)``, `CLAMPED` when the clamp bit."""
    spec = roi.vertices[handle].handle
    p0 = _vec(roi.vertices[handle].position)
    axis = handle_axis(roi, handle)
    p = project_onto_dof(spec.mode, p0, axis, p1)
    if spec.mode == _PRIMARY and (spec.HasField("min_extent") or spec.HasField("max_extent")):
        u = axis / np.linalg.norm(axis)
        anchor = _vec(roi.vertices[spec.extent_anchor].position)
        t = float(np.dot(p - anchor, u))
        lo = spec.min_extent if spec.HasField("min_extent") else -np.inf
        hi = spec.max_extent if spec.HasField("max_extent") else np.inf
        clamped = min(hi, max(lo, t))
        if clamped != t:
            return p + (clamped - t) * u, _CLAMPED
    return p, _OK


def propagate(roi, handle: int, delta) -> None:
    """§5, one level: the handle moves by ``delta`` and every vertex bound to
    it (`target == handle`) by ``M·delta``; nothing further.  Mutates
    ``roi``."""
    delta = np.asarray(delta, dtype=np.float64)
    deltas = {handle: delta}
    for b in roi.bindings:
        if b.target == handle:
            m = np.asarray(b.m, dtype=np.float64).reshape(3, 3)
            deltas[b.moved] = deltas.get(b.moved, np.zeros(3)) + m @ delta
    pts = positions(roi)
    for i, d in deltas.items():
        pts[i] = pts[i] + d
    set_positions(roi, pts)


def drag_declared(roi, handle: int, p1) -> tuple[object, int]:
    """A DECLARED region's drag of ``handle`` toward ``p1``: a moved copy and
    the resolution."""
    out = type(roi)()
    out.CopyFrom(roi)
    p_actual, resolution = constrain(out, handle, p1)
    propagate(out, handle, p_actual - _vec(out.vertices[handle].position))
    return out, resolution


def surface_problem(roi) -> str | None:
    """Why the ROI scheme refuses ``roi``'s enclosing surface (§8.2), or
    ``None`` -- the rule a declared drag stops at: rule 0 in every mode; then, for `ENCLOSED` only,
    closure when an enclosing facet or patch is declared, else STAR's rules
    under `AUTO_SURFACE_STAR`, else at least 4 enclosing points."""
    why = rule0_problem(roi)
    if why is not None or roi.inclusion.mode != roi_pb2.INCLUDE_ENCLOSED:
        return why
    if _enclosing_surface_declared(roi):
        return closure_problem(roi)
    if roi.auto_surface == roi_pb2.AUTO_SURFACE_STAR:
        return star_problem(roi)
    if len(enclosing_vertices(roi)) < 4:
        return "ENCLOSED needs a surface, and the region declares no enclosing facet or patch and fewer than 4 enclosing points"
    return None


def shape_result_problem(roi, result) -> str | None:
    """Why the host does not apply a `ShapeResult` that is not `REJECTED`
    (§8.8): not one position per vertex, a non-finite component,
    or positions that leave the region no inside (:func:`surface_problem`
    on a copy of ``roi`` -- the drag's rest form -- moved to them).
    `contribution_refusal`'s inclusion-mode and type checks are about
    contributing, not reshaping, and are not applied."""
    if len(result.positions) != len(roi.vertices):
        return f"{len(result.positions)} positions for a region of {len(roi.vertices)} vertices"
    for i, p in enumerate(result.positions):
        if not all(np.isfinite((p.z, p.y, p.x))):
            return f"position {i} ({p.z}, {p.y}, {p.x}) is not finite"
    shaped = type(roi)()
    shaped.CopyFrom(roi)
    set_positions(shaped, [(p.z, p.y, p.x) for p in result.positions])
    why = surface_problem(shaped)
    return None if why is None else f"the region would have no inside: {why}"

