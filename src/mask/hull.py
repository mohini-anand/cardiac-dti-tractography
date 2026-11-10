"""Smooth 3-D LV hull and the RV hull: remove tissue that protrudes into a ventricle's blood pool but is fused to the wall.

The LV hull is built from the dark points of the LV cavity (the label plus dark peeled voxels): per short-axis slab a
rounded, smoothed outline, smoothed again along the long axis into one surface, pushed out over any dark point left
outside and made convex. Tissue inside the surface is removed, except a protected band near the RV cavity and the
exterior over the middle of the long axis (middle_half_guard).

The RV hull removes tissue inside a 3-D closing of the RV blood pool, after an opening and a size floor. It reads no
image; only the LV hull's dark points use T1 (support_set).
"""
import math
from types import SimpleNamespace

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull, Delaunay
from skimage.measure import points_in_poly

from .exclusion import EXTERIOR, LV, RV, S26, base_side, basal_slab, finalize, voxel_sizes

N_PHI = 360
N_RESAMPLE = 512
PHI = np.linspace(-np.pi, np.pi, N_PHI, endpoint=False)
E = np.c_[np.cos(PHI), np.sin(PHI)]
SLAB_MM = 1.0
RV_HULL = SimpleNamespace(close_mm=6.0, open_mm=1.5, min_mm3=50.0)
MIN_CAVITY_POINTS = 4
BOUNDARY_RADIUS = 1e-6     # a point this close to the outline counts as inside
PUSH_TOL = 1e-3            # extra push beyond a dark point, mm
S_QUANT = 0.05             # position along the axis is quantised for the surface test, in slabs
MAX_PUSH_ITER = 60


def rounded_voxel_sizes(affine):
    """Voxel sizes rounded to 6 decimals, so a mm^3 floor does not gain a voxel from 0.99999998 mm spacing: the RV hull's
    50 mm^3 floor needs 50 voxels (51 unrounded). Exact output: exclusion.voxel_sizes does not round; keep the two apart."""
    return np.round(voxel_sizes(affine), 6)


def ball(radius_mm, zooms):
    z = np.asarray(zooms, float)
    n = np.ceil(radius_mm / z - 1e-9).astype(int)
    grid = np.mgrid[-n[0]:n[0] + 1, -n[1]:n[1] + 1, -n[2]:n[2] + 1].astype(float)
    grid *= z[:, None, None, None]
    return (grid ** 2).sum(0) <= radius_mm * radius_mm + 1e-6


def min_voxels(mm3, zooms):
    return int(math.ceil(mm3 / float(np.prod(zooms)) - 1e-6))


def opened_core(candidate, radius_mm, min_mm3, zooms):
    """Opening with a ball, then 6-connected components of at least min_mm3."""
    opened = ndi.binary_opening(candidate, structure=ball(radius_mm, zooms))
    components, _ = ndi.label(opened)
    sizes = np.bincount(components.ravel())[1:]
    return np.isin(components, np.where(sizes >= min_voxels(min_mm3, zooms))[0] + 1)


def slab_frame(shape, affine, axis):
    """Slab index along the axis and in-plane coordinates, per voxel."""
    long_axis = np.asarray(axis, float)
    norm = float(np.linalg.norm(long_axis))
    if abs(norm - 1.0) > 1e-12:    # Exact output: normalised only if needed; re-normalising changes the last bits of the axis
        long_axis = long_axis / norm
    e = np.array([1.0, 0.0, 0.0]) if abs(long_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(long_axis, e)
    u /= np.linalg.norm(u)
    v = np.cross(long_axis, u)
    index = np.indices(shape).reshape(3, -1).T
    world = index @ np.asarray(affine)[:3, :3].T + np.asarray(affine)[:3, 3]
    return np.floor((world @ long_axis) / SLAB_MM).astype(int), world @ u, world @ v


def blood_pool(cavity, label, radius_mm, zooms):
    """The cavity class opened with a ball, keeping parts that touch the cavity label; a radius of 0 leaves it unchanged."""
    if radius_mm <= 0:
        return cavity
    opened = ndi.binary_opening(cavity, structure=ball(radius_mm, zooms))
    components, _ = ndi.label(opened, structure=S26)
    keep = np.unique(components[opened & label])
    return np.isin(components, keep[keep > 0])


def rv_hull_core(rv_pool, tissue, zooms):
    """Tissue inside the 3-D closing of the RV pool, opened, small pieces dropped."""
    structure = ball(RV_HULL.close_mm, zooms)
    pad = [int(np.ceil(s / 2)) + 1 for s in structure.shape]
    closed = ndi.binary_closing(np.pad(rv_pool, [(p, p) for p in pad]), structure=structure)[pad[0]:-pad[0], pad[1]:-pad[1], pad[2]:-pad[2]]
    return opened_core(closed & tissue, RV_HULL.open_mm, RV_HULL.min_mm3, zooms)


def shoelace(points):
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _resample_closed(points, n):
    closed = np.vstack([points, points[:1]])
    segment = np.hypot(*np.diff(closed, axis=0).T)
    s = np.r_[0, np.cumsum(segment)]
    t = np.linspace(0, s[-1], n, endpoint=False)
    return np.c_[np.interp(t, s, closed[:, 0]), np.interp(t, s, closed[:, 1])], float(s[-1])


def smooth_outline(boundary, area_target, sigma):
    """Convex hull of a closed polyline, Gaussian-smoothed along its arc length, rescaled to the target area."""
    hull = ConvexHull(boundary)
    resampled, perimeter = _resample_closed(boundary[hull.vertices], N_RESAMPLE)
    k = np.fft.fftfreq(N_RESAMPLE, d=1.0 / N_RESAMPLE)
    complex_curve = np.fft.ifft(np.fft.fft(resampled[:, 0] + 1j * resampled[:, 1]) * np.exp(-0.5 * (2 * np.pi * k * sigma / perimeter) ** 2))
    smoothed = np.c_[complex_curve.real, complex_curve.imag]
    centre = smoothed.mean(0)
    return centre + (smoothed - centre) * np.sqrt(area_target / shoelace(smoothed))


def _edge_arcs(vertices, area, cap):
    """One circular arc per polygon edge, from the radius of the equal-area circle, bulge capped at `cap`; a cap of 0 gives none."""
    radius = float(np.sqrt(area / np.pi))
    arcs = []
    for i in range(len(vertices)):
        a, b = vertices[i], vertices[(i + 1) % len(vertices)]
        d = b - a
        chord = float(np.hypot(*d))
        if chord == 0 or cap == 0:
            continue
        outward = np.array([d[1], -d[0]]) / chord
        if chord < 2 * radius:
            r = radius
            bulge = r - np.sqrt(r * r - chord * chord / 4)
        else:
            r, bulge = None, np.inf
        if cap is not None and bulge > cap:
            bulge = cap
            r = (cap * cap + chord * chord / 4) / (2 * cap)
        if r is None:
            r, bulge = chord / 2, chord / 2
        middle = (a + b) / 2
        arcs.append((a, b, middle - outward * (r - bulge), r, bulge, outward, middle))
    return arcs


def _arc_points(arc, n=40):
    """Points along an arc, taking the sweep that bulges outward (the shorter one if both do)."""
    a, b, centre, r, _, outward, middle = arc
    ta, tb = np.arctan2(*(a - centre)[::-1]), np.arctan2(*(b - centre)[::-1])
    best = None
    for dt in ((tb - ta) % (2 * np.pi), -((ta - tb) % (2 * np.pi))):
        t = ta + np.linspace(0, dt, n)
        points = centre + r * np.c_[np.cos(t), np.sin(t)]
        if (points[n // 2] - middle) @ outward > 0 and (best is None or abs(dt) < abs(best[0])):
            best = (dt, points)
    return best[1]


def slab_outlines(pool, frame, params):
    """Per slab with enough support points: the rounded outline smoothed at its own area."""
    bins, pu, pv = frame
    flat = pool.ravel()
    outlines = {}
    for b in np.unique(bins[flat]):
        current = flat & (bins == b)
        if current.sum() < MIN_CAVITY_POINTS:
            continue
        points = np.c_[pu[current], pv[current]]
        try:
            # Exact output: the result is unused, but Delaunay raises on some point sets ConvexHull accepts, and such a
            # slab is skipped. No slab of this heart is skipped (70 of 70 have an outline).
            Delaunay(points)
            hull = ConvexHull(points)
        except Exception:
            continue
        vertices = points[hull.vertices]
        arcs = _edge_arcs(vertices, hull.volume, params["arc_cap_mm"])
        boundary = np.vstack([_arc_points(x) for x in arcs]) if arcs else np.vstack([vertices, vertices[:1]])
        area = abs(shoelace(boundary[:-1] if np.allclose(boundary[0], boundary[-1]) else boundary))
        outlines[int(b)] = smooth_outline(boundary, area, params["sigma_mm"])
    return outlines


def ray_radii(polygon, centre, phis=PHI):
    """Distance from a centre to a closed polygon along each direction."""
    a, b = polygon, np.roll(polygon, -1, axis=0)
    d = b - a
    e = np.c_[np.cos(phis), np.sin(phis)]
    denominator = e[:, None, 0] * (-d[None, :, 1]) - e[:, None, 1] * (-d[None, :, 0])
    rx, ry = a[None, :, 0] - centre[0], a[None, :, 1] - centre[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (rx * (-d[None, :, 1]) - ry * (-d[None, :, 0])) / denominator
        lam = (e[:, None, 0] * ry - e[:, None, 1] * rx) / denominator
    ok = (lam >= -1e-9) & (lam <= 1 + 1e-9) & (t > 0)
    return np.where(ok, t, np.inf).min(1)


def _wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _convexify(ring):
    """Convex hull vertices of a ring, counter-clockwise."""
    hull = ConvexHull(ring)
    vertices = ring[hull.vertices]
    return vertices if shoelace(vertices) > 0 else vertices[::-1]


def _radial(ring):
    """A ring re-sampled at the fixed directions E about its centre."""
    centre = ring.mean(0)
    return centre + ray_radii(ring, centre)[:, None] * E


def _radial_convex(ring):
    """As `_radial`, with the radii measured to the ring's convex hull."""
    centre = ring.mean(0)
    return centre + ray_radii(_convexify(ring), centre)[:, None] * E


def ring_at(rings, slabs, position):
    """The surface's ring at a position along the axis, in slab units (slab b spans [b, b + 1), its ring sits at b + 0.5)."""
    fraction = position - 0.5 - slabs[0]
    if fraction < 0:
        return rings[0]
    if fraction > len(slabs) - 1:
        return rings[-1]
    j = int(np.clip(np.floor(fraction), 0, len(slabs) - 2))
    a = float(np.clip(fraction - j, 0, 1))
    return (1 - a) * rings[j] + a * rings[j + 1]


def inside_polygon(polygon, points):
    """Point-in-polygon test that counts points within BOUNDARY_RADIUS of the outline as inside."""
    inside = points_in_poly(points, polygon)
    for shift in ((BOUNDARY_RADIUS, 0.0), (-BOUNDARY_RADIUS, 0.0), (0.0, BOUNDARY_RADIUS), (0.0, -BOUNDARY_RADIUS)):
        inside |= points_in_poly(points + np.array(shift), polygon)
    return inside


def distance_to_outline(polygon, point):
    a, b = polygon, np.roll(polygon, -1, axis=0)
    e = b - a
    t = np.clip(((point - a) * e).sum(1) / np.maximum((e ** 2).sum(1), 1e-12), 0, 1)
    return float(np.hypot(*(a + t[:, None] * e - point).T).min())


def outside_surface(rings, slabs, positions, points, return_distance=False):
    """Are points outside the surface? Positions beyond the end rings by more than half a slab are outside."""
    q = np.round(positions / S_QUANT).astype(int)
    outside = np.zeros(len(points), bool)
    distance = np.zeros(len(points))
    for value in np.unique(q):
        selected = np.flatnonzero(q == value)
        position = value * S_QUANT
        if position < slabs[0] - 1e-6 or position > slabs[-1] + 1 + 1e-6:
            outside[selected] = True
            continue
        ring = ring_at(rings, slabs, position)
        inside = inside_polygon(np.vstack([ring, ring[:1]]), points[selected])
        outside[selected] = ~inside
        if return_distance and (~inside).any():
            for i in selected[~inside]:
                distance[i] = distance_to_outline(ring, points[i])
    return (outside, distance) if return_distance else outside


def build_surface(outlines, positions, points, params):
    """Rings smoothed along the axis, pushed out over dark points left outside, then convex hull and push again.

    The two push loops differ. The first adds each pass's outside distance to a running deficit per point and rebuilds
    the rings from the smoothed rings on every pass, with a long-axis weight at least 0.5 slab wide. The second pushes
    the convex rings themselves by each pass's distance, with the plain long-axis width, and makes them convex again.

    Latent: a loop that has not brought every dark point inside after MAX_PUSH_ITER passes stops without a message,
    leaving those points outside the surface. On this heart the two loops need 7 and 8 passes.
    """
    sigma_long, sigma_mm = params["sigma_long_slabs"], params["sigma_mm"]
    slabs = np.array(sorted(outlines))
    rings = np.stack([outlines[b].mean(0) + ray_radii(outlines[b], outlines[b].mean(0))[:, None] * E for b in slabs])
    rings = ndi.gaussian_filter1d(rings, sigma_long, axis=0, mode="nearest")
    covered = (positions >= slabs[0]) & (positions < slabs[-1] + 1)
    positions, points = positions[covered], points[covered]

    deficit = np.zeros(len(points))
    base_rings = rings.copy()
    centres = base_rings.mean(1)
    for _ in range(MAX_PUSH_ITER):
        missing, distance = outside_surface(rings, slabs, positions, points, return_distance=True)
        if not missing.any():
            break
        deficit[missing] += distance[missing] + PUSH_TOL
        bump = np.zeros(base_rings.shape[:2])
        for i in np.flatnonzero(deficit > 0):
            fraction = positions[i] - 0.5 - slabs[0]
            low = int(max(0, np.floor(fraction - 4 * sigma_long)))
            high = int(min(len(slabs) - 1, np.ceil(fraction + 4 * sigma_long)))
            for j in range(low, high + 1):
                weight = np.exp(-0.5 * ((j - fraction) / max(sigma_long, 0.5)) ** 2)
                angle = np.arctan2(*(points[i] - centres[j])[::-1])
                arc = np.abs(_wrap(PHI - angle)) * np.hypot(*(base_rings[j] - centres[j]).T)
                bump[j] = np.maximum(bump[j], deficit[i] * weight * np.exp(-0.5 * (arc / sigma_mm) ** 2))
        rings = centres[:, None, :] + (base_rings - centres[:, None, :]) * (
            1 + bump / np.maximum(np.hypot(*(base_rings - centres[:, None, :]).transpose(2, 0, 1)), 1e-9))[..., None]

    convex = []
    for ring in rings:
        hull = _convexify(ring)
        convex.append(_radial_convex(_radial(smooth_outline(np.vstack([hull, hull[:1]]), abs(shoelace(hull)), sigma_mm))))
    convex = np.stack(convex)
    for _ in range(MAX_PUSH_ITER):
        missing, distance = outside_surface(convex, slabs, positions, points, return_distance=True)
        if not missing.any():
            break
        centres = convex.mean(1)
        bump = np.zeros(convex.shape[:2])
        for i in np.flatnonzero(missing):
            fraction = positions[i] - 0.5 - slabs[0]
            low = int(max(0, np.floor(fraction - 4 * sigma_long)))
            high = int(min(len(slabs) - 1, np.ceil(fraction + 4 * sigma_long)))
            for j in range(low, high + 1):
                weight = np.exp(-0.5 * ((j - fraction) / sigma_long) ** 2)
                angle = np.arctan2(*(points[i] - centres[j])[::-1])
                arc = np.abs(_wrap(PHI - angle)) * np.hypot(*(convex[j] - centres[j]).T)
                bump[j] = np.maximum(bump[j], (distance[i] + PUSH_TOL) * weight * np.exp(-0.5 * (arc / sigma_mm) ** 2))
        for j in np.flatnonzero(bump.max(1) > 0):
            r = np.hypot(*(convex[j] - centres[j]).T)
            convex[j] = _radial_convex(centres[j] + (convex[j] - centres[j]) * ((r + bump[j]) / np.maximum(r, 1e-9))[:, None])
    return convex, slabs


def support_set(res, t1, zooms, params):
    """The dark points: the LV label and dark peeled voxels near it or near an LV enclosed structure."""
    lv_label, rv_label, tissue, classes = res["lv_label"], res["rv_label"], res["tissue"], res["classes"]
    pre_peel = res["pre_peel"]
    peeled = pre_peel & ~tissue & ~lv_label & ~rv_label
    dark = t1 <= params["dark_t1_max"]
    d_lv = ndi.distance_transform_edt(~lv_label, sampling=zooms)
    d_lv_enclosed = ndi.distance_transform_edt(~(lv_label | res["enclosed_lv"]), sampling=zooms)
    pool = (lv_label & ~tissue) | (peeled & dark & (d_lv <= params["support_radius_mm"]))
    return pool | (peeled & dark & (d_lv_enclosed <= params["support_radius_mm"]) & (classes == LV))


def middle_half_guard(res, affine, zooms, params):
    """Protected band, not the ring guard of chain.py: tissue within guard_mm of the RV cavity class or the exterior,
    for long-axis positions between guard_s[0] and guard_s[1] of the tissue's extent, is never removed by the LV hull. It
    keeps the wall there, so the hull opens no new in-plane path to the outside. The width and the limits were set for
    this heart."""
    tissue, classes = res["tissue"], res["classes"]
    solid = tissue | np.isin(classes, (LV, RV))
    slab = basal_slab(tissue.shape, solid)
    near = ((classes == RV) | (classes == EXTERIOR)) & ~slab
    k = np.arange(tissue.shape[2])
    occupied = np.nonzero(res["pre_peel"].any(axis=(0, 1)))[0]
    beyond = k > occupied[-1] if base_side(solid) > 0 else k < occupied[0]
    near &= ~np.broadcast_to(beyond.reshape(1, 1, -1), tissue.shape)
    zone = tissue & (ndi.distance_transform_edt(~near, sampling=zooms) < params["guard_mm"])
    # Exact output: normalised here on its own, separately from the conditional normalisation in slab_frame.
    axis = np.asarray(res["axis"], float)
    axis = axis / np.linalg.norm(axis)
    index = np.indices(tissue.shape).reshape(3, -1).T
    projection = ((index @ np.asarray(affine)[:3, :3].T + np.asarray(affine)[:3, 3]) @ axis).reshape(tissue.shape)
    s = (projection - projection[tissue].min()) / (projection[tissue].max() - projection[tissue].min())
    return zone & (s >= params["guard_s"][0]) & (s <= params["guard_s"][1])


def slab_positions(shape, affine, axis):
    """Position of every voxel along the axis, in slab units."""
    # Exact output: a separate normalisation again, as in middle_half_guard.
    long_axis = np.asarray(axis, float)
    long_axis = long_axis / np.linalg.norm(long_axis)
    index = np.indices(shape).reshape(3, -1).T
    return (index @ np.asarray(affine)[:3, :3].T + np.asarray(affine)[:3, 3]) @ long_axis / SLAB_MM


def smooth_lv_hull(res, t1, affine, rv_core, frame, params):
    """LV removal: voxels inside the surface, minus the guard; a removal that would leave a hole is put back."""
    zooms = voxel_sizes(affine)
    tissue = res["tissue"]
    pool = support_set(res, t1, zooms, params)
    _, pu, pv = frame
    outlines = slab_outlines(pool, frame, params)
    positions = slab_positions(tissue.shape, affine, res["axis"])
    flat_pool = pool.ravel()
    rings, slabs = build_surface(outlines, positions[flat_pool], np.c_[pu[flat_pool], pv[flat_pool]], params)
    candidates = np.flatnonzero(tissue.ravel() & (positions >= slabs[0] - 0.5) & (positions <= slabs[-1] + 1.5))
    inside = np.zeros(tissue.size, bool)
    inside[candidates[~outside_surface(rings, slabs, positions[candidates], np.c_[pu[candidates], pv[candidates]])]] = True
    lv = inside.reshape(tissue.shape) & ~middle_half_guard(res, affine, zooms, params)
    enclosed = res["enclosed_lv"] | res["enclosed_rv"]
    existing_holes = ndi.binary_fill_holes(tissue) & ~tissue
    rv = rv_core & ~lv
    for _ in range(5):
        rv = rv_core & ~lv
        kept = finalize(tissue, enclosed | lv | rv)
        back = ndi.binary_fill_holes(kept) & ~kept & ~existing_holes & lv
        if not back.any():
            break
        lv = lv & ~back
    return lv, rv_core & ~lv


def add_hull(res, t1, affine, params):
    """Final mask after two-sided exclusion plus the LV and RV hulls; largest component."""
    tissue, classes = res["tissue"], res["classes"]
    rounded = rounded_voxel_sizes(affine)
    rv_pool = blood_pool(classes == RV, res["rv_label"], params["cavity_open_mm"], rounded)
    rv_core = rv_hull_core(rv_pool, tissue, rounded)
    frame = slab_frame(tissue.shape, affine, res["axis"])
    lv, rv = smooth_lv_hull(res, t1, affine, rv_core, frame, params)
    return finalize(tissue, res["enclosed_lv"] | res["enclosed_rv"] | lv | rv)
