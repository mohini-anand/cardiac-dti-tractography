"""RV trabecular and papillary exclusion.

Tissue attached to an enclosed RV structure (an anchor) is removed by two tests. The envelope test removes tissue that no
ball of radius R_mm fitting outside the RV cavity and its cleft reaches, that is, tissue protruding into the cavity. The
free-wall cut removes free-wall tissue deeper than a polyline in the wall chart (in-plane depth from the epicardium
against arc position along it); the polyline joins the cleft points from the last one before an anchor's attachment to
the first one after it. Vetoes then put voxels back, keeping the outer wall, the bridges and the ring intact.

Terms:
- Anchor: an enclosed RV structure from exclusion.py that is resolved and whose contact is mostly free wall (septal share
  under SEPTAL_CONTACT_MAX). Resolved: at least anchor_min_mm3 of tissue, part of which survives an opening with a ball of
  radius rs_mm (resolved_core). A resolved body is envelope residue that survives the same opening together with the
  anchors, lies within R_mm of an anchor and touches one.
- Cleft: T2-bright or void tissue sub-voxels reached from the RV cavity through non-muscle tissue; the envelope treats it
  as cavity.
- Cleft points: the vertices of the lower convex hull, in the wall chart, of the dark voxels facing the free wall near an
  anchor.
- Compact wall: the outer layer of the RV free wall. Its depth is the median in-plane depth from the epicardium of the
  free-wall voxels lining the RV cavity (compact_depth).
- RV insertion points: the two ends of a slice's RV cavity crescent (horn_tips), where the free wall joins the septum.
  They split the cavity contour into its septal and free-wall sides.
- Static vetoes are applied once; iterative vetoes repeat for up to max_rounds (see remove).
"""
from types import SimpleNamespace

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import measure

from .chain import open_slices
from .exclusion import ENCLOSED, LV, RV, classify_nontissue, geodesic_flood
from .images import up

S4 = ndi.generate_binary_structure(2, 1)
S8 = np.ones((3, 3), bool)
S6 = ndi.generate_binary_structure(3, 1)
S26 = np.ones((3, 3, 3), bool)
SUB_MM = 0.5
CLAMP = SimpleNamespace(deep_mm=3.0, t2_pct=95.0, t1_pct=20.0, void_pct=1.0)
CONTOUR_STEP_MM = 0.25
SEPTAL_CONTACT_MAX = 0.5


def down_count(mask):
    """Number of True sub-voxels (0 to 8) in each 1 mm voxel."""
    mask = np.asarray(mask, np.int16)
    return sum(mask[a::2, b::2, c::2] for a in (0, 1) for b in (0, 1) for c in (0, 1))


def floor_distance(reference):
    """Per slice: in-plane distance to the border-connected 4-connected complement of `reference`."""
    out = np.zeros(reference.shape, np.float32)
    for k in range(reference.shape[2]):
        r = reference[:, :, k]
        if not r.any():
            continue
        labels, _ = ndi.label(~r, structure=S4)
        border = np.zeros_like(r)
        border[[0, -1], :] = True
        border[:, [0, -1]] = True
        ids = np.unique(labels[border & ~r])
        ids = ids[ids > 0]
        out[:, :, k] = ndi.distance_transform_edt(~np.isin(labels, ids))
    return out


def floor_allowed(distance, tissue, floor_mm):
    """Removable under the floor: the voxel and every tissue voxel 8-adjacent in-plane are at least floor_mm deep."""
    thin = tissue & (distance < floor_mm)
    bad = np.zeros_like(thin)
    for k in range(thin.shape[2]):
        if thin[:, :, k].any():
            bad[:, :, k] = ndi.binary_dilation(thin[:, :, k], structure=S8)
    return ~(bad | (distance < floor_mm))


def clamp_thresholds(tissue, t1, t2, tau, tau2, void_t1, void_t2):
    """Thresholds clamped against the deep wall, so that the cleft tests can never call more than 5% of the deep wall
    T2-bright, 20% of it non-muscle by T1, or 1% of it void: each threshold moves, if needed, to the 95th, 20th or 1st
    percentile of the deep wall (sub-voxels at least CLAMP.deep_mm inside the mask, or its deepest quarter when fewer
    than 100 are that deep). The percentiles were set during development; on this heart the clamp changes no threshold.
    """
    fine = up(tissue)
    depth = ndi.distance_transform_edt(fine, sampling=SUB_MM)
    deep = fine & (depth >= CLAMP.deep_mm)
    if deep.sum() < 100:
        deep = fine & (depth >= np.percentile(depth[fine], 75))
    t1_deep, t2_deep = t1[deep], t2[deep]
    return dict(tau2=max(tau2, float(np.percentile(t2_deep, CLAMP.t2_pct))),
                tau=min(tau, float(np.percentile(t1_deep, CLAMP.t1_pct))),
                void_t1=min(void_t1, float(np.percentile(t1_deep, CLAMP.void_pct))),
                void_t2=min(void_t2, float(np.percentile(t2_deep, CLAMP.void_pct))))


def ball_opening(mask, radius, zooms):
    """Opening by a ball of the given radius, computed from two distance transforms."""
    distance = ndi.distance_transform_edt(mask, sampling=zooms)
    centres = distance > radius
    if not centres.any():
        return np.zeros_like(mask)
    return ndi.distance_transform_edt(~centres, sampling=zooms) <= radius


def _bbox(mask, pad, shape):
    index = np.argwhere(mask)
    low = np.maximum(index.min(0) - pad, 0)
    high = np.minimum(index.max(0) + pad + 1, shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(low, high))


def resolved_core(mask, radius_mm, zooms):
    """1 mm voxels at least half (4 of 8) of whose 0.5 mm sub-voxels lie in the opening of the mask by a ball of the given
    radius."""
    out = np.zeros(mask.shape, bool)
    if not mask.any():
        return out
    box = _bbox(mask, 3, mask.shape)
    fine = up(mask[box])
    out[box] = down_count(ball_opening(fine, radius_mm, np.asarray(zooms) / 2.0) & fine) >= 4
    return out & mask


def rv_cleft(tissue, rv_cavity, lv_cavity, t1, t2, enclosed, scales, void_t1, void_t2):
    """Tissue sub-voxels that are T2-bright or void and connected to the RV cavity through non-muscle tissue."""
    t1, t2 = np.asarray(t1, np.float32), np.asarray(t2, np.float32)
    thresholds = clamp_thresholds(tissue, t1, t2, scales["tau"], scales["tau2"], void_t1, void_t2)
    void = (t1 < thresholds["void_t1"]) & (t2 < thresholds["void_t2"])
    bright = (t2 > thresholds["tau2"]) & ~void
    muscle = (t1 >= thresholds["tau"]) & (t2 <= thresholds["tau2"]) & ~void
    fine = up(tissue)
    seeds = np.zeros(fine.shape, np.int8)      # flood labels: 1 exterior, 2 LV cavity, 3 RV cavity
    seeds[~fine & ~up(rv_cavity) & ~up(lv_cavity) & ~up(np.asarray(enclosed, bool))] = 1
    seeds[up(lv_cavity)] = 2
    seeds[up(rv_cavity)] = 3
    flooded = geodesic_flood(seeds, fine & ~muscle, connectivity=S26)
    return fine & (bright | void) & (flooded == 3)


def envelope_residue(tissue, rv_cavity, cleft, radius, zooms):
    """Tissue beyond the envelope of balls that fit outside the RV cavity class and its cleft (0.5 mm grid)."""
    cavity_fine = up(rv_cavity) | cleft
    half = np.asarray(zooms) / 2.0
    pad = int(np.ceil(2 * radius / float(np.min(half)))) + 2
    box = _bbox(cavity_fine, pad, cavity_fine.shape)
    opened = ball_opening(~cavity_fine[box], radius, half)
    fine = up(tissue)[box]
    not_cleft = fine & ~cleft[box]
    residue = np.zeros(cavity_fine.shape, bool)
    within_envelope = np.zeros(cavity_fine.shape, bool)
    residue[box] = not_cleft & ~opened
    within_envelope[box] = not_cleft & opened
    return (down_count(residue) > down_count(within_envelope)) & tissue


def new_holes(mask, tissue):
    """Holes in `mask`, under 6- or 26-connectivity, that `tissue` does not already have."""
    before6 = ndi.binary_fill_holes(tissue) & ~tissue
    before26 = ndi.binary_fill_holes(tissue, structure=S26) & ~tissue
    holes6 = ndi.binary_fill_holes(mask) & ~mask & ~before6
    holes26 = ndi.binary_fill_holes(mask, structure=S26) & ~mask & ~before26
    return holes6 | holes26


def envelope_candidates(tissue, rv_cavity, lv_cavity, anchors, zooms, t1, t2, enclosed, scales, params):
    """Tissue that belongs to a resolved body and is attached to a resolved enclosed structure."""
    voxel_mm3 = float(np.prod(zooms))
    radius = params["R_mm"]
    cleft = rv_cleft(tissue, rv_cavity, lv_cavity, t1, t2, enclosed, scales, params["cleft_void_t1"], params["cleft_void_t2"])
    residue = envelope_residue(tissue, rv_cavity, cleft, radius, zooms)

    labels_a, n_anchors = ndi.label(anchors, S26)
    sizes = np.bincount(labels_a.ravel())
    sizes[0] = 0
    core = resolved_core(anchors, params["rs_mm"], zooms)
    keep = [i for i in range(1, n_anchors + 1) if sizes[i] * voxel_mm3 >= params["anchor_min_mm3"] - 1e-9 and core[labels_a == i].any()]
    resolved_anchor = np.isin(labels_a, keep)
    if not resolved_anchor.any() or not residue.any():
        return np.zeros(tissue.shape, bool), resolved_anchor

    body = resolved_core(residue | resolved_anchor, params["rs_mm"], zooms) & residue
    body &= ndi.distance_transform_edt(~resolved_anchor, sampling=zooms) <= radius
    labels_b, _ = ndi.label(body, S6)
    touching = np.unique(labels_b[ndi.binary_dilation(resolved_anchor, S6) & body])
    touching = touching[touching > 0]
    return np.isin(labels_b, touching), resolved_anchor


def exterior2d(mask):
    """Complement voxels 4-connected to the slice border."""
    labels, _ = ndi.label(~mask, structure=S4)
    border = np.zeros_like(mask)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    ids = np.unique(labels[border & ~mask])
    return np.isin(labels, ids[ids > 0])


class Chart:
    """Per-slice wall chart: d1 is the in-plane distance from the exterior (depth from the epicardium), and the arc
    position runs along the smoothed epicardial contour. It starts at the contour point farthest from the RV cavity
    centre, opposite the RV, so that the RV free wall is one unbroken stretch of arc position."""

    def __init__(self, mask, rv_cavity, contour_sigma_mm):
        self.mask = mask
        self.exterior = exterior2d(mask)
        self.d1 = ndi.distance_transform_edt(~self.exterior).astype(np.float32)
        contour = max(measure.find_contours((~self.exterior).astype(float), 0.5), key=len)
        if np.allclose(contour[0], contour[-1]):
            contour = contour[:-1]
        closed = np.vstack([contour, contour[:1]])
        segment = np.hypot(*np.diff(closed, axis=0).T)
        arc = np.r_[0, np.cumsum(segment)]
        n = max(int(arc[-1] / CONTOUR_STEP_MM), 16)
        t = np.linspace(0, arc[-1], n, endpoint=False)
        points = np.c_[np.interp(t, arc, closed[:, 0]), np.interp(t, arc, closed[:, 1])]
        sigma = contour_sigma_mm / (arc[-1] / n)
        points = np.c_[ndi.gaussian_filter1d(points[:, 0], sigma, mode="wrap"), ndi.gaussian_filter1d(points[:, 1], sigma, mode="wrap")]
        segment = np.hypot(*np.diff(np.vstack([points, points[:1]]), axis=0).T)
        arc = np.r_[0, np.cumsum(segment)][:-1]
        self.perimeter = float(segment.sum())
        if rv_cavity is not None and rv_cavity.any():
            centre = np.argwhere(rv_cavity).mean(0)
            origin = int(np.argmax(((points - centre) ** 2).sum(1)))
            points = np.roll(points, -origin, axis=0)
            segment = np.hypot(*np.diff(np.vstack([points, points[:1]]), axis=0).T)
            arc = np.r_[0, np.cumsum(segment)][:-1]
        self.arc = arc
        self.tree = cKDTree(points)

    def s_of(self, points):
        """Arc position of the contour point nearest to each given point."""
        _, nearest = self.tree.query(np.asarray(points, float))
        return self.arc[nearest]

    def arc_positions(self):
        """Arc position of every voxel of the slice."""
        index = np.indices(self.mask.shape).reshape(2, -1).T
        return self.s_of(index).reshape(self.mask.shape).astype(np.float32)


def horn_tips(rv_slice):
    """The two RV insertion points of a slice: end points of the approximate geodesic diameter of the largest
    8-connected RV-class component. Found separately from epi_trace.rv_tips, on this step's own class volume."""
    labels, n = ndi.label(rv_slice, S8)
    if n == 0:
        return []
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    component = labels == int(sizes.argmax())
    if component.sum() < 3:
        return []

    def farthest(source):
        distance = np.full(component.shape, -1, int)
        distance[source] = 0
        front = [source]
        max_distance, last = 0, source
        while front:
            following = []
            for (a, b) in front:
                for da in (-1, 0, 1):
                    for db in (-1, 0, 1):
                        u, v = a + da, b + db
                        if 0 <= u < component.shape[0] and 0 <= v < component.shape[1] and component[u, v] and distance[u, v] < 0:
                            distance[u, v] = distance[a, b] + 1
                            following.append((u, v))
                            if distance[u, v] > max_distance:
                                max_distance = distance[u, v]
                                last = (u, v)
            front = following
        return last

    start = tuple(int(x) for x in np.argwhere(component)[0])
    first = farthest(start)
    second = farthest(first)
    return [first, second]


def arc_sides(final_slice, rv_slice, lv_slice, tips):
    """Septal and free-wall tissue, from the RV cavity contour split at the two RV insertion points."""
    free = np.zeros(final_slice.shape, bool)
    septal = np.zeros_like(free)
    if len(tips) != 2 or not rv_slice.any():
        return free, septal
    labels, _ = ndi.label(rv_slice, S8)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    component = labels == int(sizes.argmax())
    contour = max(measure.find_contours(np.pad(component, 1).astype(float), 0.5), key=len) - 1.0
    tree = cKDTree(contour)
    i1, i2 = int(tree.query(tips[0])[1]), int(tree.query(tips[1])[1])
    a, b = sorted((i1, i2))
    arc = np.zeros(len(contour), np.int8)
    arc[a:b + 1] = 1
    d_rv = ndi.distance_transform_edt(~rv_slice)
    d_lv = ndi.distance_transform_edt(~lv_slice) if lv_slice.any() else np.full(rv_slice.shape, np.inf)
    rv_facing = final_slice & (d_rv < d_lv)
    index = np.argwhere(rv_facing)
    if not len(index):
        return free, septal
    side = arc[tree.query(index)[1]]
    layer = np.argwhere(rv_facing & ndi.binary_dilation(rv_slice, S8))
    layer_side = arc[tree.query(layer)[1]]
    medians = [float(np.median(d_lv[tuple(layer[layer_side == s].T)])) if (layer_side == s).any() else np.inf for s in (0, 1)]
    septal_arc = int(np.argmin(medians))
    septal[tuple(index[side == septal_arc].T)] = True
    free[tuple(index[side != septal_arc].T)] = True
    return free, septal


def attachment_span(anchor_slice, free_slice, arc_position, chart):
    """Arc range of free-wall tissue adjacent to an anchor in-plane, or None."""
    attachment = free_slice & ndi.binary_dilation(anchor_slice, S8)
    if not attachment.any():
        return None
    s = arc_position[attachment].astype(float)
    if s.max() - s.min() > chart.perimeter / 2:
        s = np.where(s < chart.perimeter / 2, s + chart.perimeter, s)
    return float(s.min()), float(s.max())


def in_scope_components(labels, n, final, free, septal, core, voxel_mm3, anchor_min_mm3):
    """Enclosed RV structures that anchor the removal: resolved and contacting mostly free wall."""
    ids = []
    for q in range(1, n + 1):
        member = labels == q
        ring = np.zeros(final.shape, bool)
        for k in sorted(set(np.nonzero(member)[2].tolist())):
            ring[:, :, k] = ndi.binary_dilation(member[:, :, k], S8) & final[:, :, k] & ~member[:, :, k]
        n_free, n_septal = int((ring & free).sum()), int((ring & septal).sum())
        resolved = bool(core[member].any()) and member.sum() * voxel_mm3 >= anchor_min_mm3 - 1e-9
        if resolved and n_septal / max(n_free + n_septal, 1) < SEPTAL_CONTACT_MAX:
            ids.append(q)
    return ids


def lower_hull(s, d):
    """Lower convex hull of points (s, d); returns hull indices into the sorted order and that order."""
    order = np.lexsort((d, s))
    sorted_s, sorted_d = s[order], d[order]
    hull = []
    for i in range(len(sorted_s)):
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            if (sorted_s[b] - sorted_s[a]) * (sorted_d[i] - sorted_d[a]) - (sorted_d[b] - sorted_d[a]) * (sorted_s[i] - sorted_s[a]) <= 0:
                hull.pop()
            else:
                break
        if hull and sorted_s[hull[-1]] == sorted_s[i]:
            continue
        hull.append(i)
    return np.array(hull, int), order


def support_hull(dark_slice, free_slice, chart, arc_position, d_anchor, radius_mm, s_reference):
    """Cleft points of one anchor on one slice: the vertices of the lower convex hull, in the wall chart, of the dark
    voxels facing the free wall within radius_mm of the anchor."""
    _, nearest = ndi.distance_transform_edt(~chart.mask, return_indices=True)
    candidates = dark_slice & free_slice[nearest[0], nearest[1]] & (d_anchor <= radius_mm)
    if candidates.sum() < 2:
        return None
    s = arc_position[candidates].astype(float)
    d = chart.d1[candidates].astype(float)
    perimeter = chart.perimeter
    s = (s - s_reference + perimeter / 2) % perimeter - perimeter / 2 + s_reference
    hull, order = lower_hull(s, d)
    vertices = order[hull]
    return s[vertices], d[vertices]


def compute_cues(final, enclosed_rv, lv_label, rv_label, manual, t1, zooms, params):
    """Classes, dark RV cavity, septal and free-wall sides, enclosed components and wall charts."""
    classes = classify_nontissue(final, lv_label, rv_label, pre_peel=manual)
    # Only the enclosed RV voxels the earlier steps removed from the mask; the support distance and the resolved core
    # below use these, not all of enclosed_rv.
    enclosed_removed = enclosed_rv & ~final
    rv_class = classes == RV
    rv_outside = rv_label & ~final
    d_support = ndi.distance_transform_edt(~(rv_label | enclosed_removed), sampling=zooms)
    dark = rv_outside | ((t1 <= params["dark_t1_max"]) & ~final & rv_class & (d_support <= params["dark_support_radius_mm"]))
    labels, _ = ndi.label(enclosed_rv, S26)
    core = resolved_core(enclosed_removed, params["rs_mm"], zooms)
    slices = [k for k in range(final.shape[2]) if final[:, :, k].any()]
    free, septal = np.zeros(final.shape, bool), np.zeros(final.shape, bool)
    for k in slices:
        rv_slice = rv_class[:, :, k]
        if not rv_slice.any():
            continue
        free[:, :, k], septal[:, :, k] = arc_sides(final[:, :, k], rv_slice, (classes == LV)[:, :, k], horn_tips(rv_slice))
    arc_position = np.full(final.shape, np.nan, np.float32)
    d1 = np.zeros(final.shape, np.float32)
    charts = {}
    for k in slices:
        rv_slice = rv_class[:, :, k]
        charts[k] = Chart(final[:, :, k], rv_slice if rv_slice.any() else None, params["contour_sigma_mm"])
        arc_position[:, :, k] = charts[k].arc_positions()
        d1[:, :, k] = charts[k].d1
    return dict(classes=classes, dark=dark, free=free, septal=septal, labels=labels, core=core, arc_position=arc_position, d1=d1, charts=charts)


def compact_depth(final, rv_cavity, lv_cavity, floor_reference, zooms):
    """Compact-wall depth: median in-plane depth from the epicardium of the free-wall voxels lining the RV cavity (those
    whose depth is less than their distance to the LV cavity)."""
    distance = floor_distance(floor_reference)
    d_lv = ndi.distance_transform_edt(~lv_cavity, sampling=zooms)
    face = final & ndi.binary_dilation(rv_cavity, S6) & (distance > 0) & (distance < d_lv)
    return float(np.median(distance[face]))


def free_wall_cut(cues, anchor_ids, final, bridge, zooms, support_radius_mm):
    """Candidates deeper than the wall-chart polyline through the cleft points, from the last one before each attachment
    to the first one after it."""
    shape = final.shape
    cut = np.zeros(shape, bool)
    in_span = np.zeros(shape, bool)
    line = np.full(shape, np.nan, np.float32)
    for q in anchor_ids:
        anchor = (cues["labels"] == q) & ~final
        if not anchor.any():
            continue
        d_anchor = ndi.distance_transform_edt(~anchor, sampling=zooms)
        for k in sorted(set(np.nonzero(anchor)[2].tolist())):
            chart = cues["charts"].get(k)
            free_slice = cues["free"][:, :, k]
            if chart is None or not free_slice.any():
                continue
            arc_position = cues["arc_position"][:, :, k]
            span = attachment_span(anchor[:, :, k], free_slice, arc_position, chart)
            if span is None:
                continue
            s_reference = 0.5 * (span[0] + span[1])
            hull = support_hull(cues["dark"][:, :, k], free_slice, chart, arc_position, d_anchor[:, :, k], support_radius_mm, s_reference)
            if hull is None:
                continue
            hull_s, hull_d = hull
            perimeter = chart.perimeter
            wrapped = (np.array(span) - s_reference + perimeter / 2) % perimeter - perimeter / 2 + s_reference
            left = np.flatnonzero(hull_s <= wrapped.min())
            right = np.flatnonzero(hull_s >= wrapped.max())
            if not len(left) or not len(right):
                continue
            i_left, i_right = int(left[-1]), int(right[0])
            if i_left == i_right or hull_s[i_right] - hull_s[i_left] < 1e-6:
                continue
            knots_s, knots_d = hull_s[i_left:i_right + 1], hull_d[i_left:i_right + 1]
            s_voxel = (arc_position.astype(float) - s_reference + perimeter / 2) % perimeter - perimeter / 2 + s_reference
            inside = (s_voxel >= knots_s[0]) & (s_voxel <= knots_s[-1]) & final[:, :, k]
            t = np.interp(s_voxel, knots_s, knots_d)
            cut[:, :, k] |= inside & free_slice & ~bridge[:, :, k] & (chart.d1 > t)
            in_span[:, :, k] |= inside
            slice_line = line[:, :, k]
            slice_line[inside] = np.fmin(slice_line[inside], t[inside])
    return cut, in_span, line


def remove(final, enclosed_rv, bridge, manual, ring_cavity, t1, t2, lv_label, rv_label, t1_dark, affine_zooms, scales, params):
    """Remove RV trabecular tissue from the mask after the bridges. Returns the new final mask."""
    zooms = affine_zooms
    cues = compute_cues(final, enclosed_rv, lv_label, rv_label, manual, t1_dark, zooms, params)
    classes = cues["classes"]
    rv_cavity, lv_cavity = classes == RV, classes == LV
    ids = in_scope_components(cues["labels"], int(cues["labels"].max()), final, cues["free"], cues["septal"], cues["core"],
                              float(np.prod(zooms)), params["anchor_min_mm3"])
    # Measured once, on the mask as it enters this step: recomputed on the reduced mask it would change.
    compact = compact_depth(final, rv_cavity, lv_cavity, final, zooms)

    anchors = np.isin(cues["labels"], ids) & ~final
    envelope, resolved = envelope_candidates(final, rv_cavity, lv_cavity, anchors, zooms, t1, t2, classes == ENCLOSED, scales, params)
    envelope &= final
    cut, in_span, line = free_wall_cut(cues, [i for i in ids if (resolved & (cues["labels"] == i)).any()], final, bridge, zooms,
                                       params["anchor_support_radius_mm"])
    removal = (envelope | cut) & final

    def veto(mask):
        removal[mask] = False

    # Static vetoes, applied once.
    veto(bridge)                                    # bridge voxels complete the wall
    # The wall floor is measured once, on the mask as it enters this step, like the compact depth above; it keeps at least
    # floor_mm of wall wherever the removal would thin it below that.
    distance = floor_distance(final)
    veto(~floor_allowed(distance, final, params["floor_mm"]))
    d_rv, d_lv = ndi.distance_transform_edt(~rv_cavity, sampling=zooms), ndi.distance_transform_edt(~lv_cavity, sampling=zooms)
    veto(d_lv < d_rv)                               # tissue nearer the LV cavity is left to the LV hull
    veto(cues["septal"])                            # septal tissue is left to the enclosure exclusion
    # Outside every attachment span, the compact wall is kept. This only protects: where a vessel channel inside the wall
    # reads as a recess of the cavity, the envelope can dip into the wall, and this keeps the wall there.
    veto(~in_span & (distance <= compact))
    veto(in_span & ~(cues["d1"] > line))           # inside a span, only tissue beyond the line is removed

    # Iterative vetoes: removal not connected to a resolved anchor, removal that would open a slice's in-plane ring, and
    # removal that would make a new hole are put back, until nothing changes.

    slices = [int(k) for k in np.nonzero(final.sum((0, 1)))[0]]
    open4_before, open8_before = open_slices(final, manual, ring_cavity, slices)
    for _ in range(params["max_rounds"]):
        changed = 0
        labels, _ = ndi.label(removal, S6)
        reached = np.unique(labels[ndi.binary_dilation(resolved, S6) & removal])
        reached = reached[reached > 0]
        cut_off = removal & ~np.isin(labels, reached)
        if cut_off.any():
            removal &= ~cut_off
            changed += int(cut_off.sum())
        if removal.any():
            remaining = sorted(set(np.nonzero(removal)[2].tolist()))
            open4_after, open8_after = open_slices(final & ~removal, manual, ring_cavity, remaining)
            for k in sorted((set(open4_after) - set(open4_before)) | (set(open8_after) - set(open8_before))):
                changed += int(removal[:, :, k].sum())
                removal[:, :, k] = False
        if removal.any():
            holes = new_holes(final & ~removal, final) & removal
            if holes.any():
                removal &= ~holes
                changed += int(holes.sum())
        if changed == 0:
            break

    rest = final & ~removal
    labels, n = ndi.label(rest, S26)
    detached = np.zeros(final.shape, bool)
    if n > 1:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        detached = rest & (labels != int(sizes.argmax()))
    detached &= ~bridge
    return final & ~(removal | detached)
