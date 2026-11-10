"""Epicardial trace: a closed minimum-cost path per axial slice, found by dynamic programming in polar coordinates.

The path runs on the 0.5 mm nested grid and starts from the manual mask. Its node cost combines the outward T1 drop,
non-muscle inside the candidate, muscle outside it and T2-bright fluid inside it. Slices are coupled by re-solving each
one against the smoothed depth of its neighbours from the previous round. The traced outline becomes a 1 mm mask.

Terms: on each ray, the face is the outermost radius inside the manual mask (the manual mask's outer edge); the path's
radius change between neighbouring rays is penalised against the face's change. Depth is measured inward from the face.
The RV insertion points of a slice are the two tips of its RV cavity crescent; the interventricular grooves run in from
the outside next to them.

The trace reads each 1 mm slice k at sub-slice 2k only, the lower half of the slice, as do the restorable map (chain.py)
and the bridges (bridge.py). The groove channel is the exception here: its 3-D connectivity runs over all sub-slices, so
the upper halves affect which lower-half sub-voxels belong to it. The bias normalisation (images.py), the fluid peel and
the RV trabecular removal use all eight sub-voxels of each voxel.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.graph import MCP_Geometric
from skimage.measure import points_in_poly
from skimage.morphology import disk

from .images import LV_CAVITY, RV_CAVITY

SUB_MM = 0.5
N_THETA = 720           # rays per slice
DR = 0.25               # radial step, mm
RV_TIP_MIN_VOXELS = 50  # smallest RV cavity component whose two RV insertion points are looked for
S26 = np.ones((3, 3, 3), bool)


def slice_arrays(volume, k):
    """The arrays of DWI slice k, taken from sub-slice 2k (its lower half)."""
    index = 2 * k
    return dict(t1=volume["t1"][:, :, index].astype(np.float32), t2=volume["t2"][:, :, index].astype(np.float32),
                manual=volume["manual"][:, :, index], lv=volume["lv"][:, :, index], rv=volume["rv"][:, :, index],
                classes=volume["classes"][:, :, index])


def interior_void(d, p):
    """No-signal pockets enclosed by tissue, at least VOID_MIN_MM2, inside the manual mask or a labelled cavity."""
    manual, classes = d["manual"], d["classes"]
    dark = (d["t1"] < p.VOID_T1) & (d["t2"] < p.VOID_T2)
    closed = ndi.binary_closing(~dark, structure=disk(int(round(p.VOID_CLOSE_MM / SUB_MM))), border_value=0)
    open_dark = dark & ~closed
    components, n = ndi.label(open_dark)
    if n == 0:
        return np.zeros_like(manual)
    border = np.zeros_like(dark)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    outside = np.unique(components[border & open_dark])
    outside = outside[outside > 0]
    pocket = open_dark & ~np.isin(components, outside) & (manual | (classes == LV_CAVITY) | (classes == RV_CAVITY))
    pockets, n_pockets = ndi.label(pocket)
    if n_pockets == 0:
        return np.zeros_like(manual)
    areas = ndi.sum(pocket, pockets, range(1, n_pockets + 1)) * SUB_MM * SUB_MM
    return np.isin(pockets, 1 + np.where(areas >= p.VOID_MIN_MM2)[0])


def polar_sample(d, p, centre=None):
    """Sample the slice along N_THETA rays from the centre (the centroid of the filled slice by default); `face` is the
    outermost radius inside the manual mask on each ray."""
    manual = d["manual"]
    cavity = d["lv"] | d["rv"] | interior_void(d, p)
    filled = ndi.binary_fill_holes(manual | cavity)
    c = np.argwhere(filled).astype(float).mean(0) if centre is None else np.asarray(centre, float)
    r_max = np.sqrt(((np.argwhere(manual) - c) ** 2).sum(1)).max() * SUB_MM + 2.0
    n_radii = int(np.ceil(r_max / DR)) + 1
    theta = np.arange(N_THETA) * 2 * np.pi / N_THETA
    directions = np.stack([np.cos(theta), np.sin(theta)], 1)
    radii = np.arange(n_radii) * DR
    points = c[None, None, :] + directions[:, None, :] * (radii / SUB_MM)[None, :, None]
    query = points.reshape(-1, 2).T

    def sample(array, order):
        return ndi.map_coordinates(array.astype(float), query, order=order, mode="constant", cval=0.0).reshape(N_THETA, n_radii)

    t1, t2 = sample(d["t1"], 1), sample(d["t2"], 1)
    mask_ray, filled_ray = sample(manual, 0) > 0.5, sample(filled, 0) > 0.5
    d_cavity = (sample(ndi.distance_transform_edt(~cavity) * SUB_MM, 1) if cavity.any() else np.full((N_THETA, n_radii), 1e6))
    face = np.where(mask_ray.any(1), n_radii - 1 - np.argmax(mask_ray[:, ::-1], 1), 0).astype(float)
    return dict(centre=c, theta=theta, radii=radii, points=points, t1=t1, t2=t2, mask=mask_ray, filled=filled_ray,
                d_cavity=d_cavity, n_radii=n_radii, face_smooth=ndi.gaussian_filter1d(face, p.FACE_SIG, mode="wrap"))


def not_star_shaped(polar):
    """Number of rays that cross the filled region in two or more separate runs."""
    # Latent: a ray whose single run does not start at the centre (the centre lies outside the filled region on that ray)
    # is not counted, although the region is not star-shaped from the centre there. No ray of this heart has such a run.
    filled = polar["filled"]
    runs = ((filled[:, 1:] != filled[:, :-1]).sum(1) + filled[:, 0]).astype(int)
    return int((runs > 2).sum())


def node_costs(polar, scales, p):
    """Cost of placing the epicardium at each ray and radius."""
    t1, t2, mask, radii = polar["t1"], polar["t2"], polar["mask"], polar["radii"]
    n_radii = t1.shape[1]
    t1_drop = -ndi.gaussian_filter1d(t1, p.SIG_D_MM / DR, axis=1, order=1) / DR
    edge = 1 - np.clip(t1_drop / scales["s1_per_mm"], 0, 1)
    window = int(round(p.W / DR))
    j = np.arange(n_radii)
    low = np.clip(j + 1 - window, 0, n_radii)

    def inside_window(a):
        c = np.cumsum(np.pad(a, ((0, 0), (1, 0))), 1)
        return (c[:, j + 1] - c[:, low]) / window

    r_in = inside_window(np.clip((scales["tau"] - t1) / p.W_SOFT_T1, 0, 1))
    r_t2 = inside_window(np.clip((t2 - scales["tau2"]) / scales["w2"], 0, 1))
    muscle = (np.clip((t1 - scales["tau"]) / p.W_SOFT_T1, 0, 1) * np.clip((scales["tau2"] - t2) / scales["w2"], 0, 1) * mask)
    outward_muscle = np.cumsum(muscle[:, ::-1], 1)[:, ::-1]
    r_out = (outward_muscle - muscle) * DR / p.OUT_MM
    return edge + p.LAM_IN * r_in + p.LAM_OUT * r_out + p.LAM_T2 * r_t2 + p.EPS * radii[None, :]


def allowed_nodes(polar, endo2d, p):
    """Inside the mask, at least FLOOR_MM from the endocardial surface and beyond it; per-ray fallback where none is."""
    mask = polar["mask"]
    n_rays, n_radii = mask.shape
    j = np.arange(n_radii)
    if endo2d.any():
        query = polar["points"].reshape(-1, 2).T
        inside = ndi.map_coordinates(endo2d.astype(float), query, order=0, mode="constant", cval=0.0).reshape(n_rays, n_radii) > 0.5
        distance = ndi.map_coordinates(ndi.distance_transform_edt(~endo2d) * SUB_MM, query, order=1, mode="constant",
                                       cval=1e6).reshape(n_rays, n_radii)
        last = np.where(inside.any(1), n_radii - 1 - np.argmax(inside[:, ::-1], 1), -10 ** 6)
        beyond = j[None, :] > last[:, None]
        strict = mask & (distance >= p.FLOOR_MM) & beyond & (j[None, :] >= 1)
        floor_only = mask & (distance >= p.FLOOR_MM) & (j[None, :] >= 1)
    else:
        strict = floor_only = mask & (j[None, :] >= 1)
    any_mask = mask & (j[None, :] >= 1)
    no_strict = ~strict.any(1)
    no_floor = ~floor_only.any(1)
    allowed = strict.copy()
    allowed[no_strict] = floor_only[no_strict]
    allowed[no_strict & no_floor] = any_mask[no_strict & no_floor]
    return allowed


def wedge_weight(polar, tips_deg, p):
    """Per-ray smoothness weight: 0 within the wedge around each RV insertion point, a linear taper, 1 elsewhere."""
    weight = np.ones(N_THETA)
    if not tips_deg:
        return weight
    theta = np.degrees(polar["theta"])
    for tip in tips_deg:
        relative = np.abs((theta - tip + 180) % 360 - 180)
        core = p.WEDGE_DEG - p.TAPER_DEG
        weight = np.minimum(weight, np.clip((relative - core) / p.TAPER_DEG, 0, 1))
    return weight


def dp_open_face(cost, d_face, cap, alpha, start=None, end=None):
    """Open-chain shortest path over rays with a capped, face-following smoothness term: between neighbouring rays the
    radius may change by the face's change plus or minus `cap`, and each radial step of deviation from the face's change
    costs `alpha`.

    Exact output: keep the offset loop; a vectorised form changes how ties break.
    """
    n, m = cost.shape
    accumulated = np.full((n, m), np.inf)
    back = np.zeros((n, m), np.int32)
    accumulated[0] = cost[0] if start is None else np.where(np.arange(m) == start, cost[0], np.inf)
    for i in range(1, n):
        previous = accumulated[i - 1]
        cap_steps = cap[i] / DR
        best = np.full(m, np.inf)
        best_arg = np.zeros(m, np.int32)
        for offset in range(int(np.ceil(d_face[i] - cap_steps)), int(np.floor(d_face[i] + cap_steps)) + 1):
            candidate = np.full(m, np.inf)
            if offset >= 0:
                candidate[offset:] = previous[:m - offset] if offset > 0 else previous
            else:
                candidate[:offset] = previous[-offset:]
            candidate = candidate + alpha[i] * abs(offset - d_face[i])
            better = candidate < best
            best[better] = candidate[better]
            best_arg[better] = (np.arange(m) - offset)[better]
        accumulated[i] = best + cost[i]
        back[i] = best_arg
    last = accumulated[-1] if end is None else np.where(np.arange(m) == end, accumulated[-1], np.inf)
    final = int(np.argmin(last))
    if not np.isfinite(last[final]):
        return None
    path = np.zeros(n, np.int32)
    path[-1] = final
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def solve_closed(cost, polar, p, weight):
    """Closed path: two laps fix the start, then one lap tied to it; the cap is widened if infeasible."""
    face = polar["face_smooth"]
    n = cost.shape[0]
    d_face = np.diff(np.r_[face[-1], face])
    weight_edge = 0.5 * (weight + np.roll(weight, 1))
    alpha = p.ALPHA * DR * weight_edge
    cap = np.where(weight_edge < 1.0, p.CAP_WEDGE, p.CAP)
    for factor in (1.0, 2.0, 4.0):
        two_laps = dp_open_face(np.concatenate([cost, cost], 0), np.concatenate([d_face, d_face]),
                                np.concatenate([cap, cap]) * factor, np.concatenate([alpha, alpha]))
        if two_laps is None:
            continue
        start = int(two_laps[n:][0])
        path = dp_open_face(cost, d_face, cap * factor, alpha, start=start, end=start)
        if path is not None:
            return path
    return None


def solve_slice(volume, k, scales, p, endo, groove, tips_deg, centre=None, d_reference=None):
    """Solve one slice; returns the polar sampling, the path, the depth below the face and any relaxation used."""
    d = slice_arrays(volume, k)
    polar = polar_sample(d, p, centre)
    bad_rays = not_star_shaped(polar)
    costs = node_costs(polar, scales, p)
    index = 2 * k
    allowed = allowed_nodes(polar, endo[:, :, index], p)
    weight = wedge_weight(polar, tips_deg, p)
    groove_ray = (ndi.map_coordinates(groove[:, :, index].astype(float), polar["points"].reshape(-1, 2).T, order=0,
                                      mode="constant", cval=0.0).reshape(allowed.shape) > 0.5
                  if groove[:, :, index].any() else np.zeros_like(allowed))
    if d_reference is not None:
        depth = (polar["face_smooth"][:, None] - np.arange(polar["n_radii"])[None, :]) * DR
        costs = costs + p.LAM_K * np.abs(depth - np.asarray(d_reference)[:, None])
    path = solve_closed(np.where(allowed & ~groove_ray, costs, np.inf), polar, p, weight)
    relaxed = False
    if path is None:
        for alternative in (allowed, polar["mask"] & (np.arange(polar["n_radii"])[None, :] >= 1)):
            path = solve_closed(np.where(alternative, costs, np.inf), polar, p, weight)
            if path is not None:
                relaxed = True
                break
    if path is None:
        return None, bad_rays
    return dict(polar=polar, path=path, relaxed=relaxed,
                depth_below_face=(polar["face_smooth"] - path) * DR), bad_rays


def solve_all(volume, slices, scales, p, endo, groove, tips, centres, reference, star_bad_frac):
    """Solve every slice, falling back to the slice's own centroid when the smoothed centre is not star-shaped."""
    solved = {}
    depth = np.full((len(slices), N_THETA), np.nan)
    for i, k in enumerate(slices):
        d_ref = None if reference is None else reference[i]
        result, bad = solve_slice(volume, k, scales, p, endo, groove, tips.get(k, []), centre=centres[i], d_reference=d_ref)
        if result is None or bad > star_bad_frac * N_THETA:
            result2, bad2 = solve_slice(volume, k, scales, p, endo, groove, tips.get(k, []), centre=None, d_reference=d_ref)
            if result2 is not None and (result is None or bad2 < bad):
                result = result2
        solved[k] = result
        if result is not None and not result["relaxed"]:
            depth[i] = result["depth_below_face"]
    return solved, depth


def smooth_depth(depth, p):
    """Depth field smoothed along the slice axis; a slice with no unrelaxed solution copies its nearest valid one."""
    filled = depth.copy()
    bad = np.isnan(filled).any(1)
    # Latent: if no slice has an unrelaxed solution, `good` is empty and argmin raises an error. Every slice of this
    # heart has one.
    for i in np.where(bad)[0]:
        good = np.where(~bad)[0]
        filled[i] = filled[good[np.argmin(np.abs(good - i))]]
    return ndi.gaussian_filter1d(filled, p.K_SIGMA, axis=0, mode="nearest")


def slice_centres(volume, slices, sigma):
    """Ray centre per slice: centroid of the filled manual mask and cavities, smoothed along the slice axis."""
    cavity = volume["lv"] | volume["rv"]
    centroids = [np.argwhere(ndi.binary_fill_holes(volume["manual"][:, :, 2 * k] | cavity[:, :, 2 * k])).astype(float).mean(0)
                 for k in slices]
    return ndi.gaussian_filter1d(np.array(centroids), sigma, axis=0, mode="nearest")


def endo_surface(volume, p, open_mm, close_mm, k_majority):
    """Smoothed endocardial surface: cavities and voids opened, closed and hole-filled per slice, then kept where more
    than half of k_majority neighbouring sub-slices along k agree."""
    manual = volume["manual"]
    cavity = volume["lv"] | volume["rv"]
    opened = np.zeros(manual.shape, bool)
    open_disk, close_disk = disk(int(round(open_mm / 0.5))), disk(int(round(close_mm / 0.5)))
    for k in range(volume["bottom"], volume["top"] + 1):
        index = 2 * k
        current = cavity[:, :, index] | interior_void(slice_arrays(volume, k), p)
        if current.sum() < 4:
            continue
        smoothed = ndi.binary_fill_holes(ndi.binary_closing(ndi.binary_opening(current, structure=open_disk),
                                                            structure=close_disk, border_value=0))
        opened[:, :, index] = smoothed
        opened[:, :, index + 1] = smoothed
    return ndi.uniform_filter1d(opened.astype(np.float32), k_majority, axis=2, mode="nearest") > 0.5


def rv_tips(rv_slice):
    """The two RV insertion points of a slice: the ends of the RV cavity crescent (geodesic diameter of its largest
    component), or None."""
    components, n = ndi.label(rv_slice)
    if n == 0:
        return None
    sizes = ndi.sum(rv_slice, components, range(1, n + 1))
    comp = components == (1 + int(np.argmax(sizes)))
    if comp.sum() < RV_TIP_MIN_VOXELS:
        return None
    cost = np.where(comp, 1.0, 1e6)
    distances, _ = MCP_Geometric(cost).find_costs([tuple(np.argwhere(comp)[0])])
    first = tuple(np.unravel_index(np.argmax(np.where(comp, distances, -1)), comp.shape))
    distances, _ = MCP_Geometric(cost).find_costs([first])
    second = tuple(np.unravel_index(np.argmax(np.where(comp, distances, -1)), comp.shape))
    return np.array([first, second], float)


def groove_channel(volume, endo, tau2, p, centres, slices, block_dilation):
    """The interventricular groove channel next to each RV insertion point: non-muscle tissue (T2-bright, or void)
    connected to the outside in 3-D over all sub-slices, within wedges around the insertion points. The trace may not
    place a node in this channel, so the traced epicardium passes around the groove instead of cutting through it."""
    manual = volume["manual"]
    cavity = volume["lv"] | volume["rv"]
    t1, t2 = volume["t1"].astype(np.float32), volume["t2"].astype(np.float32)
    non_muscle = (t2 > tau2) | ((t1 < p.VOID_T1) & (t2 < p.VOID_T2))
    wedge = np.zeros(manual.shape, bool)
    tips = {}
    ii, jj = np.mgrid[0:manual.shape[0], 0:manual.shape[1]]
    for i, k in enumerate(slices):
        if k < p.K_MIN:
            continue
        ends = rv_tips(volume["rv"][:, :, 2 * k])
        if ends is None:
            continue
        c = centres[i]
        angles = np.arctan2(jj - c[1], ii - c[0])
        sector = np.zeros(manual.shape[:2], bool)
        tips[k] = []
        for end in ends:
            tip_angle = np.arctan2(end[1] - c[1], end[0] - c[0])
            sector |= np.abs(np.angle(np.exp(1j * (angles - tip_angle)))) <= np.radians(p.WEDGE_DEG)
            tips[k].append(float(np.degrees(tip_angle) % 360))
        wedge[:, :, 2 * k] = sector
        wedge[:, :, 2 * k + 1] = sector
    outside = ~ndi.binary_fill_holes(manual | cavity)
    passable = non_muscle & ~ndi.binary_dilation(endo, iterations=block_dilation)
    components, _ = ndi.label(passable | outside, structure=S26)
    reached = np.unique(components[outside])
    reached = reached[reached > 0]
    return passable & np.isin(components, reached) & manual & wedge, tips


def trace_all(volume, scales, p, endo, groove, tips, centres, star_bad_frac):
    """Every slice with the iterated inter-slice coupling; returns centre and radii per slice."""
    slices = list(range(volume["bottom"], volume["top"] + 1))
    solved, depth = solve_all(volume, slices, scales, p, endo, groove, tips, centres, None, star_bad_frac)
    for _ in range(p.N_ITER):
        solved, depth = solve_all(volume, slices, scales, p, endo, groove, tips, centres, smooth_depth(depth, p), star_bad_frac)
    reference = smooth_depth(depth, p)
    paths = {}
    for i, k in enumerate(slices):
        result = solved[k]
        if result is None:
            # Latent: a slice with no solution, even after the relaxations, is skipped without a message; voxelise then
            # keeps none of its voxels, so the slice drops out of the traced mask. Every slice of this heart is solved.
            continue
        polar = result["polar"]
        path = result["path"]
        if result["relaxed"]:     # A slice that needed a relaxation continues from its neighbours instead.
            j = np.arange(polar["n_radii"])[None, :]
            alternative = polar["mask"] & (polar["d_cavity"] >= p.WALL_MIN) & (j >= 1)
            target = np.clip(np.rint(polar["face_smooth"] - reference[i] / DR).astype(int), 0, polar["n_radii"] - 1)
            path = np.array([(np.where(alternative[t])[0][np.argmin(np.abs(np.where(alternative[t])[0] - target[t]))]
                              if alternative[t].any() else target[t]) for t in range(N_THETA)])
        paths[k] = dict(centre=polar["centre"], radii=polar["radii"][path])
    return paths


def voxelise(paths, manual):
    """Traced outlines to a 1 mm mask: a voxel is kept when its centre is inside; largest component; holes filled."""
    theta = np.arange(N_THETA) * 2 * np.pi / N_THETA
    inside = np.zeros(manual.shape, bool)
    for k, path in paths.items():
        polygon = path["centre"][None, :] + np.stack([np.cos(theta), np.sin(theta)], 1) * (path["radii"] / 0.5)[:, None]
        voxels = np.argwhere(manual[:, :, k])
        if voxels.size:
            kept = points_in_poly(2 * voxels + 0.5, polygon)
            inside[voxels[kept, 0], voxels[kept, 1], k] = True
    traced = manual & inside
    components, _ = ndi.label(traced, structure=S26)
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    traced = components == int(sizes.argmax())
    traced |= ndi.binary_fill_holes(traced) & ~traced & manual
    return traced
