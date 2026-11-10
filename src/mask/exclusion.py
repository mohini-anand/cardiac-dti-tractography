"""Papillary and trabecular exclusion by two-sided in-plane enclosure.

For every tissue voxel near a cavity, rays are cast in the short-axis plane; each records the first cavity or exterior
class it meets, passing through tissue and enclosed voids. A structure standing in a cavity has that cavity on both
sides along most in-plane lines; a wall has it on one side only. Uses only the geometry of the mask and the cavity
labels.

The class codes below (TISSUE, LV, RV, EXTERIOR, ENCLOSED) are a coding separate from the cavity classes of images.py.
"""
import math
from types import SimpleNamespace

import numpy as np
from scipy import ndimage as ndi

S6 = ndi.generate_binary_structure(3, 1)
S26 = np.ones((3, 3, 3), bool)
TISSUE, LV, RV, EXTERIOR, ENCLOSED = 0, 1, 2, 3, 4

PARAMS = SimpleNamespace(n_dirs=72, step_mm=0.4, r_max_mm=25.0, d_cand_mm=12.0, theta_core=0.55, theta_grow=0.45,
                         grow_max_mm=3.0, min_component_mm3=27.0)


def voxel_sizes(affine):
    """Unrounded voxel sizes. The size floors are written as 27 mm^3 (3 x 3 x 3 voxels at 1 mm); on this grid the sizes
    read 0.99999998 mm, so a 27 mm^3 floor needs 28 voxels, not 27. Exact output: hull.rounded_voxel_sizes rounds them
    instead; keep the two apart."""
    return np.sqrt((np.asarray(affine)[:3, :3] ** 2).sum(0))


def geodesic_flood(seed_classes, domain, connectivity=S6):
    """Nearest-first flood of seed labels (>0) through `domain`; ties at the same step go to the lower label."""
    out = seed_classes.astype(np.int8).copy()
    pending = domain & (out == 0)
    if not pending.any():
        return out
    nonzero = np.nonzero(pending)
    low = [max(int(a.min()) - 1, 0) for a in nonzero]
    high = [min(int(a.max()) + 2, s) for a, s in zip(nonzero, pending.shape)]
    box = tuple(slice(a, b) for a, b in zip(low, high))
    local, remaining = out[box], pending[box].copy()
    labels = [int(v) for v in np.unique(local[local > 0])]
    while remaining.any():
        new = np.zeros_like(local)
        for value in labels:
            grown = ndi.binary_dilation(local == value, structure=connectivity) & remaining & (new == 0)
            new[grown] = value
        if not new.any():
            break
        local[new > 0] = new[new > 0]
        remaining &= new == 0
    out[box] = local
    return out


def classify_nontissue(tissue, lv_cavity, rv_cavity, pre_peel=None):
    """Class volume: tissue, LV cavity, RV cavity, exterior, enclosed void.

    Voxels peeled from `pre_peel` take the cavity or exterior they reach first through non-tissue.
    """
    tissue = tissue.astype(bool)
    classes = np.zeros(tissue.shape, np.int8)
    classes[lv_cavity & ~tissue] = LV
    classes[rv_cavity & ~tissue] = RV
    non_tissue = ~tissue & (classes == TISSUE)
    if pre_peel is None:
        classes[non_tissue] = EXTERIOR
    else:
        classes[non_tissue & ~pre_peel] = EXTERIOR
        pending = non_tissue & pre_peel
        classes = geodesic_flood(classes, pending)
        classes[pending & (classes == TISSUE)] = ENCLOSED
    holes = ndi.binary_fill_holes(tissue) & ~tissue & (classes == EXTERIOR)
    classes[holes] = ENCLOSED
    return classes


def base_side(solid, slice_axis=2):
    """+1 if the larger end face along the slice axis (the flat basal cut) is at high index, else -1."""
    area = solid.sum(axis=tuple(a for a in range(3) if a != slice_axis))
    slices = np.nonzero(area)[0]
    low, high = area[slices[0]:slices[0] + 3].mean(), area[slices[-1] - 2:slices[-1] + 1].mean()
    return 1 if high >= low else -1


def basal_slab(shape, solid, slice_axis=2):
    """Slices beyond the basal-most occupied slice (the open cut)."""
    area = solid.sum(axis=tuple(a for a in range(3) if a != slice_axis))
    slices = np.nonzero(area)[0]
    k = np.arange(shape[slice_axis])
    selected = k > slices[-1] if base_side(solid, slice_axis) > 0 else k < slices[0]
    sh = [1, 1, 1]
    sh[slice_axis] = shape[slice_axis]
    return np.broadcast_to(selected.reshape(sh), shape)


def distance_to(mask, zooms):
    """Euclidean distance (mm) to the nearest True voxel; infinite when `mask` is empty."""
    if not mask.any():
        return np.full(mask.shape, np.inf)
    return ndi.distance_transform_edt(~mask, sampling=zooms)


def wall_thickness_ratio(tissue, classes, zooms, slab):
    """Median LV-side wall thickness over median RV-side wall thickness."""
    cavity = np.isin(classes, (LV, RV))
    endo = tissue & ndi.binary_dilation(cavity, S6)
    epi = tissue & ndi.binary_dilation((classes == EXTERIOR) & ~slab, S6) & ~endo
    thickness = ndi.distance_transform_edt(~endo, sampling=zooms) + ndi.distance_transform_edt(~epi, sampling=zooms)
    d_lv, d_rv = distance_to(classes == LV, zooms), distance_to(classes == RV, zooms)
    lv_side, rv_side = tissue & (d_lv < d_rv), tissue & (d_rv < d_lv)
    return float(np.median(thickness[lv_side])) / max(float(np.median(thickness[rv_side])), 1e-9)


def split_cavity(cavity, tissue, zooms, pre_peel=None, slice_axis=2):
    """Split one cavity label into LV and RV: the thicker wall belongs to the LV, whichever component is larger."""
    components, _ = ndi.label(cavity)
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    order = np.argsort(sizes)[::-1]
    # Latent: if the cavity label is one connected component, order[1] is label 0 and the whole background becomes the
    # second cavity. This heart's cavity label has two components.
    a, b = components == order[0], components == order[1]
    classes = classify_nontissue(tissue, a, b, pre_peel)
    slab = basal_slab(tissue.shape, tissue | a | b, slice_axis)
    if wall_thickness_ratio(tissue, classes, zooms, slab) >= 1.0:
        return a, b
    return b, a


def short_axis_directions(axis, n):
    """n unit vectors evenly spaced in the plane perpendicular to the long axis."""
    long_axis = np.asarray(axis, float)
    # Exact output: np.asarray returns the caller's float64 array itself, so this normalises the caller's axis in place;
    # exclude() passes that array on to the hull step. Keep it.
    long_axis /= np.linalg.norm(long_axis)
    e = np.eye(3)[int(np.argmin(np.abs(long_axis)))]
    u = np.cross(long_axis, e)
    u /= np.linalg.norm(u)
    v = np.cross(long_axis, u)
    phi = 2 * np.pi * np.arange(n) / n
    return np.cos(phi)[:, None] * u + np.sin(phi)[:, None] * v


def first_hits(classes, affine, axis, candidates):
    """First cavity or exterior class met along each in-plane direction from each candidate voxel; tissue and enclosed
    voids are passed through, and outside the array counts as exterior (0: none met within r_max_mm)."""
    p = PARAMS
    inverse = np.linalg.inv(np.asarray(affine)[:3, :3])
    shape = np.array(classes.shape)
    flat = classes.ravel()
    index = np.argwhere(candidates)
    n = len(index)
    hits = np.zeros((n, p.n_dirs), np.int8)
    steps = np.arange(p.step_mm, p.r_max_mm + 1e-9, p.step_mm)
    origin = index.astype(np.float64)
    for direction, d in enumerate(short_axis_directions(axis, p.n_dirs)):
        step_voxels = inverse @ d
        alive = np.arange(n)
        for s in steps:
            q = np.rint(origin[alive] + s * step_voxels).astype(np.int64)
            in_bounds = np.all((q >= 0) & (q < shape), axis=1)
            current = np.full(len(alive), EXTERIOR, np.int8)
            current[in_bounds] = flat[np.ravel_multi_index(q[in_bounds].T, classes.shape)]
            done = (current != TISSUE) & (current != ENCLOSED)
            hits[alive[done], direction] = current[done]
            alive = alive[~done]
            if not len(alive):
                break
    return index, hits


def two_sided_enclosure(classes, affine, axis, tissue, zooms):
    """Fraction of in-plane lines whose two ends both meet the LV (or RV) cavity, per voxel near a cavity."""
    candidates = tissue & (distance_to(np.isin(classes, (LV, RV)), zooms) <= PARAMS.d_cand_mm)
    index, hits = first_hits(classes, affine, axis, candidates)
    half = PARAMS.n_dirs // 2
    fractions = {}
    for name, cavity in (("lv", LV), ("rv", RV)):
        both = (hits[:, :half] == cavity) & (hits[:, half:] == cavity)
        fraction = np.zeros(classes.shape, np.float32)
        fraction[tuple(index.T)] = both.mean(1)
        fractions[name] = fraction
    return fractions["lv"], fractions["rv"]


def detect_enclosed(enclosure_lv, enclosure_rv, tissue, zooms):
    """Hysteresis on each ventricle's enclosure, geodesically capped, then a size floor."""
    p = PARAMS
    voxel_mm3 = float(np.prod(zooms))
    min_voxels = int(math.ceil(p.min_component_mm3 / voxel_mm3 - 1e-9))
    n_grow = int(round(p.grow_max_mm / float(np.min(zooms))))
    out = {}
    for name, enclosure, other in (("lv", enclosure_lv, enclosure_rv), ("rv", enclosure_rv, enclosure_lv)):
        domain = tissue & (enclosure > other)
        core = domain & (enclosure >= p.theta_core)
        grow = domain & (enclosure >= p.theta_grow)
        reach = core.copy()
        for _ in range(n_grow):
            grown = ndi.binary_dilation(reach, structure=S6) & grow
            if (grown == reach).all():
                break
            reach = grown
        components, _ = ndi.label(reach, structure=S26)
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        out[name] = np.isin(components, np.nonzero(sizes >= min_voxels)[0])
    return out["lv"], out["rv"]


def finalize(tissue, excluded):
    """Largest 26-connected component of the tissue minus the excluded voxels."""
    kept = tissue & ~excluded
    components, _ = ndi.label(kept, structure=S26)
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    return components == int(sizes.argmax())


def exclude(tissue, lv_cavity, rv_cavity, affine, pre_peel, axis):
    """Two-sided enclosure exclusion. Returns the arrays the hull step and the RV trabecular step read."""
    zooms = voxel_sizes(affine)
    tissue = tissue.astype(bool).copy()
    classes = classify_nontissue(tissue, lv_cavity.astype(bool), rv_cavity.astype(bool), pre_peel.astype(bool))
    # Exact output: normalised here into a new array, which short_axis_directions then normalises again in place, so the
    # hull step receives the axis normalised twice and normalises it again itself (conditionally in hull.slab_frame, and
    # in middle_half_guard and slab_positions). Merging any of these normalisations changes the last bits of the axis.
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    enclosure_lv, enclosure_rv = two_sided_enclosure(classes, affine, axis, tissue, zooms)
    enclosed_lv, enclosed_rv = detect_enclosed(enclosure_lv, enclosure_rv, tissue, zooms)
    return dict(classes=classes, axis=axis, tissue=tissue, enclosed_lv=enclosed_lv, enclosed_rv=enclosed_rv,
                pre_peel=pre_peel.astype(bool), lv_label=lv_cavity.astype(bool) & (classes == LV),
                rv_label=rv_cavity.astype(bool) & (classes == RV))
