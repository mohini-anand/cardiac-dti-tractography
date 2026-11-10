"""The surface peels applied to the traced mask, the in-plane ring test, and the ring guard.

The ring guard runs after a step and compares the mask with the one before it: where the step opened the in-plane
myocardial ring of a slice, it restores a minimal path of removed tissue. The mask step runs it after the T1 peel, after
the attenuation peel, and once more after the exclusion, the hulls and the fluid peel (see mask/__main__.py).
"""
import numpy as np
from scipy import ndimage as ndi

S26 = np.ones((3, 3, 3), bool)
S6 = ndi.generate_binary_structure(3, 1)
S4 = ndi.generate_binary_structure(2, 1)
S8 = np.ones((3, 3), bool)
GUARD_LAYERS = 30


def largest_component(mask):
    components, _ = ndi.label(mask, structure=S26)
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    return components == int(sizes.argmax())


def t1_peel(mask, side, t1, tau_lv, tau_rv):
    """Remove surface-connected voxels below a T1 threshold (LV and RV sides differ).

    Removal spreads from the open epicardial and endocardial surfaces only, never from the base: the exterior voxels
    directly above the top slice (the basal cut face) seed nothing, while exterior voxels lower down, where the
    epicardium curves over, stay open. The exterior and the spread are 26-connected. The attenuation peel protects the
    whole slab above the top instead and spreads 6-connected. Exact output: keep the two different.
    """
    keep = np.where(side < 0, t1 > tau_lv, t1 > tau_rv)
    failing = mask & ~keep
    exterior = ~mask & ndi.binary_dilation(mask, structure=S26)
    top = int(np.max(np.nonzero(mask.sum((0, 1)))[0]))
    cut = np.zeros_like(mask)
    # Latent: if the mask reaches the last array slice, top + 1 is out of range and this raises an IndexError. This
    # heart's mask ends by slice 76, of slices 0 to 119.
    cut[:, :, top + 1] = mask[:, :, top]
    cut &= exterior
    open_surface = exterior & ~cut
    components, _ = ndi.label(open_surface | failing, structure=S26)
    reached = np.unique(components[open_surface])
    reached = reached[reached > 0]
    peeled = largest_component(mask & ~(failing & np.isin(components, reached)))
    return (peeled | (ndi.binary_fill_holes(peeled) & ~peeled)) & mask


def attenuation_peel(mask, attenuation, tau):
    """Remove surface-connected voxels with no diffusion attenuation; the whole slab above the top is protected.

    The flag is per voxel and noisy, so removal spreads only through face neighbours (6-connected): on such a flag a
    26-connected spread reaches deep into the wall by chance.
    """
    top = int(np.max(np.nonzero(mask.sum((0, 1)))[0]))
    z = np.arange(mask.shape[2])[None, None, :]
    exterior = ~mask & ndi.binary_dilation(mask, structure=S26)
    open_exterior = exterior & (z <= top)
    flag = mask & (attenuation < tau)
    components, _ = ndi.label(flag | open_exterior, structure=S6)
    reached = np.unique(components[open_exterior])
    reached = reached[reached > 0]
    kept = largest_component(mask & ~(flag & np.isin(components, reached)))
    return (kept | (ndi.binary_fill_holes(kept) & ~kept)) & mask


def restorable_map(t1, t2, tau, tau2, void_t1, void_t2, min_subvoxels):
    """1 mm voxels with at least `min_subvoxels` of their four in-plane sub-voxels at sub-slice 2k (the lower
    half-slice) reading as myocardium: T1 at or above tau, T2 at or below tau2, and not void."""
    t1, t2 = t1.astype(np.float32), t2.astype(np.float32)
    ok = (t1 >= tau) & (t2 <= tau2) & ~((t1 < void_t1) & (t2 < void_t2))
    count = sum(ok[a::2, b::2, ::2].astype(np.int8) for a in (0, 1) for b in (0, 1))
    return count >= min_subvoxels


def _border(shape):
    border = np.zeros(shape, bool)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    return border


def _exterior_labels(mask, structure):
    """Labels of the complement of `mask`, and the ids of the components touching the slice border."""
    labels, _ = ndi.label(~mask, structure=structure)
    exterior = np.unique(labels[_border(mask.shape) & ~mask])
    return labels, exterior[exterior > 0]


def inner_region(manual_slice, cavity_slice):
    """Cavity labels plus regions the manual mask encloses in-plane."""
    labels, exterior = _exterior_labels(manual_slice, S8)
    return (~manual_slice & ~np.isin(labels, exterior)) | cavity_slice


def ring_open(mask, inner, structure):
    """Whether any of the inner region connects to the slice border through the complement of the mask."""
    labels, exterior = _exterior_labels(mask, structure)
    return bool(np.isin(labels[inner & ~mask], exterior).any())


def open_slices(mask, manual, cavity, slices):
    """Slices whose myocardial ring is open, under 4- and 8-connectivity."""
    open_4, open_8 = [], []
    for k in slices:
        current = mask[:, :, k]
        if not current.any():
            continue
        inner = inner_region(manual[:, :, k], cavity[:, :, k])
        if inner.any():
            if ring_open(current, inner, S8):
                open_8.append(int(k))
            if ring_open(current, inner, S4):
                open_4.append(int(k))
    return open_4, open_8


def _closed(mask, inner, border):
    """Whether the ring is closed: no inner-region voxel connects to the border through the complement (8-connected)."""
    labels, _ = ndi.label(~mask, structure=S8)
    exterior = np.unique(labels[border & ~mask])
    exterior = exterior[exterior > 0]
    return not np.isin(labels[inner & ~mask], exterior).any()


def guard(previous, current, cavity, restorable):
    """Ring guard: where a step opened the in-plane ring, restore a pruned minimal path through removed voxels that the
    restorable map marks as tissue. Not the protected band of hull.middle_half_guard."""
    result = current.copy()
    for k in range(result.shape[2]):
        before, after = previous[:, :, k], result[:, :, k].copy()
        if not before.any() or not after.any():
            continue
        border = _border(before.shape)
        labels, _ = ndi.label(~before, structure=S8)
        exterior = np.unique(labels[border & ~before])
        exterior = exterior[exterior > 0]
        inner = (~before & ~np.isin(labels, exterior)) | cavity[:, :, k]
        if not inner.any() or _closed(after, inner, border):
            continue
        candidates = (before & ~after) & restorable[:, :, k]
        grown = after.copy()
        added = np.zeros_like(after)
        for _ in range(GUARD_LAYERS):
            layer = candidates & ndi.binary_dilation(grown, structure=S8) & ~grown
            if not layer.any():
                break
            grown |= layer
            added |= layer
            if _closed(grown, inner, border):
                break
        if not _closed(grown, inner, border):
            continue
        for voxel in np.argwhere(added):      # Exact output: pruning in scan order, each voxel tried once
            grown[voxel[0], voxel[1]] = False
            if not _closed(grown, inner, border):
                grown[voxel[0], voxel[1]] = True
        result[:, :, k] = grown
    return result
