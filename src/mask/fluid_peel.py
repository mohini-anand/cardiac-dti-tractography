"""Surface peel of voxels at least half fluid (coronary vessel lumens in the grooves and at the base), found on T1 and T2.

Each image is divided by its own local muscle level, taken from a band 1 to 3 mm deep in the mask; a sub-voxel is fluid
when its T2 is at least twice and its T1 under half that level.
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter

from .images import nested_grid, resample_native, up

S6 = ndi.generate_binary_structure(3, 1)
S26 = np.ones((3, 3, 3), bool)

SUB_MM = 0.5
NORM_SIGMA_MM = 3.0
NORM_SIGMA_FALLBACK_MM = 8.0
NORM_MIN_SUPPORT = 0.02
REF_DEPTH_MM = (1.0, 3.0)
DEEP_MM = 2.0
REF_T2_MAX = 2.0
REF_T1_MIN = 0.5
FLUID_T2N = 2.0
FLUID_T1N = 0.5
MAJORITY = 4          # at least half: 4 of the 8 sub-voxels of a 1 mm voxel


def is_fluid(t1_fine, t2_fine, mask_fine):
    """Per sub-voxel, whether it reads fluid after dividing each image by the local muscle level."""
    depth = ndi.distance_transform_edt(mask_fine, sampling=SUB_MM) - SUB_MM / 2
    deep = mask_fine & (depth > DEEP_MM)
    median_t1, median_t2 = float(np.median(t1_fine[deep])), float(np.median(t2_fine[deep]))
    reference = (mask_fine & (depth >= REF_DEPTH_MM[0]) & (depth <= REF_DEPTH_MM[1]) & (t2_fine < REF_T2_MAX * median_t2)
                 & (t1_fine > REF_T1_MIN * median_t1)).astype(np.float32)
    sigma, sigma_fallback = NORM_SIGMA_MM / SUB_MM, NORM_SIGMA_FALLBACK_MM / SUB_MM
    weight, weight_fallback = gaussian_filter(reference, sigma), gaussian_filter(reference, sigma_fallback)
    scale = {}
    for key, image in (("t1", t1_fine), ("t2", t2_fine)):
        local = gaussian_filter(image * reference, sigma) / np.maximum(weight, 1e-6)
        wider = gaussian_filter(image * reference, sigma_fallback) / np.maximum(weight_fallback, 1e-6)
        scale[key] = np.where(weight > NORM_MIN_SUPPORT, local, wider).astype(np.float32)
    return (t2_fine / scale["t2"] >= FLUID_T2N) & (t1_fine / scale["t1"] < FLUID_T1N)


def peel(mask, affine, t1_image, t2_image, transform):
    """Remove surface-connected voxels with at least MAJORITY (half) fluid sub-voxels; keep the largest component."""
    mask = np.asarray(mask, bool)
    fine_affine, fine_shape = nested_grid(affine, mask.shape)
    # The native T1 and T2, resampled here: no T2 slab correction and no bias normalisation from images.nested_volume;
    # is_fluid divides each by its own local muscle level instead.
    fluid = is_fluid(resample_native(t1_image, transform, fine_affine, fine_shape),
                     resample_native(t2_image, transform, fine_affine, fine_shape), up(mask))
    s = fluid.shape
    counts = fluid.reshape(s[0] // 2, 2, s[1] // 2, 2, s[2] // 2, 2).sum((1, 3, 5)).astype(np.uint8)
    candidates = mask & (counts >= MAJORITY)
    exterior = ~np.pad(mask, 1, constant_values=False)
    touching = np.asarray(ndi.binary_dilation(exterior, S6), bool)[1:-1, 1:-1, 1:-1] & candidates
    components = np.asarray(ndi.label(candidates, S6)[0])
    ids = np.unique(components[touching])
    peeled = np.isin(components, ids[ids > 0]) & candidates
    result = mask & ~peeled
    labels, _ = ndi.label(result, S26)
    return labels == int(np.argmax(np.bincount(labels.ravel())[1:])) + 1
