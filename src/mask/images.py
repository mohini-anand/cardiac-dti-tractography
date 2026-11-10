"""The manual mask, its cavity split and the bias-normalised T1 and T2 images on a 0.5 mm grid nested in the DWI grid.

Each DWI voxel is exactly 2 x 2 x 2 sub-voxels; sub-voxel I covers DWI voxel coordinates [0.5 I - 0.5, 0.5 I], so
sub-slice 2k is the lower half of DWI slice k and sub-slice 2k + 1 its upper half.

T2 has no registration of its own: it is moved with the T1 transform, and its slab joins are measured with a T1 tissue
mask on its own grid. T1 and T2 must therefore share one voxel grid and be aligned in scanner space; measured on this
heart during development, they agree within 0.16 mm.
"""
import nibabel as nib
import numpy as np
from dipy.align.imaffine import AffineMap
from nibabel.processing import resample_from_to
from scipy import ndimage as ndi

from . import exclusion

SUB_MM = 0.5
NOT_CAVITY, LV_CAVITY, RV_CAVITY = 0, 1, 2     # cavity class codes; a coding separate from the classes in exclusion.py


def up(mask):
    """Repeat a 1 mm volume onto the nested 0.5 mm grid."""
    return np.repeat(np.repeat(np.repeat(mask, 2, 0), 2, 1), 2, 2)


def nested_grid(affine, shape):
    """Affine and shape of the 0.5 mm grid nested in a 1 mm grid."""
    half = np.array([[0.5, 0, 0, -0.25], [0, 0.5, 0, -0.25], [0, 0, 0.5, -0.25], [0, 0, 0, 1]])
    return np.asarray(affine, float) @ half, tuple(2 * np.asarray(shape))


def resample_native(image, transform, affine, shape):
    """Native anatomical image on the nested grid; `transform` maps DWI-frame world to anatomical world (DIPY's
    convention), so resampling with it brings the image into the DWI frame."""
    data = np.asarray(image.dataobj).astype(np.float32)
    # Exact output: DIPY's AffineMap (trilinear); the mask steps read these resampled values, so keep this resampler.
    return AffineMap(np.asarray(transform, float), domain_grid_shape=shape, domain_grid2world=affine,
                     codomain_grid_shape=data.shape, codomain_grid2world=image.affine).transform(data).astype(np.float32)


def load_base(manual_image, segmentation, cavity_label):
    """Manual mask, LV and RV cavities, slice range and the 1 mm cavity class volume (LV_CAVITY and RV_CAVITY where a
    cavity voxel is outside the manual mask, NOT_CAVITY elsewhere)."""
    manual = np.asarray(manual_image.dataobj) > 0
    affine = manual_image.affine
    labels = np.asarray(resample_from_to(segmentation, (manual.shape, affine), order=0).dataobj).round().astype(np.int16)
    lv, rv = exclusion.split_cavity(labels == cavity_label, manual, exclusion.voxel_sizes(affine))
    slices = np.nonzero(manual.sum((0, 1)))[0]
    top, bottom = int(slices[-1]), int(slices[0])
    classes = np.full(manual.shape, NOT_CAVITY, np.uint8)
    classes[lv & ~manual] = LV_CAVITY
    classes[rv & ~manual] = RV_CAVITY
    return dict(affine=affine, manual=manual, lv=lv, rv=rv, classes=classes, top=top, bottom=bottom)


def t2_slab_join(t2, t1, joins, t1_window):
    """Intensity step of the native T2 at each slab join, to be divided out.

    The T2 volume was acquired in slabs; a slab join is the boundary between two slabs, named by the first slice above
    it. For each pair of neighbouring slices the step is the median difference of log T2 over tissue that both slices
    share: T1 inside t1_window, eroded by 3 voxels, and at least 500 voxels. The step at a join is then detrended: the
    mean of the steps of the slice pairs just below and just above it is subtracted.
    """
    t1, t2 = np.asarray(t1, float), np.asarray(t2, float)
    tissue = ndi.binary_erosion((t1 > t1_window[0]) & (t1 < t1_window[1]), iterations=3)
    log_t2 = np.log(np.maximum(t2, 1.0))
    n_slices = t2.shape[2]
    step = np.full(n_slices, np.nan)
    for k in range(1, n_slices):
        common = tissue[:, :, k] & tissue[:, :, k - 1]
        if common.sum() >= 500:
            step[k] = float(np.median(log_t2[:, :, k][common] - log_t2[:, :, k - 1][common]))
    # Latent: if fewer than 500 eroded tissue voxels are shared across a join, or across the slice step on either side of
    # it, that join's step is NaN and every T2 slice from the join up becomes NaN. Both joins of this heart are measured.
    detrended = np.full(n_slices, np.nan)
    for k in range(2, n_slices - 1):
        detrended[k] = step[k] - 0.5 * (step[k - 1] + step[k + 1])
    return {int(j): float(detrended[j]) for j in joins}


def bias_normalise(image, manual_fine, deep_mm, sigma_mm):
    """Divide by a smooth receive-bias estimate taken from the deep wall."""
    depth = ndi.distance_transform_edt(manual_fine, sampling=SUB_MM)
    deep = manual_fine & (depth >= deep_mm)
    median = float(np.median(image[deep]))
    reference = (deep & (image > 0.5 * median) & (image < 2.0 * median)).astype(np.float32)
    sigma = sigma_mm / SUB_MM
    bias = ndi.gaussian_filter(image * reference, sigma) / np.maximum(ndi.gaussian_filter(reference, sigma), 1e-6)
    return (image / np.maximum(bias, 1e-3)).astype(np.float32)


def nested_volume(base, transform, t1_image, t2_image, params):
    """Slab-corrected, bias-normalised T1 and T2 on the nested grid, plus the 1 mm classes repeated onto it."""
    fine_affine, fine_shape = nested_grid(base["affine"], base["manual"].shape)
    t1 = np.asarray(t1_image.dataobj).astype(np.float32)
    t2 = np.asarray(t2_image.dataobj).astype(np.float32)
    corrected = t2.copy()
    # Applied cumulatively: slices after the later join are divided by both factors.
    for join, step in t2_slab_join(t2, t1, params["t2_slab_joins"], params["t1_window"]).items():
        corrected[:, :, join:] /= np.float32(np.exp(step))
    t1_fine = resample_native(nib.Nifti1Image(t1, t1_image.affine), transform, fine_affine, fine_shape)
    t2_fine = resample_native(nib.Nifti1Image(corrected, t2_image.affine), transform, fine_affine, fine_shape)
    manual_fine = up(base["manual"])
    normalised = {}
    for name, image in (("t1", t1_fine), ("t2", t2_fine)):
        scaled = bias_normalise(image, manual_fine, params["norm_deep_mm"], params["norm_sigma_mm"])
        normalised[name] = ndi.gaussian_filter(scaled, params["img_sigma_mm"] / SUB_MM).astype(np.float32)
    # Exact output: the float16 cast stays; the trace and the later steps read these rounded values.
    return dict(t1=normalised["t1"].astype(np.float16), t2=normalised["t2"].astype(np.float16), manual=manual_fine,
                lv=up(base["lv"]), rv=up(base["rv"]), classes=up(base["classes"]).astype(np.uint8),
                top=base["top"], bottom=base["bottom"])
