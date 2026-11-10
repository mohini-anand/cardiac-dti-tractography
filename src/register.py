"""Align the T1 image to the diffusion frame, and remove a per-volume shift along the phase-encode axis.

The T1 image is aligned in two stages. `register_t1` aligns it to the mean b0 and saves the resampled image, which the
mask step uses for the T1 peel, the LV hull's dark points and the RV trabecular removal. `refit_t1_affine` then
registers the native T1 to that resampled copy and saves the affine it finds, which the mask step uses to resample T1
and T2 onto its 0.5 mm grid; T2 has no registration of its own. Both affines follow DIPY's convention: they map world
coordinates of the diffusion frame to world coordinates of the native T1, so resampling with one brings the T1 into
the diffusion frame.

Exact output: the mask step resamples with the re-fitted affine, not with register_t1's own. Measured on this heart,
the two differ by at most 0.015 mm over the myocardium, yet the final mask built with register_t1's own affine differs
by 165 voxels; the re-fit is kept so that the result reproduces exactly.

Each volume of each diffusion repeat gets one sub-voxel translation along the phase-encode axis. Measured on this heart
during development, the drift between repeats along the other two axes was at most 0.04 and 0.16 voxels and the
rotations at most 0.3 degrees. Nothing else is corrected: no rotation, no shift along the other two axes, and none of
the scaling or shear along the phase-encode axis that eddy currents can cause.
"""
import itertools
import multiprocessing as mp

import nibabel as nib
import numpy as np
from dipy.align.imaffine import (AffineMap, AffineRegistration, MutualInformationMetric,
                                 transform_centers_of_mass)
from dipy.align.transforms import AffineTransform3D, RigidTransform3D, TranslationTransform3D
from scipy import ndimage as ndi
from scipy.optimize import minimize_scalar

from .common import load_config, repeat_names, work_path


def register_t1(cfg):
    """Mutual-information alignment of the T1 image to the mean b0 (volume 0 of the averaged diffusion series): centre of
    mass, translation, rigid, affine. Saves the T1 resampled onto the b0 grid; refit_t1_affine then finds the affine
    that the mask step uses."""
    dwi = nib.load(work_path(cfg, "convert", "dwi_avg.nii.gz"))
    b0 = np.asarray(dwi.dataobj[..., 0], dtype=np.float32)
    t1 = nib.load(work_path(cfg, "convert", "T1_FLASH.nii.gz"))
    moving = np.asarray(t1.get_fdata(), dtype=np.float32)

    registration = AffineRegistration(metric=MutualInformationMetric(nbins=32, sampling_proportion=None),
                                      level_iters=[10000, 1000, 100], sigmas=[3.0, 1.0, 0.0], factors=[4, 2, 1],
                                      verbosity=0)
    transform = transform_centers_of_mass(b0, dwi.affine, moving, t1.affine).affine
    for model in (TranslationTransform3D(), RigidTransform3D(), AffineTransform3D()):
        transform = registration.optimize(b0, moving, model, None, dwi.affine, t1.affine,
                                          starting_affine=transform).affine
    aligned = AffineMap(transform, b0.shape, dwi.affine, moving.shape, t1.affine).transform(moving)
    nib.save(nib.Nifti1Image(aligned.astype(np.float32), dwi.affine), work_path(cfg, "register", "T1_reg2b0.nii.gz"))


def refit_t1_affine(cfg):
    """Register the native T1 (moving) to its registered copy (static) and save the affine found. In DIPY's convention
    it maps diffusion-frame world coordinates to native T1 world coordinates; the mask step resamples T1 and T2 with
    it."""
    native = nib.load(work_path(cfg, "convert", "T1_FLASH.nii.gz"))
    target = nib.load(work_path(cfg, "register", "T1_reg2b0.nii.gz"))
    moving = np.asarray(native.dataobj).astype(np.float32)
    static = np.asarray(target.dataobj).astype(np.float32)
    registration = AffineRegistration(metric=MutualInformationMetric(32, None), level_iters=[200, 100, 50],
                                      sigmas=[2, 1, 0], factors=[4, 2, 1], verbosity=0)
    start = registration.optimize(static, moving, TranslationTransform3D(), None, target.affine, native.affine,
                                  starting_affine=np.eye(4)).affine
    affine = registration.optimize(static, moving, AffineTransform3D(), None, target.affine, native.affine,
                                   starting_affine=start).affine
    # The file name says which way the image moves (T1 onto the b0 grid); the matrix maps the other way, from b0-frame
    # world coordinates to T1 world coordinates.
    np.save(work_path(cfg, "register", "T1_to_b0_affine.npy"), affine)


def fourier_shift(volume, shift, axis):
    """Shift a volume by a fraction of a voxel along one axis; keeps the real part and clips at zero."""
    if shift == 0:
        return np.asarray(volume, np.float32).copy()
    vector = np.zeros(3)
    vector[axis] = shift
    spectrum = np.fft.fftn(np.asarray(volume, np.float64))
    return np.clip(np.real(np.fft.ifftn(ndi.fourier_shift(spectrum, vector))), 0, None).astype(np.float32)


def correlation(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum())


# The worker pool uses the spawn start method, so its functions must stay at module level.
_worker = {}


def _init_worker(stack_path, target, roi, crop, n_volumes, params):
    _worker["image"] = nib.load(stack_path, mmap=True)
    _worker["target"] = target
    _worker["roi"] = roi
    _worker["crop"] = tuple(slice(a, b) for a, b in crop)
    _worker["n_volumes"] = n_volumes
    _worker["params"] = params
    shape = [1, 1, 1]
    shape[params["pe_axis"]] = -1
    _worker["frequencies"] = np.fft.fftfreq(target.shape[params["pe_axis"]]).reshape(shape)


def _load_volume(repeat, volume_index):
    index = repeat * _worker["n_volumes"] + volume_index
    volume = np.asarray(_worker["image"].dataobj[..., index], dtype=np.float32)
    return np.asarray(volume[_worker["crop"]], np.float64)


def estimate_shift(static, moving):
    """Shift along the phase-encode axis that best matches `moving` to `static`: coarse grid, then refinement."""
    params, roi, freq = _worker["params"], _worker["roi"], _worker["frequencies"]
    spectrum = np.fft.fft(moving, axis=params["pe_axis"])
    reference = static[roi]

    def cost(s):
        shifted = np.real(np.fft.ifft(spectrum * np.exp(-2j * np.pi * freq * s), axis=params["pe_axis"]))
        return -correlation(reference, shifted[roi])

    grid = np.arange(*params["grid"])
    start = grid[int(np.argmin([cost(s) for s in grid]))]
    half = params["refine_half"]
    result = minimize_scalar(cost, bounds=(start - half, start + half), method="bounded",
                             options={"xatol": params["xatol"]})
    return float(result.x)


def _estimate_job(task):
    kind, volume_index, a, b = task
    if kind == "pair":
        return task, estimate_shift(_load_volume(a, volume_index), _load_volume(b, volume_index))
    return task, estimate_shift(_worker["target"], _load_volume(a, volume_index))


def estimate_shifts(stack_path, target, roi, crop, n_repeats, n_volumes, params):
    """One shift per repeat and volume: the differences between repeats come from the repeat-to-repeat shifts by least
    squares (zero mean over the repeats), and the common offset is the mean of the shifts against the target."""
    tasks = [("pair", d, a, b) for d in range(n_volumes) for a, b in itertools.combinations(range(n_repeats), 2)]
    tasks += [("target", d, r, None) for d in range(n_volumes) for r in range(n_repeats)]
    with mp.get_context("spawn").Pool(params["processes"], initializer=_init_worker,
                                      initargs=(stack_path, target, roi, crop, n_volumes, params)) as pool:
        results = pool.map(_estimate_job, tasks, chunksize=4)

    pairwise = np.zeros((n_volumes, n_repeats, n_repeats))
    direct = np.zeros((n_repeats, n_volumes))
    for (kind, d, a, b), shift in results:
        if kind == "pair":
            pairwise[d, a, b] = shift
        else:
            direct[a, d] = shift

    relative = np.zeros((n_repeats, n_volumes))
    for d in range(n_volumes):
        rows, values = [], []
        for a, b in itertools.combinations(range(n_repeats), 2):
            row = np.zeros(n_repeats)
            row[b], row[a] = 1, -1
            rows.append(row)
            values.append(pairwise[d, a, b])
        rows.append(np.ones(n_repeats))
        values.append(0.0)
        relative[:, d] = np.linalg.lstsq(np.array(rows), np.array(values), rcond=None)[0]
    return direct.mean(0, keepdims=True) + relative


def register_phase_encode(cfg):
    """Estimate and remove a shift along the phase-encode axis for every volume of every repeat, b0 included."""
    params = cfg["registration"]
    axis = params["pe_axis"]
    # The repeats are listed from the DICOM folder, as in conversion, not from the files in the work folder.
    images = [nib.load(work_path(cfg, "convert", "raw", f"{name}.nii.gz")) for name in repeat_names(cfg)]
    n_repeats, n_volumes = len(images), images[0].shape[3]
    stack = np.concatenate([np.asarray(im.dataobj, dtype=np.float32) for im in images], axis=-1)

    # The workers read this stack by memory map instead of receiving a copy.
    header = images[0].header.copy()
    header.set_data_dtype(np.float32)
    header.set_slope_inter(1.0, 0.0)
    stack_path = work_path(cfg, "register", "raw_stack.nii")
    nib.save(nib.Nifti1Image(stack, images[0].affine, header), stack_path)
    raw = nib.load(stack_path, mmap=True)

    # The target is the mean of the repeats' b0 volumes, because the manual mask, the anatomical images and the surfaces
    # are all in the b0 frame. The matching region leaves out bright fluid (b0 above fluid_b0, a value set for this heart).
    myocardium = np.asarray(nib.load(work_path(cfg, "convert", "myo_mask_dwi.nii.gz")).dataobj) > 0
    b0_mean = np.stack([stack[..., r * n_volumes] for r in range(n_repeats)]).mean(0)
    dilated = np.argwhere(ndi.binary_dilation(myocardium, iterations=params["crop_dilate"]))
    low, high = dilated.min(0), dilated.max(0) + 1
    low[axis], high[axis] = 0, myocardium.shape[axis]
    crop = [(int(a), int(b)) for a, b in zip(low, high)]
    crop_slices = tuple(slice(a, b) for a, b in crop)
    roi = (ndi.binary_dilation(myocardium, iterations=params["roi_dilate"]) & (b0_mean < params["fluid_b0"]))[crop_slices]
    target = b0_mean[crop_slices].astype(np.float64)
    del stack

    shifts = estimate_shifts(stack_path, target, roi, crop, n_repeats, n_volumes, params)

    registered = np.zeros(raw.shape, np.float32)
    for r in range(n_repeats):
        for d in range(n_volumes):
            index = r * n_volumes + d
            volume = np.asarray(raw.dataobj[..., index], np.float32)
            registered[..., index] = fourier_shift(volume, float(shifts[r, d]), axis)
    nib.save(nib.Nifti1Image(registered, images[0].affine, header),
             work_path(cfg, "register", "dwi_raw_registered.nii"))


def main(cfg):
    register_t1(cfg)
    refit_t1_affine(cfg)
    register_phase_encode(cfg)


if __name__ == "__main__":
    main(load_config())
