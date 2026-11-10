"""MP-PCA denoising of the registered repeats, averaging per volume, then Gibbs-ringing removal.

MP-PCA is DIPY's implementation of Veraart et al. 2016 (Denoising of diffusion MRI using random matrix theory,
NeuroImage 142:394-406); Gibbs removal is DIPY's implementation of Kellner et al. 2016 (Gibbs-ringing artifact removal
based on local subvoxel-shifts, Magn Reson Med 76:1574-1581).
"""
import multiprocessing as mp

import nibabel as nib
import numpy as np
from dipy.denoise.gibbs import gibbs_removal
from dipy.denoise.localpca import mppca

from .common import load_config, work_path


def _denoise_slab(job):
    slab, patch_radius = job
    # With patch radius 2 a patch has 125 voxels. Because 125 - 1 is fewer than the 155 volumes, DIPY warns that this
    # "might have a performance impact"; suppress_warning silences that notice. DIPY then estimates the noise from the
    # 124 largest eigenvalues of each patch; the others are zero.
    return mppca(slab, patch_radius=patch_radius, pca_method="eig", suppress_warning=True)


def slab_mppca(stack, params):
    """MP-PCA in overlapping slabs along the slice axis; the estimator is local to each patch, so the slabs do not change
    the result provided the overlap is at least twice the patch radius."""
    n_slices = stack.shape[2]
    edges = np.linspace(0, n_slices, params["slabs"] + 1).astype(int)
    jobs, keep = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        low, high = max(a - params["overlap"], 0), min(b + params["overlap"], n_slices)
        jobs.append((np.ascontiguousarray(stack[:, :, low:high]), params["patch_radius"]))
        keep.append((a - low, b - low, a, b))
    with mp.get_context("spawn").Pool(params["processes"]) as pool:
        slabs = pool.map(_denoise_slab, jobs)
    denoised = np.zeros(stack.shape, np.float32)
    for slab, (s0, s1, a, b) in zip(slabs, keep):
        denoised[:, :, a:b] = slab[:, :, s0:s1]
    return denoised


def main(cfg):
    params = cfg["denoise"]
    source = nib.load(work_path(cfg, "register", "dwi_raw_registered.nii"))
    denoised = slab_mppca(np.asarray(source.dataobj, dtype=np.float32), params)

    n_volumes = np.loadtxt(work_path(cfg, "convert", "dwi.bval")).size
    n_repeats = denoised.shape[3] // n_volumes
    average = np.zeros(denoised.shape[:3] + (n_volumes,), np.float32)
    for d in range(n_volumes):
        average[..., d] = denoised[..., [r * n_volumes + d for r in range(n_repeats)]].mean(axis=-1)
    del denoised

    # Order: MP-PCA on the repeats, then average, then Gibbs removal on the average. MP-PCA runs before averaging so it
    # sees the redundancy of the five repeats, and before Gibbs removal because Gibbs removal changes the noise structure
    # MP-PCA relies on. The sub-voxel Fourier shift of the registration leaves the k-space window unchanged, so Gibbs
    # removal can follow it.
    final = gibbs_removal(average, slice_axis=params["gibbs_slice_axis"], inplace=False,
                          num_processes=params["processes"]).astype(np.float32)
    header = source.header.copy()
    header.set_data_dtype(np.float32)
    header.set_slope_inter(1.0, 0.0)
    nib.save(nib.Nifti1Image(final, source.affine, header), work_path(cfg, "denoise", "dwi_denoised.nii.gz"))


if __name__ == "__main__":
    main(load_config())
