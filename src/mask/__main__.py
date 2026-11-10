"""Refine the manual mask: trace the epicardium, peel it, exclude papillary and trabecular muscle, bridge cuts in the wall."""
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pyvista as pv
from scipy import ndimage as ndi

from . import bridge, chain, epi_trace, exclusion, fluid_peel, hull, images, rv_trabeculae
from ..common import load_array, load_config, work_path


def attenuation_volume(cfg, manual):
    """Per-voxel b0 minus mean diffusion-weighted signal, in units of the noise of one volume averaged over the repeats
    (not the noise of the difference), in the dilated manual mask."""
    stack = nib.load(work_path(cfg, "register", "dwi_raw_registered.nii"), mmap=True)
    bvals = np.loadtxt(work_path(cfg, "convert", "dwi.bval"))
    n_volumes = bvals.size
    n_repeats = stack.shape[3] // n_volumes
    # Exact output: default cross structure and np.where order; this voxel order is what the attenuation map is defined on.
    region = np.where(ndi.binary_dilation(manual, iterations=3))
    samples = np.empty((n_repeats, region[0].size, n_volumes), np.float32)
    for r in range(n_repeats):
        for d in range(n_volumes):
            samples[r, :, d] = np.asarray(stack.dataobj[..., r * n_volumes + d], np.float32)[region]
    b0_threshold = cfg["dicom"]["b0_threshold"]
    is_b0 = np.where(bvals <= b0_threshold)[0]
    is_dw = np.where(bvals > b0_threshold)[0]
    # Exact output: the noise is a pooled variance, across repeats per voxel and volume, averaged over volumes, then one
    # square root. Computed any other way, such as a mean of standard deviations, voxels can cross the peel threshold.
    mean = samples.mean(0)
    sigma = np.sqrt(samples.var(0, ddof=1).mean(1))      # across-repeat noise of one repeat (float32, ddof 1)
    noise_of_average = sigma / np.sqrt(n_repeats)
    attenuation = (mean[:, is_b0].mean(1) - mean[:, is_dw].mean(1)) / np.maximum(noise_of_average, 1e-6)
    volume = np.full(manual.shape, np.inf)
    volume[region] = attenuation
    return volume


def lv_rv_side(manual, affine, cfg):
    """LV or RV side of each manual-mask voxel from the transmural depth: -1 where the depth is below 0, else +1."""
    transmural = pv.read(cfg["paths"]["transmural_vtk"])
    ii, jj, kk = np.where(manual)
    world = (affine[:3, :3] @ np.stack([ii, jj, kk], 1).T.astype(float)).T + affine[:3, 3]
    # Exact output: float32 world coordinates, the tolerance and the snap flag together decide which cell each point samples.
    depth = np.asarray(pv.PolyData(world.astype(np.float32)).sample(transmural, tolerance=1.0, snap_to_closest_point=True)["TransmuralDepth"]).astype(np.float32)
    side = np.zeros(manual.shape, np.int8)
    # Latent: unlike clean.py, this ignores vtkValidPointMask; a voxel the transmural volume does not reach samples depth 0
    # and is labelled RV, so it gets the RV T1 peel threshold. Every manual-mask voxel of this heart is reached.
    side[ii, jj, kk] = np.where(depth < 0, -1, 1)
    return side


def refine_epicardium(cfg, base, t1_image, t2_image, transform, t1_reg, attenuation, side):
    """The epicardial refinement: traced mask, peels with ring guards, exclusions and the fluid peel, then the basal cut
    and the bridges. The RV trabecular removal follows it (main)."""
    params, manual, affine = cfg["mask"], base["manual"], base["affine"]
    trace = SimpleNamespace(**params["trace"])
    scales = params["trace_scales"]
    cavity = base["lv"] | base["rv"]
    volume = images.nested_volume(base, transform, t1_image, t2_image, params["images"])
    slices = list(range(volume["bottom"], volume["top"] + 1))

    surface = params["endo_surface"]
    endo = epi_trace.endo_surface(volume, trace, surface["open_mm"], surface["close_mm"], surface["k_majority_subslices"])
    centres = epi_trace.slice_centres(volume, slices, params["trace_centre_sigma_slices"])
    groove, tips = epi_trace.groove_channel(volume, endo, scales["tau2"], trace, centres, slices, params["groove_block_dilation_subvox"])
    paths = epi_trace.trace_all(volume, scales, trace, endo, groove, tips, centres, params["trace_star_bad_frac"])
    traced = epi_trace.voxelise(paths, manual)

    peels = params["peels"]
    restorable = chain.restorable_map(volume["t1"], volume["t2"], scales["tau"], scales["tau2"], trace.VOID_T1, trace.VOID_T2,
                                      peels["restorable_min_subvox"])
    t1_peeled = chain.guard(traced, chain.t1_peel(traced, side, t1_reg.astype(float), peels["t1_lv"], peels["t1_rv"]), cavity, restorable)
    attenuation_peeled = chain.guard(t1_peeled, chain.attenuation_peel(t1_peeled, attenuation, peels["attenuation_max"]), cavity, restorable)

    exclusion_result = exclusion.exclude(attenuation_peeled, base["lv"], base["rv"], affine, manual, cfg["geometry"]["long_axis"])
    after_hulls = hull.add_hull(exclusion_result, t1_reg, affine, dict(params["hull"], dark_t1_max=params["dark_t1_max"]))
    # The fluid peel resamples the native T1 and T2 itself, without the T2 slab correction, the bias normalisation and the
    # float16 rounding of the images above; it divides each image by its own local muscle level instead.
    fluid_peeled = fluid_peel.peel(after_hulls, affine, t1_image, t2_image, transform)
    # This ring guard compares with the mask after the attenuation peel, so where the exclusion, the hulls or the fluid
    # peel opened a slice's ring it can put back voxels any of them removed.
    peeled = chain.guard(attenuation_peeled, fluid_peeled, cavity, restorable)

    # Exact output: the basal cut comes after the peels; they use the top slice to tell the open surface from the cut face.
    last_slice = params["basal_cut_last_slice"]
    peeled[:, :, last_slice + 1:] = False
    endo_strict, endo_distance = bridge.voxel_classes(endo)
    bridger = bridge.Bridger(peeled, traced, manual, cavity, endo_strict, endo_distance, params["bridge"])
    bridges = bridger.run([k for k in slices if k <= last_slice])
    return peeled | bridges, bridges, volume, exclusion_result["enclosed_rv"]


def main(cfg):
    manual_image = nib.load(work_path(cfg, "convert", "myo_mask_dwi.nii.gz"))
    manual = np.asarray(manual_image.dataobj) > 0
    affine = manual_image.affine
    segmentation = nib.load(cfg["paths"]["segmentation"])
    transform = np.load(work_path(cfg, "register", "T1_to_b0_affine.npy"))
    t1_image = nib.load(work_path(cfg, "convert", "T1_FLASH.nii.gz"))
    t2_image = nib.load(work_path(cfg, "convert", "T2_TSE.nii.gz"))
    t1_reg = load_array(work_path(cfg, "register", "T1_reg2b0.nii.gz"))     # float32; the T1 peel reads it as float64

    base = images.load_base(manual_image, segmentation, cfg["mask"]["cavity_label"])
    attenuation = attenuation_volume(cfg, manual)
    side = lv_rv_side(manual, affine, cfg)
    final, bridges, volume, enclosed_rv = refine_epicardium(cfg, base, t1_image, t2_image, transform, t1_reg, attenuation, side)

    # The RV trabecular removal runs after the bridges and keeps them.
    params = cfg["mask"]["rv_trabeculae"]
    trabeculae = dict(params, dark_t1_max=cfg["mask"]["dark_t1_max"], cleft_void_t1=cfg["mask"]["trace"]["VOID_T1"],
                      cleft_void_t2=cfg["mask"]["trace"]["VOID_T2"])
    # Exact output: the RV step widens the float16 images to float32; the float16 rounding stays in the values.
    final = rv_trabeculae.remove(final, enclosed_rv, bridges, manual, base["lv"] | base["rv"], volume["t1"].astype(np.float32),
                                 volume["t2"].astype(np.float32), base["lv"].astype(bool), base["rv"].astype(bool), t1_reg,
                                 exclusion.voxel_sizes(affine), cfg["mask"]["trace_scales"], trabeculae)

    header = manual_image.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(final.astype(np.uint8), affine, header), work_path(cfg, "mask", "mask_final.nii.gz"))
    nib.save(nib.Nifti1Image((final & ~bridges).astype(np.uint8), affine, header), work_path(cfg, "mask", "seed_mask.nii.gz"))


if __name__ == "__main__":
    main(load_config())
