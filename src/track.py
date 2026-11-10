"""Deterministic tensor tractography along the primary eigenvector, inside the refined myocardial mask.

The direction getter is DIPY's PeaksAndMetrics, the EuDX-style getter (Garyfallidis 2012), given one peak per voxel:
its primary eigenvector, used as computed, not rounded to a set of sphere directions. At each step it takes the eight
voxels around the current point. A voxel contributes if it has a peak and its eigenvector lies within max_angle_deg of
the direction of travel, ignoring the eigenvector's sign; it then adds its eigenvector, flipped to agree with that
direction, weighted by its trilinear weight. A voxel outside the angle cap, or without a peak, adds nothing, and its
weight does not count. If the weight that counts is below total_weight, the streamline ends there; otherwise the step
follows the normalised weighted sum, a mix of neighbouring voxels' directions.
"""
import nibabel as nib
import nibabel.streamlines as nib_streamlines
import numpy as np
from dipy.core.sphere import Sphere
from dipy.direction.peaks import PeaksAndMetrics
from dipy.tracking import utils
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
from dipy.tracking.streamline import length
from nibabel.streamlines import Field, LazyTractogram

from .common import load_array, load_config, work_path
from .tensor import border_connected

# DIPY's direction getter uses a voxel's peak only where the peak value exceeds this threshold. Every usable peak has
# value 1.0 (exact_v1_peaks), so any threshold from 0 up to, but not including, 1 keeps them all and gives the same
# streamlines. The value is DIPY's own default.
PEAK_VALUE_THRESHOLD = 0.0239


def exact_v1_peaks(v1, weight, max_angle, qa_threshold, total_weight):
    """Peaks object for DIPY's PeaksAndMetrics direction getter with one peak per usable voxel: its primary eigenvector,
    unrounded, with peak value 1.0. Voxels outside `weight` have no peak."""
    v1 = np.asarray(v1, float)
    norm = np.linalg.norm(v1, axis=-1)
    usable = weight & np.isfinite(v1).all(-1) & (norm > 1e-6)
    ii, jj, kk = np.nonzero(usable)
    directions = v1[ii, jj, kk] / norm[ii, jj, kk, None]
    peaks = PeaksAndMetrics()
    peaks.sphere = Sphere(xyz=directions)
    indices = np.full(v1.shape[:3] + (1,), -1, np.int32)
    indices[ii, jj, kk, 0] = np.arange(len(ii), dtype=np.int32)
    values = np.zeros(v1.shape[:3] + (1,), np.float64)
    values[ii, jj, kk, 0] = 1.0
    peaks.peak_indices = indices
    peaks.peak_values = values
    peaks.qa = values.copy()
    peaks.peak_dirs = np.zeros(v1.shape[:3] + (1, 3))
    peaks.peak_dirs[ii, jj, kk, 0] = directions
    peaks.ang_thr = float(max_angle)
    peaks.qa_thr = float(qa_threshold)
    peaks.total_weight = float(total_weight)
    return peaks


def main(cfg):
    params = cfg["tracking"]
    reference = nib.load(work_path(cfg, "fill", "v1.nii.gz"))
    affine = reference.affine
    v1 = load_array(work_path(cfg, "fill", "v1.nii.gz"))
    track_weight = load_array(work_path(cfg, "fill", "track_weight.nii.gz")) > 0
    fa_stop = load_array(work_path(cfg, "fill", "fa_stop.nii.gz"))
    filled = load_array(work_path(cfg, "fill", "filled.nii.gz"), np.uint8) > 0
    mask = load_array(work_path(cfg, "mask", "mask_final.nii.gz"), np.uint8) > 0
    seed_mask = load_array(work_path(cfg, "mask", "seed_mask.nii.gz"), np.uint8) > 0

    # The final mask drives direction and stopping; the seed mask only seeds.
    weight = (track_weight & mask) | (filled & mask)
    peaks = exact_v1_peaks(v1, weight, params["max_angle_deg"], PEAK_VALUE_THRESHOLD, params["total_weight"])
    # Low-FA voxels (failed fits read FA 0) stop tracking only where they connect to the mask border through other
    # low-FA voxels. Low-FA voxels enclosed in the wall are treated as holes in the field, and a hole should not end a
    # streamline, so they are not stops. A voxel without a peak does not stop tracking by itself either; DIPY ends a
    # streamline where less than total_weight of the trilinear weight around it has a usable peak, which can still
    # happen next to an unfilled failed fit.
    low_fa = (fa_stop < cfg["tensor"]["fa_stop"]) & mask & ~filled
    keep_going = mask & ~border_connected(low_fa, mask)
    criterion = BinaryStoppingCriterion(np.ascontiguousarray(keep_going, dtype=np.uint8))
    seeds = utils.seeds_from_mask(seed_mask, affine, density=params["seeds_per_axis"])    # regular grid; no random numbers
    inverse = np.linalg.inv(affine)
    # np.rint gives each seed's voxel (seeds sit 0.25 voxel from the centre on each axis, so none lies on a half); the
    # seed-voxel filter in clean.py reads these indices.
    seed_ijk = np.rint(seeds @ inverse[:3, :3].T + inverse[:3, 3]).astype(np.int32)
    # Exact output: the step length in mm uses the first voxel size only.
    step_mm = params["step_size_vox"] * float(reference.header.get_zooms()[0])
    kept = np.zeros(len(seeds), bool)

    def streamlines():
        # The writer calls this twice (it peeks at the first item), so kept flags are per seed and idempotent.
        # return_all=True with max_cross=1 yields exactly one streamline per seed, so the index below is the seed index.
        tracker = LocalTracking(peaks, criterion, seeds, affine, step_size=step_mm, maxlen=params["maxlen_steps"], max_cross=1,
                                return_all=True, save_seeds=False)
        for i, streamline in enumerate(tracker):
            # Exact output: the length is taken in float64; the float32 cast comes after.
            if float(length(streamline)) >= params["min_length_mm"]:
                kept[i] = True
                yield np.asarray(streamline, np.float32)

    header = {Field.VOXEL_TO_RASMM: affine.astype(np.float32), Field.DIMENSIONS: tuple(int(x) for x in reference.shape[:3]),
              Field.VOXEL_SIZES: tuple(float(z) for z in reference.header.get_zooms()[:3]),
              Field.VOXEL_ORDER: "".join(nib.aff2axcodes(affine))}
    # Streamed to disk: holding the kept streamlines in a list would need about 10 GB.
    nib_streamlines.save(LazyTractogram(streamlines, affine_to_rasmm=np.eye(4, dtype=np.float32)),
                         str(work_path(cfg, "track", "tractogram.trk")), header=header)
    index = np.nonzero(kept)[0]
    np.savez_compressed(work_path(cfg, "track", "seeds.npz"), input_seeds=seeds.astype(np.float32), streamline_seed_index=index,
                        streamline_seed_xyz=seeds[index].astype(np.float32), streamline_seed_ijk=seed_ijk[index])


if __name__ == "__main__":
    main(load_config())
