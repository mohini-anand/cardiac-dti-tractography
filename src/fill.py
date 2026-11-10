"""Bootstrap reliability of the fibre direction, and log-Euclidean filling of unreliable voxels from reliable neighbours."""
import os

# This module limits the numerical libraries to one thread each, unless the environment already sets these variables; the
# bootstrap runs in parallel worker processes instead. They must be set before numpy is imported.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import multiprocessing as mp  # noqa: E402

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from dipy.core.gradients import gradient_table  # noqa: E402
from dipy.reconst.dti import design_matrix  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

from .common import load_array, load_config, save_float32, work_path  # noqa: E402

MIN_POSITIVE_SIGNAL = 1e-4  # DIPY replaces signals below this before taking the logarithm
RADIUS = 2                  # 5 x 5 x 5 support window, centre excluded
S6 = ndi.generate_binary_structure(3, 1)

# Filled before the fork so that the bootstrap workers share the samples.
_worker = {}


def resample_weights(b, n_repeats, n_volumes, seed_base):
    """Stratified repetition bootknife (Hesterberg 2004, Proceedings of the Section on Statistics & the Environment,
    American Statistical Association): each acquired volume is a separate stratum; for each, one repeat is left out at
    random and n_repeats are drawn with replacement from the remaining repeats. Returns how often each repeat is drawn."""
    rng = np.random.default_rng(seed_base + b)
    weights = np.zeros((n_repeats, n_volumes), np.float32)
    # Exact output: the draw order is fixed; per volume, one `integers` draw and then one `choice` draw from this generator.
    for d in range(n_volumes):
        drop = rng.integers(n_repeats)
        pool = [r for r in range(n_repeats) if r != drop]
        np.add.at(weights[:, d], rng.choice(pool, n_repeats, replace=True), 1.0)
    return weights


def fast_v1(average, design, pseudo_inverse):
    """DIPY's WLS estimator through its normal equations; returns the eigenvector of the largest eigenvalue."""
    log_signal = np.log(np.maximum(average, MIN_POSITIVE_SIGNAL))    # Exact output: float32, as DIPY uses for float32 input
    beta_ols = log_signal @ pseudo_inverse.T
    weights = np.exp(beta_ols @ design.T) ** 2
    normal = np.einsum("nk,ki,kj->nij", weights, design, design)
    right_hand = np.einsum("nk,ki,nk->ni", weights, design, log_signal)
    beta = np.linalg.solve(normal, right_hand[..., None])[..., 0]
    tensor = np.empty((len(beta), 3, 3))
    tensor[:, 0, 0], tensor[:, 1, 1], tensor[:, 2, 2] = beta[:, 0], beta[:, 2], beta[:, 5]
    tensor[:, 0, 1] = tensor[:, 1, 0] = beta[:, 1]
    tensor[:, 0, 2] = tensor[:, 2, 0] = beta[:, 3]
    tensor[:, 1, 2] = tensor[:, 2, 1] = beta[:, 4]
    _, vectors = np.linalg.eigh(tensor)
    return vectors[..., -1]


def _resample(b):
    """One resample: the reweighted average of the repeats and its primary eigenvector, stored as float16."""
    samples = _worker["samples"]
    n_repeats, _, n_volumes = samples.shape
    weights = resample_weights(b, n_repeats, n_volumes, _worker["seed_base"])
    average = np.einsum("rd,rnd->nd", weights, samples) / n_repeats
    # Exact output: float16 is the stored precision; it is renormalised to float32 before the cone angles are taken.
    return b, fast_v1(average, _worker["design"], _worker["pseudo_inverse"]).astype(np.float16)


def cone_of_uncertainty(directions):
    """95th percentile of the per-resample angle to the mean dyadic direction: the 95% cone of uncertainty (Jones 2003,
    Magn Reson Med 49:7-12)."""
    dyadic = np.einsum("bni,bnj->nij", directions, directions) / len(directions)
    _, vectors = np.linalg.eigh(dyadic)
    mean = vectors[..., -1]
    angles = np.degrees(np.arccos(np.clip(np.abs(np.einsum("bni,ni->bn", directions, mean)), 0, 1)))
    return np.percentile(angles, 95, axis=0)


def bootstrap(cfg):
    """Cone-of-uncertainty map of V1 over the manual mask, from the registered raw repeats.

    The resamples use the registered repeats before denoising: MP-PCA runs on all repeats together and shares information
    between them, which would shrink the spread between repeats. The map therefore measures the repeatability of the raw
    measurement, not of the denoised field that is tracked.
    """
    params = cfg["fill"]
    manual = nib.load(work_path(cfg, "convert", "myo_mask_dwi.nii.gz"))
    region = np.where(np.asarray(manual.dataobj) > 0)
    stack = nib.load(work_path(cfg, "register", "dwi_raw_registered.nii"), mmap=True)
    bvals = np.loadtxt(work_path(cfg, "convert", "dwi.bval"))
    bvecs = np.loadtxt(work_path(cfg, "convert", "dwi.bvec"))
    n_volumes = bvals.size
    n_repeats = stack.shape[3] // n_volumes
    samples = np.empty((n_repeats, region[0].size, n_volumes), np.float32)
    for r in range(n_repeats):
        for d in range(n_volumes):
            samples[r, :, d] = np.asarray(stack.dataobj[..., r * n_volumes + d], np.float32)[region]
    # Latent: unlike tensor.gradient_table_for, these directions are not x-flipped. A single global reflection leaves the
    # angles between resamples, and so the cone, unchanged. The resampled V1 would be mirrored if it were used as a
    # direction; it is used only for the cone, so the result is not affected.
    design = design_matrix(gradient_table(bvals, bvecs=bvecs.T if bvecs.shape[0] == 3 else bvecs,
                                          b0_threshold=cfg["dicom"]["b0_threshold"]))
    _worker.update(samples=samples, design=design, pseudo_inverse=np.linalg.pinv(design), seed_base=params["seed_base"])

    resamples = params["resamples"]
    stored = np.zeros((resamples, region[0].size, 3), np.float16)
    with mp.get_context("fork").Pool(params["processes"]) as pool:
        for b, direction in pool.imap_unordered(_resample, range(resamples), chunksize=4):
            stored[b] = direction
    directions = np.empty(stored.shape, np.float32)
    for b in range(resamples):
        vector = stored[b].astype(np.float32)
        directions[b] = vector / np.linalg.norm(vector, axis=-1, keepdims=True)
    volume = np.zeros(manual.shape[:3], np.float32)
    volume[region] = cone_of_uncertainty(directions)
    save_float32(volume, manual.affine, work_path(cfg, "fill", "cou95.nii.gz"))


def window_kernel(sigma):
    """Window offsets in C order and Gaussian weights, centre excluded."""
    offsets = np.indices((2 * RADIUS + 1,) * 3).reshape(3, -1).T - RADIUS
    gaussian = np.exp(-(offsets ** 2).sum(1) / (2.0 * sigma ** 2))
    gaussian[(offsets == 0).all(1)] = 0.0
    return offsets.astype(np.int64), gaussian


def gather(volume, index, offsets, fill):
    """Neighbours of each index in the window; positions outside the array read `fill`."""
    shape = np.array(volume.shape[:3])
    neighbours = index[:, None, :] + offsets[None, :, :]
    inside = ((neighbours >= 0) & (neighbours < shape)).all(-1)
    clipped = np.where(inside[..., None], neighbours, 0)
    flat = np.ravel_multi_index((clipped[..., 0], clipped[..., 1], clipped[..., 2]), tuple(shape))
    out = volume.reshape((-1,) + volume.shape[3:])[flat]
    if (~inside).any():
        out[~inside] = fill
    return out


def support_weights(index, mask, reliable, offsets, gaussian):
    """Gaussian weight of reliable neighbours 6-connected to the centre through the mask inside the window."""
    in_mask = gather(mask, index, offsets, False)
    weights = gaussian[None, :] * gather(reliable, index, offsets, False)
    connected = np.zeros_like(in_mask)
    for n in range(len(index)):
        labels, _ = ndi.label(in_mask[n].reshape(5, 5, 5), S6)
        connected[n] = (labels == labels[2, 2, 2]).ravel()
    return weights * connected


def certainty_and_offset(weights, offsets, gaussian):
    """Support as a fraction of the full window, and the magnitude of its centre-of-mass offset."""
    total = weights.sum(1)
    certainty = total / float(gaussian.sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        offset = np.linalg.norm((weights[:, :, None] * offsets[None, :, :]).sum(1) / total[:, None], axis=1)
    offset[total <= 0] = np.inf
    return certainty, offset


def support_coherence(index, weights, v1, offsets):
    """Largest eigenvalue of the weighted mean of v1 v1^T over the support; NaN without support."""
    total = weights.sum(1)
    used = weights > 0
    vectors = np.where(used[..., None], gather(v1, index, offsets, 0.0).astype(np.float64), 0.0)
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    vectors = np.divide(vectors, norm, out=np.zeros_like(vectors), where=norm > 0)
    coherence = np.full(len(index), np.nan)
    has_support = total > 0
    if has_support.any():
        dyadic = np.einsum("nk,nki,nkj->nij", weights[has_support], vectors[has_support], vectors[has_support]) / total[has_support, None, None]
        coherence[has_support] = np.linalg.eigvalsh(dyadic)[:, 2]
    return coherence


def log_euclidean_v1(index, weights, evals, evecs, offsets):
    """Primary eigenvector of the weighted log-Euclidean mean tensor of the support (Arsigny et al. 2006, Magn Reson Med
    56:411-421)."""
    total = weights.sum(1)
    used = weights > 0
    eigenvalues = gather(evals, index, offsets, 1.0).astype(np.float64)
    eigenvectors = gather(evecs, index, offsets, 0.0).astype(np.float64)
    eigenvalues = np.where(used[..., None], eigenvalues, 1.0)      # Exact output: unused rows read log(1) = 0
    log_tensor = np.einsum("nkij,nkj,nklj->nkil", eigenvectors, np.log(eigenvalues), eigenvectors)
    mean = (weights[:, :, None, None] * log_tensor).sum(1) / total[:, None, None]
    mean = 0.5 * (mean + np.swapaxes(mean, 1, 2))
    _, vectors = np.linalg.eigh(mean)                               # Exact output: float64 eigh, then float32
    return vectors[:, :, ::-1].copy()[:, :, 0].astype(np.float32)


def tensor_fill(cfg):
    """Fill voxels with an unreliable direction when their reliable neighbourhood supports it; write the track maps."""
    params = cfg["fill"]
    reference = nib.load(work_path(cfg, "tensor", "v1.nii.gz"))
    evals, evecs = load_array(work_path(cfg, "tensor", "evals.nii.gz")), load_array(work_path(cfg, "tensor", "evecs.nii.gz"))
    v1 = load_array(work_path(cfg, "tensor", "v1.nii.gz"))
    fa_stop = load_array(work_path(cfg, "tensor", "fa_stop.nii.gz"))
    failed = load_array(work_path(cfg, "tensor", "failed.nii.gz")) > 0
    track_weight = load_array(work_path(cfg, "tensor", "track_weight.nii.gz")) > 0
    mask = np.asarray(nib.load(work_path(cfg, "mask", "mask_final.nii.gz")).dataobj) > 0
    cone = np.asarray(nib.load(work_path(cfg, "fill", "cou95.nii.gz")).dataobj, np.float32)

    trigger = (cone > params["cou95_trigger_deg"]) & mask
    depth = ndi.distance_transform_edt(mask)
    candidates = (trigger | failed) & mask
    reliable = mask & ~trigger & ~failed
    offsets, gaussian = window_kernel(params["sigma"])

    index = np.argwhere(candidates)
    weights = support_weights(index, mask, reliable, offsets, gaussian)
    certainty, offset = certainty_and_offset(weights, offsets, gaussian)
    coherence = support_coherence(index, weights, v1, offsets)
    accepted = ((depth[tuple(index.T)] > params["depth_min"]) & (certainty >= params["c_min"]) & (offset <= params["m_max"])
                & (coherence >= params["coherence_min"]))
    filled_index = index[accepted]
    new_v1 = log_euclidean_v1(filled_index, weights[accepted], evals, evecs, offsets)

    filled = np.zeros(mask.shape, bool)
    filled[tuple(filled_index.T)] = True
    # Unreliable voxels that are not filled keep their V1 and their tracking weight (failed fits have none) and are tracked
    # through; only the cleaning step treats them separately, through unreliable_unfilled.
    v1_out = v1.copy()
    v1_out[tuple(filled_index.T)] = new_v1
    for name, array in (("v1", v1_out), ("track_weight", track_weight | filled), ("fa_stop", fa_stop), ("filled", filled),
                        ("unreliable_unfilled", candidates & ~filled)):
        save_float32(array, reference.affine, work_path(cfg, "fill", f"{name}.nii.gz"))


def main(cfg):
    bootstrap(cfg)
    tensor_fill(cfg)


if __name__ == "__main__":
    main(load_config())
