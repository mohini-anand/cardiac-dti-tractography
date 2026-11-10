"""Weighted least-squares tensor fit with an explicit map of the voxels whose fit failed.

DIPY clips non-positive signals and eigenvalues silently, so a failed fit can read FA 0.7 to 1.0 or FA 0. Failed fits
are therefore found from the data and from an unclipped re-solve of the same estimator, never from FA.
"""
import numpy as np
import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel, design_matrix
from scipy import ndimage as ndi

from .common import load_array, load_config, save_float32, work_path

MIN_POSITIVE_SIGNAL = 1e-4  # DIPY replaces signals below this before taking the logarithm
EIGENVALUE_TOL = 1e-6       # DIPY's eigenvalue floor is this over minus the smallest design-matrix entry (about b)
AGREEMENT_TOL = 1e-4        # relative eigenvalue difference that marks a numerically unstable fit; set in development
CHUNK = 100_000


def gradient_table_for(bvals, bvecs, affine, b0_threshold):
    """DIPY gradient table; dcm2niix bvecs match the voxel axes only when the image is stored radiologically."""
    bvecs = np.array(bvecs, float)
    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
        bvecs = bvecs.T
    if np.linalg.det(np.asarray(affine, float)[:3, :3]) > 0:
        bvecs[:, 0] *= -1
    return gradient_table(np.asarray(bvals, float).ravel(), bvecs=bvecs, b0_threshold=b0_threshold)


def lower_triangular_to_symmetric(beta):
    tensors = np.empty(beta.shape[:-1] + (3, 3))
    tensors[..., 0, 0], tensors[..., 1, 1], tensors[..., 2, 2] = beta[..., 0], beta[..., 2], beta[..., 5]
    tensors[..., 0, 1] = tensors[..., 1, 0] = beta[..., 1]
    tensors[..., 0, 2] = tensors[..., 2, 0] = beta[..., 3]
    tensors[..., 1, 2] = tensors[..., 2, 1] = beta[..., 4]
    return tensors


def solve_rows(matrices, right_hand):
    """Solve each row's system; a singular row falls back to least squares on its own."""
    try:
        return np.linalg.solve(matrices, right_hand[..., None])[..., 0]
    except np.linalg.LinAlgError:
        solved = np.empty(right_hand.shape)
        for i in range(len(matrices)):
            try:
                solved[i] = np.linalg.solve(matrices[i], right_hand[i])
            except np.linalg.LinAlgError:
                solved[i] = np.linalg.lstsq(matrices[i], right_hand[i], rcond=None)[0]
        return solved


def unclipped_eigenvalues(signal, design):
    """DIPY's WLS estimator solved in batches with eigenvalues left unclipped, largest first."""
    design = np.asarray(design, float)
    pseudo_inverse = np.linalg.pinv(design)
    n = len(signal)
    eigenvalues = np.full((n, 3), np.nan)
    for start in range(0, n, CHUNK):
        batch = signal[start:start + CHUNK]
        finite = np.isfinite(batch).all(1)
        # Exact output: the logarithm is taken in float32, as DIPY does for float32 input.
        log_signal = np.log(np.maximum(batch[finite], MIN_POSITIVE_SIGNAL)).astype(float)
        beta_ols = log_signal @ pseudo_inverse.T
        with np.errstate(all="ignore"):
            weights = np.exp(beta_ols @ design.T) ** 2
            normal = np.einsum("nk,ki,kj->nij", weights, design, design)
            right_hand = np.einsum("nk,ki,nk->ni", weights, design, log_signal)
            beta = solve_rows(normal, right_hand)
        index = np.arange(start, start + len(batch))[finite]
        solved = np.isfinite(beta).all(1)
        if solved.any():
            # Exact output: eigh rather than eigvalsh; the two differ in the last digits and these values are thresholded.
            eigenvalues[index[solved]] = np.linalg.eigh(lower_triangular_to_symmetric(beta[solved]))[0][:, ::-1]
    return eigenvalues


def fit_tensor(data, gtab, mask):
    """DIPY's WLS fit plus the per-voxel quantities the failed-fit test needs."""
    design = design_matrix(gtab)
    floor = float(EIGENVALUE_TOL / -np.min(design))
    nonfinite_volume = mask & ~np.isfinite(data).all(-1)
    fit = TensorModel(gtab, fit_method="WLS").fit(data, mask=mask & ~nonfinite_volume)
    evals = np.asarray(fit.evals, np.float32)
    evecs = np.asarray(fit.evecs, np.float32)
    with np.errstate(all="ignore"):
        fa = np.asarray(fit.fa, np.float32)
    fa[nonfinite_volume] = np.nan

    signal = data[mask]
    is_b0 = np.asarray(gtab.b0s_mask, bool)
    unclipped = unclipped_eigenvalues(signal, design)
    nonfinite = nonfinite_volume[mask]
    dipy_evals = evals[mask].astype(float)
    with np.errstate(all="ignore"):
        b0 = signal[:, is_b0].mean(1)
        mean_dw = signal[:, ~is_b0].mean(1)
    positive_definite = ~nonfinite & np.isfinite(unclipped).all(1) & (unclipped[:, 2] > floor)
    relative = np.zeros(len(signal))
    relative[positive_definite] = (np.abs(dipy_evals[positive_definite] - unclipped[positive_definite])
                                   / np.maximum(np.abs(unclipped[positive_definite][:, :1]), floor)).max(1)
    # Where no eigenvalue needs clipping, DIPY's solve and the re-solve compute the same estimator by different routes,
    # so they should agree; a fit on which they differ by more than AGREEMENT_TOL is numerically unstable. It is counted
    # with the non-finite fits in failed_fits (its `non_finite` test). No voxel of this heart is unstable.
    unstable = positive_definite & (relative > AGREEMENT_TOL)
    facts = {"b0": b0, "mean_dw": mean_dw, "unclipped": unclipped, "floor": floor, "nonfinite": nonfinite,
             "unstable": unstable, "n_clipped": (signal <= MIN_POSITIVE_SIGNAL).sum(1)}
    return evals, evecs, fa, facts


def failed_fits(facts, mask, d_min, d_max):
    """Voxels whose fit is not physically possible, from any of six tests, which can overlap: no attenuation (mean
    diffusion-weighted signal at or above the b0, or a b0 that is not positive), a non-finite or unstable fit, a
    non-positive eigenvalue (at or below DIPY's eigenvalue floor, about 1e-9 mm^2/s here), an eigenvalue or mean
    diffusivity above d_max (free water), a signal clipped by DIPY's floor, and a smallest eigenvalue below d_min."""
    b0, mean_dw, eigenvalues, floor = facts["b0"], facts["mean_dw"], facts["unclipped"], facts["floor"]
    with np.errstate(invalid="ignore"):
        no_attenuation = np.isfinite(b0) & np.isfinite(mean_dw) & ((mean_dw >= b0) | (b0 <= 0))
    finite = np.isfinite(eigenvalues).all(1)
    non_finite = ~finite | facts["nonfinite"] | facts["unstable"]
    non_positive = finite & (eigenvalues <= floor).any(1)
    above_free_water = finite & ((eigenvalues[:, 0] > d_max) | (eigenvalues.mean(1) > d_max))
    clipped_signal = facts["n_clipped"] > 0
    near_zero = finite & (eigenvalues[:, 2] > floor) & (eigenvalues[:, 2] < d_min)
    failed = np.zeros(mask.shape, bool)
    failed[mask] = no_attenuation | non_finite | non_positive | above_free_water | clipped_signal | near_zero
    return failed


def border_connected(low, mask):
    """Voxels of `low` connected, through `low` voxels, to the outside of `mask` (6-connected)."""
    low = low & mask
    structure = ndi.generate_binary_structure(3, 1)
    outside = ~np.pad(mask, 1, constant_values=False)
    touching = np.asarray(ndi.binary_dilation(outside, structure), bool)[1:-1, 1:-1, 1:-1] & low
    labels = np.asarray(ndi.label(low, structure)[0])
    ids = np.unique(labels[touching])
    return np.isin(labels, ids[ids > 0]) & low


def main(cfg):
    params = cfg["tensor"]
    image = nib.load(work_path(cfg, "denoise", "dwi_denoised.nii.gz"))
    data = np.asarray(image.dataobj, np.float32)
    mask = load_array(work_path(cfg, "convert", "myo_mask_dwi.nii.gz"), np.uint8) > 0
    gtab = gradient_table_for(np.loadtxt(work_path(cfg, "convert", "dwi.bval")),
                              np.loadtxt(work_path(cfg, "convert", "dwi.bvec")), image.affine, cfg["dicom"]["b0_threshold"])

    evals, evecs, fa, facts = fit_tensor(data, gtab, mask)
    failed = failed_fits(facts, mask, params["d_min"], params["d_max"])
    fa_stop = np.nan_to_num(fa, nan=0.0, posinf=0.0, neginf=0.0)
    fa_stop[failed] = 0.0
    track_weight = ~failed & mask

    for name, array in (("evals", evals), ("evecs", evecs), ("v1", evecs[..., 0]), ("fa_stop", fa_stop),
                        ("failed", failed), ("track_weight", track_weight)):
        save_float32(array, image.affine, work_path(cfg, "tensor", f"{name}.nii.gz"))


if __name__ == "__main__":
    main(load_config())
