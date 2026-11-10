"""Convert the DICOM series to NIfTI, average the diffusion repeats and put the manual mask on the diffusion grid."""
import subprocess
from pathlib import Path

import nibabel as nib
import nrrd
import numpy as np
from nibabel.processing import resample_from_to

from .common import load_config, repeat_names, work_dir, work_path


def dcm2niix(dicom_dir, out_dir, name):
    """Run dcm2niix on one series: compressed output, no JSON sidecar, overwrite earlier results."""
    subprocess.run(["dcm2niix", "-z", "y", "-b", "n", "-w", "1", "-f", name, "-o", str(out_dir), str(dicom_dir)],
                   check=True, capture_output=True)


def convert_diffusion(cfg):
    """Write one NIfTI per repeat and the voxelwise average of the repeats."""
    dicom_root = Path(cfg["paths"]["dwi_dicom"])
    raw_dir = work_dir(cfg, "convert", "raw")
    names = repeat_names(cfg)
    for name in names:
        dcm2niix(dicom_root / name, raw_dir, name)

    # The first series in sorted order supplies the header and the gradient table.
    first = nib.load(raw_dir / f"{names[0]}.nii.gz")
    stack = np.stack([np.asarray(nib.load(raw_dir / f"{n}.nii.gz").dataobj, dtype=np.float32) for n in names])
    average = stack.mean(axis=0).astype(np.float32)
    # Exact output: the average is written as int16, the data type of the dcm2niix header it reuses. nibabel resets that
    # header's scale factor and, when saving, fits a new slope and intercept to the average's own range (a step of about
    # 0.031 on this heart), so the saved average is quantised to that step. The T1 registration reads this file and its
    # result depends on these exact values. Maps that need full precision, such as the tensor outputs, are written as
    # float32 instead (common.save_float32).
    nib.save(nib.Nifti1Image(average, first.affine, first.header), work_path(cfg, "convert", "dwi_avg.nii.gz"))
    bvals = np.loadtxt(raw_dir / f"{names[0]}.bval")
    bvecs = np.loadtxt(raw_dir / f"{names[0]}.bvec")
    # Exact output: keep these text formats; the tensor fit and the bootstrap read the gradient table from these files,
    # at this precision.
    np.savetxt(work_path(cfg, "convert", "dwi.bval"), bvals[None, :], fmt="%g")
    np.savetxt(work_path(cfg, "convert", "dwi.bvec"), bvecs, fmt="%.6f")


def nrrd_affine(header):
    """Index-to-world affine in RAS from an NRRD header stored in LPS."""
    directions = np.asarray(header["space directions"], float)
    origin = np.asarray(header["space origin"], float)
    space = header.get("space", "left-posterior-superior").lower()
    # Latent: only a space value starting with "left-posterior" is converted from LPS. Any other value, such as the
    # abbreviation "LPS" or "left-anterior-superior", is read as RAS, and the mask would be placed mirrored. This
    # heart's mask is stored as "left-posterior-superior".
    flip = np.diag([-1.0, -1.0, 1.0]) if space.startswith("left-posterior") else np.eye(3)
    affine = np.eye(4)
    affine[:3, :3] = flip @ directions.T
    affine[:3, 3] = flip @ origin
    return affine


def resample_manual_mask(cfg):
    """Move the manual myocardial mask onto the diffusion grid by world coordinates, nearest neighbour."""
    dwi = nib.load(work_path(cfg, "convert", "dwi_avg.nii.gz"))
    data, header = nrrd.read(cfg["paths"]["manual_mask"])
    resampled = resample_from_to(nib.Nifti1Image(data.astype(np.int16), nrrd_affine(header)),
                                 (dwi.shape[:3], dwi.affine), order=0)
    mask = np.asarray(resampled.dataobj) > 0
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), dwi.affine, dwi.header),
             work_path(cfg, "convert", "myo_mask_dwi.nii.gz"))


def main(cfg):
    convert_diffusion(cfg)
    out_dir = work_dir(cfg, "convert")
    # The names give this dataset's sequences: a 3-D FLASH T1-weighted and a 3-D TSE T2-weighted series.
    dcm2niix(cfg["paths"]["t1_dicom"], out_dir, "T1_FLASH")
    dcm2niix(cfg["paths"]["t2_dicom"], out_dir, "T2_TSE")
    resample_manual_mask(cfg)


if __name__ == "__main__":
    main(load_config())
