"""Settings loading and NIfTI helpers shared by the pipeline steps."""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml


def load_config():
    """Read the YAML settings file named on the command line."""
    with open(sys.argv[1], encoding="ascii") as f:
        return yaml.safe_load(f)


def work_path(cfg, *parts):
    """Path of an intermediate file under the work folder, creating its parent folder."""
    path = Path(cfg["paths"]["work"]).joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def work_dir(cfg, *parts):
    """A folder under the work folder, created if needed."""
    path = Path(cfg["paths"]["work"]).joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def repeat_names(cfg):
    """The diffusion repeats: the subfolders of the diffusion DICOM folder, in sorted order. Conversion and registration
    both take the repeats from this list, so no other file in the work folder can become a repeat."""
    return sorted(p.name for p in Path(cfg["paths"]["dwi_dicom"]).iterdir() if p.is_dir())


def load_array(path, dtype=np.float32):
    """Voxel array of a NIfTI file with the requested data type."""
    return np.asarray(nib.load(path).dataobj, dtype)


def save_float32(array, affine, path):
    """Write float32 with a fresh header; a reference header's int16 type would quantise the data."""
    affine = np.asarray(affine, float)
    image = nib.Nifti1Image(np.asarray(array, np.float32), affine)
    image.set_data_dtype(np.float32)
    image.header.set_slope_inter(1.0, 0.0)
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    nib.save(image, path)
