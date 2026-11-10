# Cardiac DTI tractography from raw DICOM

Code that turns the raw diffusion and anatomical DICOM series of one ex vivo porcine heart into a deterministic
tensor tractogram of the myocardium. The fibre direction used for tracking is the primary eigenvector of the diffusion
tensor; there is no multi-fibre model and no probabilistic tracking.

The pipeline is tuned to one specimen. It is a record of how one tractogram was made, not a general tool.

## Data

The diffusion, T1-weighted and T2-weighted images are from the
[Ex Vivo Porcine Heart DT MRI Data](https://med.stanford.edu/cmrgroup/data/ex_vivo_dt_mri.html), a collaboration
between the CMR Group at Stanford University and the Cardiac Biomechanics Lab at UCSF. The dataset page does not
identify its hearts individually. The series used are the five repeats of the RESOLVE (readout-segmented EPI)
diffusion series, the 3-D FLASH T1-weighted series and the 3-D TSE T2-weighted series. The thresholds in
`config.yaml` were set on these images; images from other sequences would need other values.

The myocardial mask, the segmentation, the three surfaces and the transmural volume were made by the author of this
repository for this heart. They are not part of the Stanford download. None of the data is included in this
repository.

## Required inputs

Paths are set in `config.yaml` and are relative to the repository root, the folder the pipeline must be started from.
The defaults expect the layout below under `data/`. Links to the files, or a single link to a folder that holds them,
are fine; or the paths in `config.yaml` can be changed.

| Path in `config.yaml` | Input | Used for |
|---|---|---|
| `data/dwi_dicom/` | Five repeated diffusion DICOM series, one subfolder per repeat and nothing else. Each repeat has 31 volumes: the b0 first, then 30 directions at b = 1000 s/mm². | Phase-encode registration, denoising, tensor fit, attenuation peel, bootstrap |
| `data/t1_dicom/` | 3-D FLASH T1-weighted DICOM series | Registration to the b0, epicardial trace, T1 peel, LV hull, fluid peel, RV trabecular removal |
| `data/t2_dicom/` | 3-D TSE T2-weighted DICOM series | Epicardial trace, fluid peel, RV trabecular removal |
| `data/myocardium_mask.nrrd` | Manual myocardial mask (non-zero is myocardium) | Region of interest and starting mask |
| `data/segmentation.nii.gz` | Label volume: 0 background, 1 myocardium, 2 cavity | LV and RV cavities (only the cavity label is read) |
| `data/transmural_depth.vtk` | Volume mesh with a `TransmuralDepth` point array | LV/RV side of each voxel (T1 peel), LV/RV junction line (end trim) |
| `data/epicardium.vtk`, `data/lv_endocardium.vtk`, `data/rv_endocardium.vtk` | Epicardial, LV and RV endocardial surfaces | LV/RV junction line (end trim) |

The mask, segmentation, surfaces and transmural volume are not produced by this pipeline, and this repository does not
describe how they were made. The code assumes the following and checks none of it:

- Every one of these hand-made inputs is already aligned with the diffusion images in world coordinates; only the T1
  image is registered. The mask and segmentation may be on other grids: they are resampled by world coordinates. Points
  in the `.vtk` files are read as world coordinates in mm, in the frame of the diffusion NIfTI's affine (RAS). The
  `.nrrd` mask is converted from LPS to RAS.
- `TransmuralDepth` is a Laplace solution that is −2 on the LV endocardium, +1 on the RV endocardium and 0 on the
  epicardium; in this file it ranges from −2 to 1. Its sign gives the side, negative for the LV and positive for the
  RV. Across the septum its zero level is the LV/RV divide, with about two thirds of the septal wall on the LV side.
  Near the epicardium it is close to 0 on both sides: 54% of its points have |depth| below 0.05. The side map of the
  mask step labels depth below 0 as LV and everything else as RV. The end trim uses the sign only where |depth| is at
  least 0.05; nearer zero, the nearer endocardial surface decides.
- The cavity label holds the LV and RV cavities as two separate connected components. The two largest components are
  used, and the one whose wall is thicker is taken as the LV.
- T1 and T2 share one voxel grid and are aligned in scanner space: the T2 image has no registration of its own and is
  moved with the T1 transform, and the T2 slab-join correction reads a T1 tissue mask on the T2 grid.

Geometry assumptions built into the code:

- The slices run along array axis 2 and lie roughly perpendicular to the heart's long axis. The epicardial trace works
  slice by slice in polar coordinates, and the ring tests are in-plane.
- The base is at the highest slice index and ends in a flat cut face.
- Slice numbers in `config.yaml` are 0-based array indices on the diffusion grid, except `t2_slab_joins`, which are on
  the T2 image's own grid.

Acquisition assumptions built into the code:

- 1 mm isotropic diffusion voxels. Several settings in mm are compared directly with distances in voxels.
- The b0 is volume 0 of every repeat, and all repeats share one gradient table (taken from the first series in sorted
  order).
- The diffusion series is 2-D multi-slice. Its phase encoding runs along the image columns, the scanner's
  anterior-posterior axis, which is array axis 1 after conversion (`registration.pe_axis`).
- The diffusion series has a full, symmetric k-space window, with no partial Fourier, which the Gibbs-ringing removal
  assumes: the acquired and reconstructed phase-encode matrices are both 128 at 100% sampling (with parallel imaging,
  factor 2), and dcm2niix v1.0.20241211 reported `PartialFourier` 1. The version listed under Install does not write
  that field.
- The T1 and T2 series are 3-D acquisitions made in slabs and stored as one 240-slice volume each, on the same 0.5 mm
  grid. Only T2 is corrected for its intensity steps at the slab joins inside the heart, between its slices 39 and 40
  and between 79 and 80 (`mask.images.t2_slab_joins` names the first slice above each join).
- dcm2niix converts each diffusion subfolder to exactly one `<subfolder>.nii.gz`.

## Install

- Python 3.11, in a new virtual environment made with a Python 3.11 interpreter: `python3.11 -m venv .venv` (or the
  interpreter's full path, if `python3.11` is not on the `PATH`), then `source .venv/bin/activate`.
- `pip install -r requirements.txt`. The packages the code imports are pinned; the packages they depend on in turn are
  not. The diffusion processing uses DIPY (Garyfallidis et al. 2014).
- [dcm2niix](https://github.com/rordenlab/dcm2niix) (Li et al. 2016) on the `PATH`, tested with v1.0.20261029. The
  code relies on its output file names and its gradient-direction convention.
- macOS or Linux. The bootstrap uses a `fork` worker pool, which Windows does not provide.

## Run

From the repository root, with the virtual environment active:

```
bash run.sh [config]
KEEP_WORK=1 bash run.sh [config]    # the same, keeping the intermediate files
```

The config file defaults to `config.yaml`. `run.sh` first deletes the work folder named by `paths.work` (by default
`work/`), so no file from an earlier run is reused, and stops if that setting is empty. It then runs every step in
order and leaves two files:

- `results/tractogram.trk`: the final streamlines.
- `results/seeds.npz`: `input_seeds`, every seed; `streamline_seed_index`, the row of `input_seeds` that seeded each
  streamline; `streamline_seed_xyz`, that seed's coordinates. Seeds are world coordinates in mm, in the frame of the
  diffusion NIfTI's affine, the same frame as the streamline points when the `.trk` file is loaded with nibabel.

Streamlines shorter than 20 mm are dropped when tracking (`tracking.min_length_mm`). A streamline of exactly 20 mm can
measure a few millionths of a millimetre less when its single-precision points are read back.

At the end, `run.sh` deletes the work folder again unless `KEEP_WORK=1` is set. With `KEEP_WORK=1`, `work/` keeps every
intermediate file, among them:

- the final and seed masks (`work/mask/`);
- the tensor maps (`work/tensor/`): eigenvalues, eigenvectors, V1, the failed-fit map, the tracking-weight map and
  `fa_stop`, which is FA with failed and non-finite fits set to 0. No other FA map is computed; the fill step writes an
  identical copy of `fa_stop` into `work/fill/`;
- the reliability and filled maps, and the maps tracking reads (`work/fill/`);
- the root tractogram, the uncleaned output of tracking (`work/track/`).

Besides `work/` and `results/`, Python writes its usual `__pycache__` folders under `src/`. The `.gitignore` excludes
these, the data folder and a local virtual environment.

One step can be run alone with `python -m src.<step> config.yaml`, also from the repository root. It reads the earlier
steps' outputs from the work folder, so it works only after a run with `KEEP_WORK=1`; the next `bash run.sh` deletes
them.

A full run from DICOM took about 21 minutes on an 18-core, 48 GB Apple silicon Mac. Per step: conversion under
1 minute, T1 registration about 2 minutes, phase-encode registration under 1 minute, denoising about 9 minutes, tensor
fit under 1 minute, mask refinement about 1 minute, fill about 1 minute, tracking about 4 minutes and cleaning about
2.5 minutes. The work and results folders together need about 23 GB of free disk. The largest single process, the
denoising, used about 5 to 7 GB of memory, depending on the run. Counting the denoising's worker processes, one run
peaked at about 9 to 15 GB in all.

As a check of a run, runs on that Mac gave a final mask of 152,732 voxels (`work/mask/mask_final.nii.gz`), 988,370
streamlines in the root tractogram (`work/track/tractogram.trk`) and 962,264 in the final tractogram. The first two
files are kept only with `KEEP_WORK=1`.

A run prints a few library warnings that are expected: DIPY notices about positional arguments during registration, and
NumPy numerical warnings during mask refinement. Apart from these it prints nothing, so it can look idle for minutes
at a time.

Results differ slightly between computers, mainly through the T1 registration. The same DICOM was processed on an
Intel Mac and on the Apple silicon Mac above. The T1 transforms found on the two machines differed by 0.09 mm on
average over the myocardium (0.28 mm at most), and that difference alone changes 2,345 voxels (about 1.5%) of the
final mask. A run entirely on the Apple silicon Mac kept 962,264 streamlines. A run whose registration, denoising,
tensor fit and bootstrap came from the Intel Mac, with the other steps on the Apple silicon Mac, kept 966,663.

## What each step does

1. **`convert`.** Converts the DICOM series to NIfTI with dcm2niix, averages the five diffusion repeats for the T1
   registration, and moves the manual mask onto the diffusion grid by world coordinates. The repeats are the
   subfolders of the diffusion DICOM folder in sorted order; registration lists them the same way.
2. **`register`.** Aligns the T1 image to the mean b0 (volume 0 of the averaged diffusion series) by mutual information
   (centre of mass, translation, rigid, affine), then registers the native T1 to that aligned copy once more and keeps
   the affine found. The mask step uses the aligned copy for the T1 peel, the LV hull and the RV trabecular removal,
   and the second affine to resample T1 and T2 onto its 0.5 mm grid. Both affines map diffusion-frame world
   coordinates to T1 world coordinates (DIPY's convention), so resampling with either brings the T1 into the diffusion
   frame. Using the second affine rather than the registration's own is an incidental detail kept so that the result
   reproduces exactly: measured on this heart, the registration's own affine changes 165 voxels of the final mask.
   The step then estimates one shift along the phase-encode axis for every volume of every repeat, against the mean b0
   and against the same volume of the other repeats, and removes it with a sub-voxel Fourier shift. Nothing else is
   corrected: no rotation, no shift along the other axes, and no eddy-current scaling or shear.
3. **`denoise`.** Applies MP-PCA (Veraart et al. 2016) to all 155 volumes of the registered repeats, averages them per
   volume, and removes Gibbs ringing (Kellner et al. 2016). MP-PCA runs before averaging so it sees the redundancy of
   the five repeats.
4. **`tensor`.** Fits the diffusion tensor by weighted least squares. A fit is marked failed by any of six tests on the
   data and on an unclipped re-solve of the same estimator; the tests can overlap. Failed fits get FA 0 in `fa_stop`
   and no direction unless the fill step gives them one.
5. **`mask`.** Refines the manual mask in two parts. The first, the epicardial refinement: a per-slice minimum-cost
   path traces the epicardium on the T1 and T2 images; surface-connected low-T1 and non-attenuating voxels are peeled;
   papillary and trabecular muscle is excluded by two-sided enclosure and by a smooth LV hull and an RV hull; and
   surface-connected voxels at least half fluid are removed. After each peel, and once after the exclusions and the
   fluid peel, a ring guard restores a minimal path of removed tissue where the step opened the in-plane myocardial
   ring. The slices above the basal cut (`basal_cut_last_slice`) are then removed, and the remaining gaps in the ring
   are bridged; bridge voxels complete the wall but are never seeded. The second part removes RV trabecular and
   papillary tissue attached to enclosed RV structures, without thinning the RV wall below `floor_mm` and without
   touching the bridges.
6. **`fill`.** Estimates the reliability of each voxel's fibre direction as a 95% cone of uncertainty (Jones 2003),
   from a 500-resample bootknife (Hesterberg 2004) of the registered repeats. Unreliable or failed voxels are filled
   from reliable neighbours by a log-Euclidean mean (Arsigny et al. 2006) where the support is adequate and coherent.
   Unreliable voxels that are not filled keep their direction and are tracked through; the cleaning step treats them
   separately.
7. **`track`.** Tracks deterministically along the primary eigenvector, from a regular grid of eight seeds per voxel,
   with a 0.1-voxel step. The direction getter is DIPY's EuDX-style one (Garyfallidis 2012). At each step it blends the
   eight voxels around the current point by trilinear weight: a voxel contributes its eigenvector, flipped to agree with
   the direction of travel, if it has one and it lies within 30 degrees of that direction; other voxels contribute
   nothing. Each voxel's eigenvector is used as computed, not rounded to a set of directions. A streamline ends where
   less than half of the trilinear weight contributes (`tracking.total_weight`), or where it leaves the region it may
   track in: the final mask without its very low-FA voxels (FA below 0.02, including failed fits that the fill step did
   not fill) that connect to the mask border through other such voxels. Low-FA voxels enclosed in the wall are holes in the field and are kept in
   that region.
8. **`clean`.** Deletes every streamline seeded in an unreliable, unfilled voxel. On samples about 1 mm apart, a sample
   is unreliable if it lies in an unreliable, unfilled voxel or outside the final mask: runs of unreliable samples that
   touch either end are trimmed off, and at an inner run of 3 or more samples (`cleaning.trigger_run`, about 2 mm) the
   streamline is cut and the part holding the seed is kept. Then each end is trimmed back past the innermost sharp turn
   within its last 10 mm; the turning-angle threshold depends on where the turn is (near the LV/RV junction line, in
   the apical part, or elsewhere in the wall). A streamline is deleted when what remains is shorter than 20 mm, or when
   the end trim would remove its seed.

## Comment markers in the code

Two markers in the comments flag lines that should not be changed casually:

- **"Exact output:"** marks a line that keeps an incidental detail of the original computation (a data type, a
  rounding, an order of operations, a call whose result is unused) so that the result reproduces exactly. These are
  not design choices, and not claims that the choice is better. Several steps threshold continuous values
  (intensities, distances, angles), so a change in the last bits of an input can move a voxel or a streamline end
  across a threshold; an equivalent-looking rewrite at these lines would change the mask or the tractogram.
- **"Latent:"** marks a condition under which the code would give a wrong result, or stop with an error, on other data,
  without a check or a warning of its own. Each says what would go wrong and that it does not affect the result on
  this heart.
  They are documented, not fixed.

## Values set for this heart

These values in `config.yaml` were set for this heart; they do not transfer to another heart, and this repository
gives no code to re-derive them:

- epicardial trace: the trace scales (`trace_scales`, below) and the first slice with groove wedges (23);
- images: the T2 slab joins (slices 40 and 80 of the T2 grid) and the T1 tissue range used to measure them (150 to
  400);
- peels: the T1 peel thresholds (LV 130, RV 80) and the attenuation limit (5);
- hulls: the dark-tissue T1 limit (80), the hull cavity opening (0 mm), and the width (3 mm) and long-axis range (0.25
  to 0.75) of the LV hull's protected band;
- RV trabecular removal: the RV wall floor (2.5 mm);
- the last basal slice kept (69), the long axis and the cavity label;
- registration: the b0 level above which the matching region counts a voxel as fluid (300);
- fill: the cone-of-uncertainty trigger (30 degrees);
- end trim: the turning-angle thresholds (35, 39.4 and 31.7 degrees), the junction radius (2.5 mm) and the apical part
  (long-axis position below 0.25).

The constants at the top of the modules, such as those in `src/mask/fluid_peel.py`, `exclusion.py`, `hull.py`,
`rv_trabeculae.py`, `bridge.py`, `chain.py`, `epi_trace.py` and `images.py`, were also set during development on this
heart.

The four trace scales were measured once, on the bias-normalised images on the 0.5 mm grid, at the lower half of five
development slices (slices 37, 48, 54, 60 and 70). `tau` is the Otsu threshold of T1 inside the manual mask, over
voxels with T1 above 0.25 (void excluded); `tau2` is the Otsu threshold of T2 over the same voxels with T2 below 3;
`w2` is the robust standard deviation (1.4826 times the median absolute deviation) of T2 at least 3 mm in-plane
inside the manual mask; and `s1_per_mm` is five times the robust standard deviation, over those deep voxels, of the
T1 derivative per mm along the two in-plane axes (Gaussian derivative with `SIG_D_MM`). The code that measured them is
not part of this repository.

## Limits

- Tuned to one ex vivo porcine heart (see "Values set for this heart").
- This is not a general tool: it assumes this acquisition, these file layouts and this mask.
- The epicardial trace, the restorable map of the ring guards and the bridges read only the lower 0.5 mm half of each
  1 mm slice, which under-represents voxels where the wall changes within a slice, such as near the apex. The other
  steps on the 0.5 mm grid use both halves: the bias normalisation, the groove channel's 3-D connectivity, the fluid
  peel and the RV trabecular removal.
- Conditions under which the code would go wrong on other data, none of which arises on this heart, are marked
  "Latent:" at the lines concerned.
- The output has not been validated as anatomically faithful.
- Helix-angle quality control is being packaged and is not included yet.

## References

- Arsigny V, Fillard P, Pennec X, Ayache N. Log-Euclidean metrics for fast and simple calculus on diffusion tensors.
  Magn Reson Med 2006;56:411-421.
- Garyfallidis E. Towards an accurate brain tractography. PhD thesis, University of Cambridge, 2012.
- Garyfallidis E, Brett M, Amirbekian B, Rokem A, van der Walt S, Descoteaux M, Nimmo-Smith I. Dipy, a library for the
  analysis of diffusion MRI data. Front Neuroinform 2014;8:8.
- Hesterberg TC. Unbiasing the bootstrap: bootknife sampling vs. smoothing. Proceedings of the Section on Statistics &
  the Environment, American Statistical Association, 2004.
- Holz M, Heil SR, Sacco A. Temperature-dependent self-diffusion coefficients of water and six selected molecular liquids
  for calibration in accurate 1H NMR PFG measurements. Phys Chem Chem Phys 2000;2:4740-4742.
- Jones DK. Determining and visualizing uncertainty in estimates of fiber orientation from diffusion tensor MRI. Magn
  Reson Med 2003;49:7-12.
- Kellner E, Dhital B, Kiselev VG, Reisert M. Gibbs-ringing artifact removal based on local subvoxel-shifts. Magn Reson
  Med 2016;76:1574-1581.
- Li X, Morgan PS, Ashburner J, Smith J, Rorden C. The first step for neuroimaging data analysis: DICOM to NIfTI
  conversion. J Neurosci Methods 2016;264:47-56.
- Veraart J, Novikov DS, Christiaens D, Ades-Aron B, Sijbers J, Fieremans E. Denoising of diffusion MRI using random
  matrix theory. NeuroImage 2016;142:394-406.

## License

BSD 3-Clause; see `LICENSE`.
