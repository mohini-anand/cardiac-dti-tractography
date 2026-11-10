"""Post-tracking cleaning in one streaming pass over the root tractogram, the uncleaned output of track.py: seed-voxel
filter, seed-side trim of unreliable runs, regional end trim."""
from pathlib import Path

import nibabel as nib
import numpy as np
import pyvista as pv
from dipy.tracking.streamline import set_number_of_points
from nibabel.streamlines import LazyTractogram
from nibabel.streamlines.trk import TrkFile, get_affine_rasmm_to_trackvis, get_affine_trackvis_to_rasmm
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from .common import load_config, work_path

FLOOR_TOL_MM = 1e-3     # a kept part of exactly the minimum length can measure slightly short from float rounding
SIDE_TD_ABS = 0.05      # transmural depth decides the LV/RV side only when at least this far from zero
SAMPLE_TOL_MM = 1.0     # tolerance when sampling the transmural volume at voxel centres


def arc_length(points):
    """Arc length at each point."""
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(np.asarray(points, np.float64), axis=0), axis=1))]


def true_runs(flags):
    """Start and exclusive end of every run of True values."""
    change = np.diff(np.r_[0, np.asarray(flags, bool).astype(np.int8), 0])
    return list(zip(np.where(change == 1)[0], np.where(change == -1)[0]))


def sample_flags(points, flagged, inverse_affine):
    """Arc length, number of samples about 1 mm apart, and each sample's unreliable flag."""
    distance = arc_length(points)
    n = max(int(round(float(distance[-1]))) + 1, 2)
    voxels = np.rint(set_number_of_points(points, n) @ inverse_affine[:3, :3].T + inverse_affine[:3, 3]).astype(np.int64)
    inside = np.all((voxels >= 0) & (voxels < np.array(flagged.shape)), axis=1)
    flags = np.ones(n, bool)
    flags[inside] = flagged[voxels[inside, 0], voxels[inside, 1], voxels[inside, 2]]
    return distance, n, flags


def seed_side_trim(distance, flags, n, seed_point, trigger_run, min_length_mm):
    """Kept point range, or None when the part kept around the seed is too short.

    Unreliable runs touching an end are trimmed; other runs of at least `trigger_run` samples cut and keep the seed's side.
    Shorter inner runs are kept.
    """
    total = float(distance[-1])
    step = total / (n - 1)
    seed_sample = int(np.clip(np.rint(distance[seed_point] / step), 0, n - 1))
    flags = np.array(flags, bool)
    flags[seed_sample] = False
    low, high = 0, n - 1
    for a, b in true_runs(flags):
        if b <= seed_sample:
            if a == 0 or b - a >= trigger_run:
                low = max(low, b)
        elif b == n or b - a >= trigger_run:
            high = min(high, a - 1)
    # Exact output: the 1e-9 margins keep a point lying exactly on a sample position, up to float rounding, inside the
    # kept part.
    start = 0 if low == 0 else int(np.searchsorted(distance, low * step - 1e-9, side="left"))
    stop = len(distance) if high == n - 1 else int(np.searchsorted(distance, high * step + 1e-9, side="right"))
    start, stop = min(start, seed_point), max(start, stop, seed_point + 1)
    kept = float(distance[stop - 1] - distance[start]) if stop - start >= 2 else 0.0
    return None if kept + FLOOR_TOL_MM < min_length_mm else (start, stop)


class RegionalThresholds:
    """Turning-angle threshold by location: near the LV/RV junction line, else in the apical part of the long axis, else
    the rest of the wall, septum included. The junction line runs along the RV insertions on the epicardium (see
    junction_thresholds)."""

    def __init__(self, line_world, axis, s_min, s_max, radius_mm, apex_s_max, wall_deg, junction_deg, apex_deg):
        self.tree = cKDTree(np.asarray(line_world, np.float64))
        self.axis = np.asarray(axis, np.float64) / np.linalg.norm(axis)
        self.s_min, self.s_max, self.radius_mm, self.apex_s_max = float(s_min), float(s_max), float(radius_mm), float(apex_s_max)
        self.wall_deg, self.junction_deg, self.apex_deg = float(wall_deg), float(junction_deg), float(apex_deg)
        self.floor_deg = min(self.wall_deg, self.junction_deg, self.apex_deg)
        self.max_threshold_deg = max(self.wall_deg, self.junction_deg, self.apex_deg)

    def __call__(self, points):
        """Turning-angle threshold, in degrees, at each point."""
        points = np.asarray(points, np.float64)
        s = (points @ self.axis - self.s_min) / (self.s_max - self.s_min)
        in_junction = self.tree.query(points)[0] < self.radius_mm
        region = np.where(in_junction, 0, np.where(s < self.apex_s_max, 1, 2)).astype(np.int8)
        return np.array([self.junction_deg, self.apex_deg, self.wall_deg])[region]


def zone_turns(points, chord, window, zone):
    """Turn angle of each window near the first end, its first kept index and its midpoint.

    A window is two chords of `chord` points whose starts are `window` points apart; windows are tested while the second
    chord's midpoint lies within the first `zone` points. Positions are doubled so the half-point midpoints stay
    integers: (2j + chord + 2 window + 1) // 2 is the first index past the second chord's midpoint, and the window's
    midpoint is the mean of points (2j + chord + window) // 2 and (2j + chord + window + 1) // 2.
    """
    n = len(points)
    last = min((2 * zone - chord - 2 * window) // 2, n - 1 - chord - window)
    if last < 0:
        return np.zeros(0), np.zeros(0, np.int64), np.zeros((0, 3))
    j = np.arange(last + 1)
    a = points[j + chord] - points[j]
    b = points[j + window + chord] - points[j + window]
    cosine = np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    turn = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    first_kept = (2 * j + chord + 2 * window + 1) // 2
    lower, upper = (2 * j + chord + window) // 2, (2 * j + chord + window + 1) // 2
    return turn, first_kept, (points[lower] + points[upper]) / 2


def end_cut(points, chord, window, zone, thresholds):
    """Index of the first point kept at this end: just past the innermost window turning above its threshold."""
    turn, first_kept, middle = zone_turns(points, chord, window, zone)
    violation = turn > float(thresholds.max_threshold_deg)
    # floor_deg, an attribute of the thresholds, is the lowest regional threshold: every window turning more than it is
    # looked up by location, and a window above the highest threshold is a violation wherever it lies.
    candidates = (turn > thresholds.floor_deg) & ~violation
    if candidates.any():
        over = turn[candidates] > np.asarray(thresholds(middle[candidates]), float)
        violation[np.flatnonzero(candidates)[over]] = True
    return int(first_kept[violation].max()) if violation.any() else 0


def end_trim(points, seed_point, chord, window, zone, thresholds, min_length_mm):
    """Kept point range after trimming both ends, or None when the streamline is deleted: what remains is shorter than
    min_length_mm, or the trim would remove the seed."""
    points = np.asarray(points, np.float64)
    n = len(points)
    start = end_cut(points, chord, window, zone, thresholds)
    stop = n - end_cut(points[::-1], chord, window, zone, thresholds)
    distance = arc_length(points)
    kept = float(distance[stop - 1] - distance[start]) if stop - 1 > start else 0.0
    if start == 0 and stop == n:
        return 0, n
    if stop - start < 2 or kept < min_length_mm - FLOOR_TOL_MM:
        return None
    if not start <= seed_point < stop:
        return None
    return start, stop


def junction_thresholds(mask, affine_track, cfg, params):
    """Build the regional thresholds. The LV/RV junction line is where LV-side and RV-side epicardial voxels meet, which
    is along the RV insertions; the side comes from the transmural depth and the surfaces."""
    paths = cfg["paths"]
    transmural = pv.read(paths["transmural_vtk"])
    epi_tree = cKDTree(np.asarray(pv.read(paths["epi_surface"]).points))
    lv_tree = cKDTree(np.asarray(pv.read(paths["lv_endo_surface"]).points))
    rv_tree = cKDTree(np.asarray(pv.read(paths["rv_endo_surface"]).points))
    reference = nib.load(work_path(cfg, "mask", "mask_final.nii.gz"))
    ii, jj, kk = np.where(mask)
    world = (reference.affine[:3, :3] @ np.stack([ii, jj, kk], 1).T.astype(float)).T + reference.affine[:3, 3]
    sampled = pv.PolyData(world.astype(np.float32)).sample(transmural, tolerance=SAMPLE_TOL_MM, snap_to_closest_point=True)
    ok_values = np.asarray(sampled["vtkValidPointMask"]).astype(bool)
    depth_values = np.asarray(sampled["TransmuralDepth"]).astype(np.float32)

    shape = mask.shape
    ok, depth = np.zeros(shape, bool), np.zeros(shape, np.float32)
    d_lv, d_rv, d_epi = np.zeros(shape, np.float32), np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    ok[ii, jj, kk], depth[ii, jj, kk] = ok_values, depth_values
    d_lv[ii, jj, kk] = lv_tree.query(world)[0].astype(np.float32)
    d_rv[ii, jj, kk] = rv_tree.query(world)[0].astype(np.float32)
    d_epi[ii, jj, kk] = epi_tree.query(world)[0].astype(np.float32)

    # The depth sign decides LV or RV except near zero, where the nearer endocardium does.
    epicardial_side = mask & ok & (d_epi < np.minimum(d_lv, d_rv))
    is_rv = ok & np.where(np.abs(depth) >= SIDE_TD_ABS, depth > 0, d_rv < d_lv)
    lv_epi, rv_epi = epicardial_side & ~is_rv, epicardial_side & is_rv
    full = np.ones((3, 3, 3), bool)
    labels, _ = ndi.label(lv_epi, full)
    largest = labels == (int(np.argmax(np.bincount(labels.ravel())[1:])) + 1)
    line = largest & ndi.binary_dilation(rv_epi, full)

    def to_world(voxels):
        return np.argwhere(voxels) @ affine_track[:3, :3].T + affine_track[:3, 3]

    axis = np.asarray(cfg["geometry"]["long_axis"], np.float64)
    axis = axis / np.linalg.norm(axis)
    along = to_world(mask) @ axis
    return RegionalThresholds(to_world(line), axis, float(along.min()), float(along.max()), params["junction_radius_mm"],
                              params["apex_s_max"], params["wall_deg"], params["junction_deg"], params["apex_deg"])


def main(cfg):
    params = cfg["cleaning"]
    min_length_mm = cfg["tracking"]["min_length_mm"]
    chord, window, zone = params["window_points"]      # 0.5 mm chord, 2 mm window, 10 mm end zone at the 0.1 mm step
    root_path = work_path(cfg, "track", "tractogram.trk")
    seed_file = np.load(work_path(cfg, "track", "seeds.npz"))
    seed_ijk = seed_file["streamline_seed_ijk"].astype(np.int64)
    seed_xyz = seed_file["streamline_seed_xyz"].astype(np.float64)
    header = nib.streamlines.load(str(root_path), lazy_load=True).header
    affine_track = np.asarray(header["voxel_to_rasmm"], float)
    inverse_track = np.linalg.inv(affine_track)

    unreliable = np.asarray(nib.load(work_path(cfg, "fill", "unreliable_unfilled.nii.gz")).dataobj) > 0
    mask = np.asarray(nib.load(work_path(cfg, "mask", "mask_final.nii.gz")).dataobj) > 0
    flagged = unreliable | ~mask          # outside the final mask counts as unreliable
    thresholds = junction_thresholds(mask, affine_track, cfg, params)

    # Exact output: an emulated .trk round trip. The root tractogram (the uncleaned output of track.py) has been saved
    # and read back once. A save converts each point from world coordinates to float32 voxel coordinates and a reload
    # converts it back; each point is given one more such conversion here, in memory, and the trims are decided on these
    # points. The writer converts the kept points the same way. This reproduces a computation that saved and reloaded
    # the tractogram twice. The trims threshold angles and lengths, so they depend on these last bits.
    to_ras = get_affine_trackvis_to_rasmm(header).astype(float)
    to_voxel = get_affine_rasmm_to_trackvis(header).astype(float)

    def voxel_from(ras):
        return (np.asarray(ras) @ to_voxel[:3, :3].T + to_voxel[:3, 3]).astype("<f4")

    def ras_from(voxel):
        return np.asarray(voxel, float) @ to_ras[:3, :3].T + to_ras[:3, 3]

    n_roots = int(header["nb_streamlines"])
    kept_rows = np.zeros(n_roots, bool)

    def cleaned():
        # The writer calls this twice (it peeks at the first item), so kept flags are per row and idempotent.
        for row, streamline in enumerate(nib.streamlines.load(str(root_path), lazy_load=True).streamlines):
            kept_rows[row] = False
            if unreliable[tuple(seed_ijk[row])]:      # seed-voxel filter
                continue
            points = ras_from(voxel_from(np.asarray(streamline, float)))
            seed = seed_xyz[row]
            seed_point = int(np.argmin(np.linalg.norm(points - seed, axis=1)))
            distance, n, flags = sample_flags(points, flagged, inverse_track)
            trimmed = seed_side_trim(distance, flags, n, seed_point, params["trigger_run"], min_length_mm)
            if trimmed is None:
                continue
            part = points[trimmed[0]:trimmed[1]]
            seed_point = int(np.argmin(np.linalg.norm(part - seed, axis=1)))
            ends = end_trim(part, seed_point, chord, window, zone, thresholds, min_length_mm)    # after the seed-side trim, always
            if ends is None:
                continue
            kept_rows[row] = True
            yield part[ends[0]:ends[1]]

    results = Path(cfg["paths"]["results"])
    results.mkdir(parents=True, exist_ok=True)
    TrkFile(LazyTractogram(cleaned, affine_to_rasmm=np.eye(4)), header=header.copy()).save(str(results / "tractogram.trk"))
    index = np.nonzero(kept_rows)[0]
    np.savez_compressed(results / "seeds.npz", input_seeds=seed_file["input_seeds"],
                        streamline_seed_index=seed_file["streamline_seed_index"][index],
                        streamline_seed_xyz=seed_file["streamline_seed_xyz"][index])


if __name__ == "__main__":
    main(load_config())
