"""Tangent-continuation bridges: where a peel cut the in-plane myocardial ring, connect the two ends along the wall.

Each cut is found as the shortest path from the cavity to the slice border through the complement of the mask. Its two
ends are joined by a cubic Hermite curve along their tangents, or by a path that follows the depth from the endocardial
surface when the curve fails. Bridge voxels complete the wall and are never used as seeds.

Here a side's face is the set of its voxels that touch the tube around the cut, not the face of epi_trace.py. Like the
trace, the bridges read the endocardial surface at sub-slice 2k (the lower half) of each 1 mm slice; their other inputs
are on the 1 mm grid.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.graph import MCP_Geometric

from .chain import S4, S8

TUBE_LADDER_MM = (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)  # tube radii tried after the configured one
HERMITE_MIN_COS = 0.3


def inplane_mean(fine):
    """Mean of the four in-plane sub-voxels at sub-slice 2k (the lower half-slice) of each 1 mm voxel."""
    return sum(fine[a::2, b::2, ::2] for a in (0, 1) for b in (0, 1)) / 4


def voxel_classes(endo):
    """All-four-in-endocardial-surface flag and mean distance to that surface, per 1 mm voxel."""
    inside = np.ones(tuple(s // 2 for s in endo.shape), bool)
    for a in (0, 1):
        for b in (0, 1):
            inside &= endo[a::2, b::2, ::2]
    return inside, inplane_mean(ndi.distance_transform_edt(~endo, sampling=0.5))


def _principal_axis(points):
    """Main direction of a point set and the ratio of its two largest singular values."""
    centre = points.mean(0)
    _, s, vt = np.linalg.svd(points - centre, full_matrices=False)
    return vt[0], (s[0] / max(s[1], 1e-6)) if len(s) > 1 else np.inf


def _exterior_labels(mask, structure):
    labels, _ = ndi.label(~mask, structure=structure)
    border = np.zeros_like(mask)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    exterior = np.unique(labels[border & ~mask])
    return labels, exterior[exterior > 0]


def _is_open(mask, inner, structure):
    """Whether any of the inner region connects to the slice border through the complement of the mask."""
    labels, exterior = _exterior_labels(mask, structure)
    return bool(np.isin(labels[inner & ~mask], exterior).any())


class Bridger:
    """Finds the cuts in the in-plane myocardial ring of each slice and the voxels that bridge them."""

    def __init__(self, peeled, traced, manual, cavity, endo_strict, endo_distance, params):
        self.peeled, self.traced, self.manual, self.cavity = peeled, traced, manual, cavity
        self.endo_distance = endo_distance
        self.allowed = traced & ~endo_strict
        self.params = params
        self.shape = peeled.shape[:2]

    def inner_of(self, k):
        """The region the manual mask encloses in-plane on slice k, plus the cavity labels."""
        slice_manual = self.manual[:, :, k]
        labels, exterior = _exterior_labels(slice_manual, S8)
        return (~slice_manual & ~np.isin(labels, exterior)) | self.cavity[:, :, k]

    def throat(self, mask, traced, inner):
        """Voxels of the trace mask on the cheapest path from the inner region to the border."""
        complement = ~mask
        cost = np.where(complement, 1.0, np.inf)
        cost = np.where(complement & traced, 4.0, cost)
        sources = np.argwhere(inner & complement)
        if len(sources) == 0:
            return None
        router = MCP_Geometric(cost, fully_connected=True)
        distance, _ = router.find_costs([tuple(s) for s in sources])
        border = np.zeros_like(mask)
        border[[0, -1], :] = True
        border[:, [0, -1]] = True
        targets = np.argwhere(border & complement & np.isfinite(distance))
        if len(targets) == 0:
            return None
        target = targets[np.argmin(distance[targets[:, 0], targets[:, 1]])]
        path = np.array(router.traceback(tuple(target)))
        return path[traced[path[:, 0], path[:, 1]]]

    def find_cuts(self, k):
        """Cuts of one slice. A throat is found and plugged, up to 12 times, until the ring is closed; a throat
        within MERGE_MM of an earlier cut joins it. Returns the merged cuts, the first throat of each (find_ends takes
        the cut's ends and tangents from it) and the inner region."""
        inner = self.inner_of(k)
        mask = self.peeled[:, :, k].copy()
        raw = []
        # Latent: a slice that needs more than 12 throats keeps its remaining openings without a message. On this heart a
        # slice needs at most 3.
        for _ in range(12):
            if not _is_open(mask, inner, S8):
                break
            found = self.throat(mask, self.traced[:, :, k], inner)
            if found is None or len(found) == 0:
                break
            raw.append(found)
            mask[found[:, 0], found[:, 1]] = True
        merged, first = [], []
        for found in raw:
            hit = None
            for i, cut in enumerate(merged):
                if np.min(np.linalg.norm(cut[:, None, :] - found[None, :, :], axis=-1)) <= self.params["MERGE_MM"]:
                    hit = i
                    break
            if hit is None:
                merged.append(found)
                first.append(found)
            else:
                merged[hit] = np.unique(np.vstack([merged[hit], found]), axis=0)
        return merged, first, inner

    @staticmethod
    def iso_tangent(distance_slice, p):
        """In-plane direction perpendicular to the smoothed gradient of the distance field at point p."""
        gradient = np.array(np.gradient(ndi.gaussian_filter(distance_slice, 1.0)))[:, p[0], p[1]]
        normal = gradient / max(np.linalg.norm(gradient), 1e-9)
        return np.array([-normal[1], normal[0]])

    def find_ends(self, peeled_slice, throat_points, distance_slice):
        """The two sides of a cut, an end point and a tangent on each; None when fewer than two sides are found.

        The cut's throat is widened into a tube; the two largest pieces of the mask touching the tube are its sides, and a
        side's face is its voxels touching the tube. A side with a small face or an elongated shape counts as thin.
        """
        P = self.params
        ii, jj = np.mgrid[:self.shape[0], :self.shape[1]]
        mid = throat_points.mean(0)
        window = np.hypot(ii - mid[0], jj - mid[1]) <= P["WIN_MM"]
        ids, labels, touching = [], None, None
        for radius in (P["TUBE_MM"],) + TUBE_LADDER_MM:
            tube = np.zeros_like(peeled_slice)
            for t in throat_points:
                tube |= np.hypot(ii - t[0], jj - t[1]) <= radius
            local = peeled_slice & window & ~tube
            labels, _ = ndi.label(local, structure=S8)
            touching = ndi.binary_dilation(tube, structure=S8) & local
            ids = np.unique(labels[touching])
            ids = ids[ids > 0]
            if len(ids) >= 2:
                break
        if len(ids) < 2:
            return None
        sizes = {i: int((labels == i).sum()) for i in ids}
        ids = sorted(ids, key=lambda i: -sizes[i])[:2]
        sides = []
        for i in ids:
            side = labels == i
            face = np.argwhere(side & touching)
            voxels = np.argwhere(side)
            near = voxels[np.linalg.norm(voxels - face.mean(0), axis=1) <= P["TAN_MM"]]
            direction, anisotropy = _principal_axis(near) if len(near) >= 3 else (None, 0.0)
            sides.append(dict(side=side, face=face, face_n=int(len(face)), direction=direction, aniso=float(anisotropy),
                              near_n=int(len(near)), face_depth=float(np.median(distance_slice[face[:, 0], face[:, 1]]))))
        thin = [s["face_n"] <= P["THIN_FACE"] or s["aniso"] >= P["ANISO"] for s in sides]
        order = (1, 0) if (thin[1] and not thin[0]) else (0, 1)
        side_a, side_b = sides[order[0]], sides[order[1]]
        thin_a, thin_b = thin[order[0]], thin[order[1]]

        def face_point(side, target):
            return side["face"][np.argmin(np.abs(distance_slice[side["face"][:, 0], side["face"][:, 1]] - target))]

        if thin_a == thin_b:
            target = 0.5 * (side_a["face_depth"] + side_b["face_depth"])
            point_a, point_b = face_point(side_a, target), face_point(side_b, target)
        else:
            face = side_a["face"]
            point_a = face[np.argmin(np.linalg.norm(face - face.mean(0), axis=1))]
            point_b = face_point(side_b, float(distance_slice[point_a[0], point_a[1]]))

        def tangent(side, point, is_thin, other):
            reliable = side["direction"] is not None and side["near_n"] >= 6 and side["aniso"] >= 1.5
            t = side["direction"] if (is_thin and reliable) else self.iso_tangent(distance_slice, point)
            return -t if np.dot(t, point - other) < 0 else t

        return dict(a=point_a, b=point_b, ta=tangent(side_a, point_a, thin_a, point_b),
                    tb=tangent(side_b, point_b, thin_b, point_a), depth_a=float(distance_slice[point_a[0], point_a[1]]),
                    depth_b=float(distance_slice[point_b[0], point_b[1]]), side_a=side_a["side"], side_b=side_b["side"])

    @staticmethod
    def hermite(a, b, ta, tb):
        """Cubic Hermite curve from a to b, sampled about every 0.1 voxel; each tangent scales with the chord and its alignment."""
        chord = b - a
        length = np.linalg.norm(chord)
        c = chord / length
        ma = -ta * length * np.clip(np.dot(-ta, c), HERMITE_MIN_COS, 1.0)
        mb = tb * length * np.clip(np.dot(tb, c), HERMITE_MIN_COS, 1.0)
        u = np.linspace(0, 1, int(np.ceil(length / 0.1)) + 1)[:, None]
        return (2 * u ** 3 - 3 * u ** 2 + 1) * a + (u ** 3 - 2 * u ** 2 + u) * ma + (-2 * u ** 3 + 3 * u ** 2) * b + (u ** 3 - u ** 2) * mb

    @staticmethod
    def trim_to_gap(voxels, side_a, side_b):
        """The part of a voxel path between its last voxel on side a and its next voxel on side b."""
        in_a = side_a[voxels[:, 0], voxels[:, 1]]
        in_b = side_b[voxels[:, 0], voxels[:, 1]]
        a = np.max(np.nonzero(in_a)[0]) if in_a.any() else 0
        b_candidates = np.nonzero(in_b)[0]
        b_candidates = b_candidates[b_candidates > a]
        b = np.min(b_candidates) if (in_b.any() and len(b_candidates) > 0) else len(voxels) - 1
        return voxels[a:b + 1]

    def snap(self, voxels, allowed_slice, peeled_slice):
        """Keep allowed or peeled voxels; move each other voxel to the nearest allowed one within SNAP_MM, or drop it."""
        out = []
        for v in voxels:
            if allowed_slice[v[0], v[1]] or peeled_slice[v[0], v[1]]:
                out.append(v)
                continue
            ii, jj = np.mgrid[max(v[0] - 2, 0):v[0] + 3, max(v[1] - 2, 0):v[1] + 3]
            candidates = np.stack([ii.ravel(), jj.ravel()], 1)
            candidates = candidates[allowed_slice[candidates[:, 0], candidates[:, 1]]]
            if len(candidates) == 0:
                continue
            distance = np.linalg.norm(candidates - v, axis=1)
            if distance.min() <= self.params["SNAP_MM"]:
                out.append(candidates[np.argmin(distance)])
        return np.array(out) if out else np.zeros((0, 2), int)

    @staticmethod
    def four_connect(chain, curve, allowed_slice, peeled_slice):
        """Insert a corner voxel at each diagonal step where one is allowed, taking the corner nearer the curve."""
        out = []
        for i, v in enumerate(chain):
            if i > 0:
                u = chain[i - 1]
                d = v - u
                if abs(d[0]) == 1 and abs(d[1]) == 1:
                    options = [np.array([u[0] + d[0], u[1]]), np.array([u[0], u[1] + d[1]])]
                    options = [c for c in options if allowed_slice[c[0], c[1]] or peeled_slice[c[0], c[1]]]
                    if options:
                        distances = [np.min(np.linalg.norm(curve - c, axis=1)) for c in options]
                        out.append(options[int(np.argmin(distances))])
            out.append(v)
        return np.array(out)

    def iso_path(self, a, b, depth_a, depth_b, k, window):
        """Path between the ends that follows the depth from the endocardial surface."""
        ii, jj = np.mgrid[:self.shape[0], :self.shape[1]]
        chord = b - a
        chord_sq = max(float(chord @ chord), 1e-9)
        s = np.clip(((ii - a[0]) * chord[0] + (jj - a[1]) * chord[1]) / chord_sq, 0, 1)
        depth_target = depth_a + (depth_b - depth_a) * s
        domain = (self.allowed[:, :, k] | self.peeled[:, :, k]) & window
        domain[a[0], a[1]] = True
        domain[b[0], b[1]] = True
        cost = np.where(domain, 1.0 + self.params["ISO_W"] * np.abs(self.endo_distance[:, :, k] - depth_target), np.inf)
        router = MCP_Geometric(cost, fully_connected=True)
        distance, _ = router.find_costs([tuple(a)])
        if not np.isfinite(distance[b[0], b[1]]):
            return None
        return np.array(router.traceback(tuple(b)))

    def finish(self, chain, k, extra, a, b):
        """Voxels the chain adds, and whether the two end points are joined under 4- and 8-connectivity."""
        mask = self.peeled[:, :, k].copy()
        if extra is not None:
            mask |= extra
        add = np.zeros_like(mask)
        for v in chain:
            if not mask[v[0], v[1]] and self.allowed[v[0], v[1], k]:
                add[v[0], v[1]] = True
        total = mask | add
        ii, jj = np.mgrid[:self.shape[0], :self.shape[1]]
        mid = (a + b) / 2
        local = total & (np.hypot(ii - mid[0], jj - mid[1]) <= self.params["WIN_MM"] + 2)

        def joined(structure):
            labels, _ = ndi.label(local, structure=structure)
            return bool(labels[a[0], a[1]] > 0 and labels[a[0], a[1]] == labels[b[0], b[1]])

        return add, joined(S4), joined(S8)

    def seal(self, k, extra, inner):
        """Close a residual leak of at most SEAL_MAX voxels next to a bridge; a longer one stays open."""
        for _ in range(5):
            combined = self.peeled[:, :, k] | extra
            if not _is_open(combined, inner, S8):
                break
            found = self.throat(combined, self.traced[:, :, k], inner)
            if found is None or len(found) == 0:
                break
            allowed = self.allowed[found[:, 0], found[:, 1], k]
            bridge_points = np.argwhere(extra)
            near = ((np.min(np.linalg.norm(found[:, None, :] - bridge_points[None, :, :], axis=-1), axis=1) <= self.params["SEAL_NEAR_MM"])
                    if len(bridge_points) else np.zeros(len(found), bool))
            candidates = found[allowed & near]
            if len(candidates) == 0 or len(found) > self.params["SEAL_MAX"]:
                break
            for v in candidates:
                extra[v[0], v[1]] = True

    def run(self, slices):
        """Bridge voxels for all slices; end tangents are averaged across neighbouring slices of a persistent cut."""
        P = self.params
        peeled = self.peeled
        cuts = {}
        for k in slices:
            if not peeled[:, :, k].any():
                continue
            merged, first, inner = self.find_cuts(k)
            if not merged:
                continue
            cuts[k] = []
            for index, (throat_points, first_points) in enumerate(zip(merged, first)):
                ends = self.find_ends(peeled[:, :, k], first_points, self.endo_distance[:, :, k])
                cuts[k].append(dict(index=index, throat=throat_points, inner=inner, ends=ends))

        groups, group_of = [], {}
        for k in sorted(cuts):
            for cut in cuts[k]:
                if cut["ends"] is None:
                    continue
                placed = None
                for group in groups:
                    previous_k, previous_index = group[-1]
                    if (previous_k == k - 1 and np.linalg.norm(cuts[previous_k][previous_index]["throat"].mean(0)
                                                               - cut["throat"].mean(0)) <= P["GROUP_MM"]):
                        placed = group
                        break
                if placed is None:
                    groups.append([(k, cut["index"])])
                else:
                    placed.append((k, cut["index"]))
                group_of[(k, cut["index"])] = len(groups) - 1 if placed is None else groups.index(placed)

        def smoothed_tangents(k, index):
            group = groups[group_of[(k, index)]]
            i = [member[0] for member in group].index(k)
            members = group[max(0, i - 1):i + 2]
            if len(members) < 2:
                return None
            reference = cuts[k][index]["ends"]
            ta, tb = np.zeros(2), np.zeros(2)
            for member_k, member_index in members:
                ends = cuts[member_k][member_index]["ends"]
                ta += ends["ta"] * np.sign(np.dot(ends["ta"], reference["ta"]) or 1)
                tb += ends["tb"] * np.sign(np.dot(ends["tb"], reference["tb"]) or 1)
            return ta / np.linalg.norm(ta), tb / np.linalg.norm(tb)

        added = np.zeros_like(peeled)
        ii, jj = np.mgrid[:self.shape[0], :self.shape[1]]
        for k in sorted(cuts):
            inner = cuts[k][0]["inner"]
            chosen_bridges, extra_iso = np.zeros(self.shape, bool), np.zeros(self.shape, bool)
            for cut in cuts[k]:
                ends = cut["ends"]
                if ends is None:
                    continue
                throat_mid = cut["throat"].mean(0)
                window = np.hypot(ii - throat_mid[0], jj - throat_mid[1]) <= P["WIN_MM"] + 3
                smoothed = smoothed_tangents(k, cut["index"])
                ta, tb = smoothed if smoothed is not None else (ends["ta"], ends["tb"])
                allowed_slice, peeled_slice = self.allowed[:, :, k], peeled[:, :, k]
                curve = self.hermite(ends["a"].astype(float), ends["b"].astype(float), ta, tb)
                voxels = np.round(curve).astype(int)
                keep = np.ones(len(voxels), bool)
                keep[1:] = np.any(voxels[1:] != voxels[:-1], axis=1)
                voxels = self.trim_to_gap(voxels[keep], ends["side_a"], ends["side_b"])
                voxels = self.snap(voxels, allowed_slice, peeled_slice)
                left_trace = int((~self.traced[voxels[:, 0], voxels[:, 1], k]).sum()) if len(voxels) else 0
                u = np.linspace(0, 1, len(voxels))[:, None] if len(voxels) else np.zeros((0, 1))
                depth_target = ends["depth_a"] + (ends["depth_b"] - ends["depth_a"]) * u[:, 0]
                deviation = float(np.max(np.abs(self.endo_distance[voxels[:, 0], voxels[:, 1], k] - depth_target))) if len(voxels) else 0.0
                voxels = self.four_connect(voxels, curve, allowed_slice, peeled_slice) if len(voxels) else voxels

                add_hermite, joined_4, _ = self.finish(voxels, k, chosen_bridges, ends["a"], ends["b"])
                fall_back = (not joined_4) or deviation > P["DEV_MM"] or left_trace > 0
                # Exact output: built for every cut, not only when the curve fails; its voxels accumulate in extra_iso,
                # which the later cuts of this slice pass to finish().
                # Latent: extra_iso also holds paths that were not chosen, so with two cuts on one slice a later cut can
                # count as joined through voxels never added, and the bridge written can then have a gap. Every slice of
                # this heart has at most one cut.
                iso = self.iso_path(ends["a"], ends["b"], ends["depth_a"], ends["depth_b"], k, window)
                if iso is not None:
                    iso = self.four_connect(iso, iso.astype(float), allowed_slice, peeled_slice)
                    add_iso, iso_joined_4, _ = self.finish(iso, k, extra_iso, ends["a"], ends["b"])
                else:
                    add_iso, iso_joined_4 = np.zeros(self.shape, bool), False
                chosen = add_iso if (fall_back and iso is not None and iso_joined_4) else add_hermite
                chosen_bridges |= chosen
                extra_iso |= add_iso
            self.seal(k, chosen_bridges, inner)
            added[:, :, k] = chosen_bridges
        return added
