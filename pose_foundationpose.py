"""
FoundationPose for ONE segment -- single still frame, no tracking (that's for
video sequences, we just have one shot). Mirrors run_demo.py's actual
initialization (mesh, ScorePredictor, PoseRefinePredictor, glctx,
FoundationPose(...)) but skips YcbineoatReader entirely -- we feed our own
cropped color/depth/mask arrays straight to register(), once.

Must be run from INSIDE the FoundationPose repo (same reason run_demo.py
works there: `from estimater import *` resolves via the script's own
directory), with that repo's conda env active.

IMPORTANT: the pose this saves is the RAW register() output -- relative to
the mesh file's own vertex frame directly (Z = long axis, base(-) to cap(+);
origin = centroid, confirmed on bottle_baked.obj). It is NOT the OBB-derived
center_pose run_demo.py computes only for drawing its visualization box --
that uses a different, arbitrarily-oriented frame and would silently break
the axis convention we deliberately set up.

PCA-informed prior (on by default, --no-pca-prior to disable): register()
internally generates a FIXED, generic grid of candidate rotations
(self.rot_grid, independent of the actual scene), refines all of them, then
scores and picks the best. Confirmed on real segments: when none of that
fixed grid's candidates are close to the true orientation, refinement alone
(even with many more iterations) can't rescue it -- it's local search, it
can only polish within whatever basin it started in, and more iterations on
a wrong basin made results WORSE, not better. Cheap, effective fix: monkey-
patch just this one method (no changes to FoundationPose's own source) to
append a handful of extra candidates built from a quick PCA axis fit on the
same segment's masked depth points -- 2 directions (PCA's axis is undirected,
can't tell cap from base) x 4 roll angles each, since the appearance-based
scorer *can* tell cap from base and can tell a good roll from a bad one, it
just needs a candidate near the right basin to be given the chance.

    python3 pose_foundationpose.py \\
        --segments bulk_pick_test/color_segments.json \\
        --depth bulk_pick_test/depth.npy \\
        --color bulk_pick_test/color.png \\
        --cad bulk_pick_test/bottle_baked_zflip.obj \\
        --segment 2 \\
        --out-dir bulk_pick_test/pose_out
"""

import os
import json
import argparse

import numpy as np
import cv2
import torch
import trimesh
from pycocotools import mask as maskutil

from estimater import *  # ScorePredictor, PoseRefinePredictor, FoundationPose, dr, set_logging_format, set_seed


def pca_axis_from_mask_depth(mask, depth, K):
    """Same math as pose_pca.py -- backproject the masked depth points, PCA
    for the long-axis direction. Returns (centroid, axis_unit_vector) in
    camera-frame meters, or (None, None) if too few valid points."""
    vs, us = np.nonzero(mask)
    Z = depth[mask]
    valid = Z > 0
    if valid.sum() < 10:
        return None, None
    us, vs, Z = us[valid].astype(np.float64), vs[valid].astype(np.float64), Z[valid].astype(np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    X = (us - cx) * Z / fx
    Y = (vs - cy) * Z / fy
    pts = np.stack([X, Y, Z], axis=1)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    cov = (centered.T @ centered) / len(pts)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    return centroid, axis / np.linalg.norm(axis)


def build_pca_prior_poses(axis, n_rolls=4, device="cuda", dtype=torch.float32):
    """A handful of candidate rotations with mesh-Z aligned to +-axis, at a
    few roll angles each -- translation left as identity, register() always
    overwrites it with its own guess_translation() right after calling this,
    so it doesn't matter what we put here."""
    ref = np.array([0.0, 1.0, 0.0]) if abs(axis[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    poses = []
    for sign in (1.0, -1.0):
        z_col = axis * sign
        x_col = np.cross(ref, z_col)
        x_col /= np.linalg.norm(x_col)
        y_col = np.cross(z_col, x_col)
        for k in range(n_rolls):
            theta = 2 * np.pi * k / n_rolls
            c, s = np.cos(theta), np.sin(theta)
            x_rot = c * x_col + s * y_col
            y_rot = -s * x_col + c * y_col
            R = np.stack([x_rot, y_rot, z_col], axis=1)  # columns
            T = np.eye(4)
            T[:3, :3] = R
            poses.append(T)
    return torch.as_tensor(np.stack(poses), device=device, dtype=dtype)

# Color camera intrinsics (full-res) -- same constants used throughout this
# project (colorize_pointcloud_from_depth.py, align_depth_to_color.py).
# depth.npy transferred here is already color-aligned -- these are the right
# intrinsics for it, not the IR ones.
COLOR_FX, COLOR_FY, COLOR_CX, COLOR_CY = 909.403, 908.526, 640.044, 353.843

ap = argparse.ArgumentParser()
ap.add_argument("--segments", required=True, help="<stem>_segments.json from combine_segments_depth.py")
ap.add_argument("--depth", required=True, help="depth.npy -- color-aligned, meters, 0=invalid")
ap.add_argument("--color", required=True, help="color.png -- full, uncropped")
ap.add_argument("--cad", required=True, help="bottle_baked.obj -- meters, Z=long axis, origin=centroid")
ap.add_argument("--segment", type=int, required=True)
ap.add_argument("--est-refine-iter", type=int, default=5)
ap.add_argument("--no-pca-prior", action="store_true", help="disable the PCA-informed candidate injection")
ap.add_argument("--out-dir", default="pose_out")
args = ap.parse_args()

set_logging_format()
set_seed(0)

with open(args.segments) as f:
    data = json.load(f)
x0, y0, w, h = data["crop"]
match = [s for s in data["segments"] if s["segment"] == args.segment]
if not match:
    raise SystemExit(f"--segment {args.segment} not found (available: {[s['segment'] for s in data['segments']]})")
mask = maskutil.decode(match[0]["mask_rle"]).astype(bool)  # already in the cropped frame

color_full = cv2.imread(args.color)
depth_full = np.load(args.depth)
color_bgr = color_full[y0:y0 + h, x0:x0 + w]
color = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
depth = depth_full[y0:y0 + h, x0:x0 + w].astype(np.float32)

if depth.shape[:2] != mask.shape or color.shape[:2] != mask.shape:
    raise SystemExit(f"shape mismatch -- color {color.shape[:2]}, depth {depth.shape}, mask {mask.shape} "
                      f"must all match after cropping")

K = np.array([[COLOR_FX, 0, COLOR_CX - x0],
              [0, COLOR_FY, COLOR_CY - y0],
              [0, 0, 1]], dtype=np.float64)

os.makedirs(args.out_dir, exist_ok=True)
debug_dir = args.out_dir

mesh = trimesh.load(args.cad, force="mesh")
print(f"loaded CAD: {args.cad}, extents={mesh.extents}")

scorer = ScorePredictor()
refiner = PoseRefinePredictor()
glctx = dr.RasterizeCudaContext()
est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
                      scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=1, glctx=glctx)
print("estimator initialization done")

if not args.no_pca_prior:
    pca_centroid, pca_axis = pca_axis_from_mask_depth(mask, depth, K)
    if pca_axis is None:
        print("PCA prior: too few valid depth points in mask, skipping injection")
    else:
        print(f"PCA prior: axis={pca_axis}, injecting extra candidates")
        n_base_candidates = [None]  # set once generate_random_pose_hypo runs, read by the scorer patch below

        original_generate = est.generate_random_pose_hypo  # bound method, already carries self

        def patched_generate_random_pose_hypo(K, rgb, depth, mask, scene_pts=None):
            base_poses = original_generate(K=K, rgb=rgb, depth=depth, mask=mask, scene_pts=scene_pts)
            extra = build_pca_prior_poses(pca_axis, device=base_poses.device, dtype=base_poses.dtype)
            n_base_candidates[0] = base_poses.shape[0]
            combined = torch.cat([base_poses, extra], dim=0)
            print(f"PCA prior: {base_poses.shape[0]} base candidates + {extra.shape[0]} injected = {combined.shape[0]} total")
            return combined

        est.generate_random_pose_hypo = patched_generate_random_pose_hypo

        # Diagnostic: register() sorts/reorders scores internally and doesn't expose
        # the raw per-candidate order, so intercept scorer.predict directly -- at
        # that point scores are still in the same order as our torch.cat above
        # (base candidates first, then ours), so we can compare them directly.
        original_scorer_predict = est.scorer.predict

        def patched_scorer_predict(*sargs, **skwargs):
            result = original_scorer_predict(*sargs, **skwargs)
            scores = result[0] if isinstance(result, tuple) else result
            n_base = n_base_candidates[0]
            if n_base is not None:
                best_base = scores[:n_base].max().item()
                best_injected = scores[n_base:].max().item()
                print(f"PCA prior diagnostic: best base-grid score={best_base:.4f}, "
                      f"best injected-candidate score={best_injected:.4f}, "
                      f"overall winner score={scores.max().item():.4f}")
            return result

        est.scorer.predict = patched_scorer_predict

pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)

pose_path = os.path.join(debug_dir, f"segment{args.segment}_pose.txt")
np.savetxt(pose_path, pose.reshape(4, 4))
print(f"\npose (mesh frame -> camera frame):\n{pose.reshape(4, 4)}")
print(f"saved {pose_path}")

# visualization -- uses center_pose (OBB-derived) ONLY for drawing the box,
# per run_demo.py's own convention; the SAVED pose above is what to actually use.
to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
center_pose = pose @ np.linalg.inv(to_origin)
vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.05, K=K, thickness=3, transparency=0, is_input_rgb=True)
vis_path = os.path.join(debug_dir, f"segment{args.segment}_vis.png")
cv2.imwrite(vis_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
print(f"saved {vis_path}")
