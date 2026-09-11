"""
Run FoundationPose across segments in one *_segments.json, saving one
visualization + pose file each, plus a combined image with every processed
segment's box/axis drawn on the same picture -- for getting a quick overall
sense of how many segments come out looking usable. No PCA-prior injection
here (that was pose_foundationpose.py's debug experiment, on the
alternative_approaches branch) -- this is the plain, vanilla register()
call.

Works with either camera this project supports: pass --intrinsics to point
at the camera's calibration JSON (fx/fy/cx/cy) -- defaults to this
project's RealSense config, pass the path to a capture's own
zivid_intrinsics.json (from read_zdf.py) for Zivid data.

Loads the model/mesh ONCE (real cost -- ScorePredictor/PoseRefinePredictor/
mesh loading takes real time), then loops register() per segment, since
register() doesn't depend on any state from a previous call (unlike
track_one(), which needs self.pose_last from a prior register()).

Must run from inside the FoundationPose repo/environment.

    # RealSense (default intrinsics):
    python3 pose_foundationpose_batch.py \\
        --segments bulk_pick_test/color_segments.json \\
        --depth bulk_pick_test/depth.npy \\
        --color bulk_pick_test/color.png \\
        --cad bulk_pick_test/bottle_baked_zflip.obj \\
        --out-dir bulk_pick_test/pose_out_all

    # Zivid, restricted to specific segments already confirmed clean via
    # the PCA quality checks:
    python3 pose_foundationpose_batch.py \\
        --segments color1_segments.json \\
        --depth zivid_depth.npy \\
        --color color1.png \\
        --intrinsics zivid_intrinsics.json \\
        --cad bottle_baked_zflip.obj \\
        --segment-ids 0,1,3,6 \\
        --out-dir pose_out_zivid
"""

import os
import json
import argparse

import numpy as np
import cv2
import trimesh
from pycocotools import mask as maskutil

from estimater import *  # ScorePredictor, PoseRefinePredictor, FoundationPose, dr, set_logging_format, set_seed


ap = argparse.ArgumentParser()
ap.add_argument("--intrinsics", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "camera_realsense.json"),
                 help="JSON with fx, fy, cx, cy -- defaults to this project's RealSense calibration; "
                      "pass a capture's zivid_intrinsics.json (from read_zdf.py) for Zivid data")
ap.add_argument("--segments", required=True)
ap.add_argument("--depth", required=True)
ap.add_argument("--color", required=True)
ap.add_argument("--cad", required=True)
ap.add_argument("--segment-ids", type=str, default=None,
                 help="comma-separated list, e.g. '0,1,3,6' -- omit to run all segments")
ap.add_argument("--est-refine-iter", type=int, default=5)
ap.add_argument("--out-dir", default="pose_out_all")
args = ap.parse_args()

with open(args.intrinsics) as f:
    _intr = json.load(f)
FX, FY, CX, CY = _intr["fx"], _intr["fy"], _intr["cx"], _intr["cy"]
print(f"intrinsics: fx={FX:.2f} fy={FY:.2f} cx={CX:.2f} cy={CY:.2f} (from {args.intrinsics})")

set_logging_format()
set_seed(0)

with open(args.segments) as f:
    data = json.load(f)
x0, y0, w, h = data["crop"]

color_full = cv2.imread(args.color)
depth_full = np.load(args.depth)
color_bgr = color_full[y0:y0 + h, x0:x0 + w]
color = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
depth = depth_full[y0:y0 + h, x0:x0 + w].astype(np.float32)

K = np.array([[FX, 0, CX - x0],
              [0, FY, CY - y0],
              [0, 0, 1]], dtype=np.float64)

os.makedirs(args.out_dir, exist_ok=True)

mesh = trimesh.load(args.cad, force="mesh")
print(f"loaded CAD: {args.cad}, extents={mesh.extents}")

scorer = ScorePredictor()
refiner = PoseRefinePredictor()
glctx = dr.RasterizeCudaContext()
est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
                      scorer=scorer, refiner=refiner, debug_dir=args.out_dir, debug=1, glctx=glctx)
print("estimator initialization done -- looping segments now\n")

to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

if args.segment_ids:
    wanted = {int(x) for x in args.segment_ids.split(",")}
    segments = [s for s in data["segments"] if s["segment"] in wanted]
    missing = wanted - {s["segment"] for s in segments}
    if missing:
        print(f"warning: segment ids not found, skipping: {sorted(missing)}")
else:
    segments = data["segments"]

results = []
combined = color.copy()  # every segment's box/axis drawn on this SAME image, saved once at the end
for seg in segments:
    n = seg["segment"]
    mask = maskutil.decode(seg["mask_rle"]).astype(bool)
    if mask.shape != depth.shape:
        print(f"segment {n}: shape mismatch, skipping")
        continue

    pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
    pose = pose.reshape(4, 4)
    np.savetxt(os.path.join(args.out_dir, f"segment{n}_pose.txt"), pose)

    center_pose = pose @ np.linalg.inv(to_origin)
    vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
    vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.05, K=K, thickness=3, transparency=0, is_input_rgb=True)
    vis_path = os.path.join(args.out_dir, f"segment{n}_vis.png")
    cv2.imwrite(vis_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    combined = draw_posed_3d_box(K, img=combined, ob_in_cam=center_pose, bbox=bbox)
    combined = draw_xyz_axis(combined, ob_in_cam=center_pose, scale=0.05, K=K, thickness=3, transparency=0, is_input_rgb=True)

    print(f"segment {n}: pose Z={pose[2, 3]:.3f}m (segment's own mean_depth_m={seg.get('mean_depth_m')}) -> {vis_path}")
    results.append(n)

combined_path = os.path.join(args.out_dir, "combined_vis.png")
cv2.imwrite(combined_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

print(f"\ndone -- {len(results)} segments processed, check {args.out_dir}/segment*_vis.png")
print(f"combined view (all segments on one image) -> {combined_path}")
