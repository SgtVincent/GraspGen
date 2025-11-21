#!/usr/bin/env python3
"""
Batch inference for URDF assets: loads URDFs, merges meshes, runs GraspGen
and saves predicted grasps into the URDF folder as an Isaac-formatted YAML.

This mirrors the logic in `demo_object_mesh.py`, but iterates over URDFs in
the given folder and uses `yourdfpy` to load and flatten meshes from URDFs.
"""
import argparse
import json
import yaml
import os
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh
import trimesh.transformations as tra

from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.dataset.eval_utils import (
    load_urdf_scene,
    save_to_isaac_grasp_format,
    dump_merged_urdf_mesh,
    find_base_config_size,
)
from grasp_gen.dataset.dataset_utils import sample_points
from grasp_gen.utils.point_cloud_utils import point_cloud_outlier_removal

###################################
# Vscode remote debug configuration
###################################
# import debugpy

# debugpy.listen(5678)
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()
# print("Debugger attached.")
###################################


def get_annotation_aabb(annotation_path: Path):
    if not annotation_path.exists():
        return None
    with annotation_path.open('r') as f:
        js = json.load(f)
    pts = js.get('affordable_points', []) or js.get('affordance_points', [])
    if len(pts) == 0:
        return None
    ary = np.asarray(pts, dtype=float)
    min_corner = ary.min(axis=0)
    max_corner = ary.max(axis=0)
    return min_corner, max_corner


def in_aabb(points: np.ndarray, aabb: tuple) -> np.ndarray:
    min_corner, max_corner = aabb
    mask = np.all((points >= min_corner) & (points <= max_corner), axis=1)
    return mask


def load_and_merge_urdf(urdf_path: Path) -> Optional[trimesh.Trimesh]:
    # use the centralized helper so the generator and viewer use identical
    # merged mesh extraction logic
    return dump_merged_urdf_mesh(str(urdf_path))


def sample_and_run(
    mesh: trimesh.Trimesh,
    sampler: GraspGenSampler,
    num_sample_points: int,
    num_grasps: int,
    threshold: float,
    topk_num_grasps: int,
    remove_outliers: bool = False,
    aabb: Optional[tuple] = None,
    scale_factor: float = 1.0,
    aabb_padding: float = 0.0,
    use_aabb_sampling: bool = False,
):
    # sample points on mesh
    xyz, _ = trimesh.sample.sample_surface(mesh, num_sample_points)
    xyz = np.array(xyz)
    print(f"Sampled {len(xyz)} points from mesh")
    if xyz.size == 0:
        return None, None
    # center point cloud
    T_subtract_pc_mean = tra.translation_matrix(-xyz.mean(axis=0))
    xyz = tra.transform_points(xyz, T_subtract_pc_mean)
    mesh.apply_transform(T_subtract_pc_mean)

    # If requested, pre-filter outliers here so we can gracefully handle the
    # case where the filter removes all points (the model expects N>0).
    if remove_outliers:
        try:
            xyz_filtered, removed = point_cloud_outlier_removal(xyz)
            print(f"Outlier filter removed {len(removed)} points")
        except Exception:
            # If filtering error occurs, revert to original points
            xyz_filtered = xyz
        xyz = np.asarray(xyz_filtered)
        if xyz.size == 0:
            print("All sampled points removed by outlier filter; skipping.")
            return None, None

    # Optionally sample only from the annotation bounds if requested.  This
    # helps ensure inferred grasps fall within the annotated region.
    if use_aabb_sampling and aabb is not None:
        pad = aabb_padding if aabb_padding is not None else 0.0
        min_corner, max_corner = aabb
        # The point cloud is centered by T_subtract_pc_mean above, so the
        # annotation bounds must be transformed into the same centered frame
        # before checking membership.
        # apply same translation as for the points
        # translation matrix is T_subtract_pc_mean; to transform points into
        # centered frame we apply T_subtract_pc_mean to annotation corners.
        min_corner = tra.transform_points(np.asarray([min_corner]), T_subtract_pc_mean)[0]
        max_corner = tra.transform_points(np.asarray([max_corner]), T_subtract_pc_mean)[0]
        if pad > 0.0:
            min_corner = min_corner - pad
            max_corner = max_corner + pad
        aabb_padded = (min_corner, max_corner)
        mask = in_aabb(xyz, aabb_padded)
        xyz_inside = xyz[mask]
        if len(xyz_inside) == 0:
            print("Warning: annotation AABB selected no sampled points. Falling back to whole mesh sampling.")
        else:
            print(f"Sampling inside annotation bounds: {len(xyz_inside)}/{len(xyz)} points kept")
            xyz = xyz_inside

    grasps, confs = None, None
    try:
        grasps, confs = GraspGenSampler.run_inference(
        xyz,
        sampler,
        grasp_threshold=threshold,
        num_grasps=num_grasps,
        topk_num_grasps=topk_num_grasps,
        remove_outliers=False,
    )
    except RuntimeError as e:
        print(f"Inference failed for this object: {e}")
        return None, None
    if len(grasps) == 0:
        return None, None
    # convert back to the mesh frame (note: mesh may already be scaled)
    inv_T = tra.inverse_matrix(T_subtract_pc_mean)
    grasps = np.asarray([inv_T @ g for g in grasps.cpu().numpy()])
    # NOTE: we intentionally do NOT convert predicted grasp translations
    # back to the original unscaled URDF coordinates. Predictions are
    # returned in the coordinate frame of the current mesh (which may have
    # been scaled by base_config.size). This ensures generated grasps are
    # directly compatible with the same scaled mesh when visualizing.
    confs = confs.cpu().numpy()
    return grasps, confs


def process_urdf(urdf_path: Path, sampler: GraspGenSampler, args):
    folder = urdf_path.parent
    annotation_path = folder / 'annotation.json'
    aabb = get_annotation_aabb(annotation_path)
    aabb_original = aabb
    mesh = load_and_merge_urdf(urdf_path)
    if mesh is None:
        print(f"Skipping {urdf_path}, no mesh found")
        return

    # If this object provides a base_config.yaml with a `size` field, scale
    # the URDF mesh and the affordance points so sampling and AABB checks use
    # the canonical object size. This rescales based on the largest mesh
    # extent (preserves proportions).
    # Read scale factor from base_config.yaml (if present). This helper
    # centralizes discovery of the `size` field across the repo so the
    # generator and viewer scale the mesh the same way.
    scale_factor = find_base_config_size(str(urdf_path)) or 1.0
    if float(scale_factor) != 1.0:
        try:
            mesh.apply_scale(float(scale_factor))
            print(f"Applied scaling factor {float(scale_factor):.4f} from base_config.size")
        except Exception as e:
            print(f"Failed to apply scale factor to mesh: {e}")

    # For sampling we use the scaled AABB (if any). Keep the original AABB
    # around for final filtering (we will filter in original URDF coords).
    aabb_for_sampling = aabb
    if aabb is not None and float(scale_factor) != 1.0:
        aabb_for_sampling = (aabb[0] * float(scale_factor), aabb[1] * float(scale_factor))

    print(f"Sample & running inference on {urdf_path}")
    # Debug: print mesh bounds and annotation for diagnosis
    if aabb is not None:
        print(f"Annotation AABB raw min={aabb[0]}, max={aabb[1]}")
    print(f"Mesh bounds min={mesh.bounds[0]}, max={mesh.bounds[1]}")
    # Use annotation AABB for sampling if requested
    use_aabb_sampling = getattr(args, 'use_aabb_sampling', False)
    grasps, confs = sample_and_run(
        mesh,
        sampler,
        args.num_sample_points,
        args.num_grasps,
        args.grasp_threshold,
        args.topk_num_grasps,
        remove_outliers=not getattr(args, 'no_remove_outliers', False),
        aabb=aabb_for_sampling,
        scale_factor=scale_factor,
        aabb_padding=getattr(args, 'aabb_padding', 0.0),
        use_aabb_sampling=use_aabb_sampling,
    )
    if grasps is None or len(grasps) == 0:
        print(f"No grasps found for {urdf_path}")
        return
    if getattr(args, 'dump_grasp_locations', 0) > 0:
        positions = np.asarray([g[:3, 3] for g in grasps])
        print(f"First {args.dump_grasp_locations} grasp positions:")
        for p in positions[: args.dump_grasp_locations]:
            print(p)

    if aabb_original is not None and confs is not None and not getattr(args, 'skip_aabb_filter', False):
        # extract translation part of each grasp
        positions = np.asarray([g[:3, 3] for g in grasps])
        # expand aabb if requested
        pad = args.aabb_padding if getattr(args, 'aabb_padding', 0.0) else 0.0
        # Use the annotation coordinates for filtering saved grasps. Because
        # grasps are saved in the scaled mesh coordinate frame, scale the
        # annotation AABB to the same frame before filtering if needed.
        min_corner, max_corner = aabb_original
        if float(scale_factor) != 1.0:
            min_corner = min_corner * float(scale_factor)
            max_corner = max_corner * float(scale_factor)
        if pad > 0:
            min_corner = min_corner - pad
            max_corner = max_corner + pad
        aabb = (min_corner, max_corner)
        print(f"Annotation AABB min={min_corner}, max={max_corner}")
        print(f"Number of predicted grasps: {len(positions)}")
        mask = in_aabb(positions, aabb)
        print(f"Grasps inside AABB: {mask.sum()}/{mask.shape[0]}")
        grasps = grasps[mask]
        confs = confs[mask]
        print(f"Filtered grasps using annotation aabb: retained {len(grasps)} grasps")
        if len(grasps) == 0:
            # log some info to help debugging
            if len(positions) > 0:
                dists = np.maximum(positions - max_corner, min_corner - positions)
                outside_dist = np.linalg.norm(np.maximum(dists, 0.0), axis=1)
                print(f"Minimum distance from predicted grasps to AABB: {outside_dist.min():.4f} m")
            print("No grasps remain after AABB filtering; skipping save.")
            return

    out_file = folder / 'predicted_grasps.yml'
    out_file.parent.mkdir(parents=True, exist_ok=True)
    # Ensure confidences are set
    if confs is None:
        confs = np.ones((len(grasps),), dtype=float)
    save_to_isaac_grasp_format(grasps, confs, str(out_file))
    print(f"Saved {len(grasps)} predicted grasps to {out_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Batch predict grasps for all URDF models under a root directory")
    parser.add_argument("--urdf-root", type=Path, default=Path("GraspGenModels/URDF"))
    parser.add_argument("--gripper-config", type=str, required=True)
    parser.add_argument("--num-grasps", type=int, default=400)
    parser.add_argument("--num-sample-points", type=int, default=2000)
    parser.add_argument("--grasp-threshold", type=float, default=-1.0)
    parser.add_argument("--return-topk", action="store_true")
    parser.add_argument("--topk-num-grasps", type=int, default=-1)
    parser.add_argument("--aabb-padding", type=float, default=0.2, help="Padding to apply to annotation AABB when filtering grasps (in meters)")
    parser.add_argument("--skip-aabb-filter", action="store_true", help="Skip filtering grasps by annotation AABB")
    parser.add_argument("--no-remove-outliers", action="store_true", help="Do not remove point cloud outliers before inference")
    parser.add_argument("--dump-grasp-locations", type=int, default=0, help="Print first N predicted grasp positions for debugging")
    parser.add_argument(
        "--use-aabb-sampling",
        action="store_true",
        help="If set, sample surface points only inside the annotation's AABB (with padding). Falls back if no points found.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.return_topk and args.topk_num_grasps == -1:
        args.topk_num_grasps = 100

    cfg = load_grasp_cfg(args.gripper_config)
    sampler = GraspGenSampler(cfg)

    urdf_root = Path(args.urdf_root)
    urdf_files = list(urdf_root.rglob("*.urdf"))
    if not urdf_files:
        print(f"No URDF files found under {urdf_root}")
        return

    for urdf_path in urdf_files:
        process_urdf(urdf_path, sampler, args)


if __name__ == "__main__":
    main()
