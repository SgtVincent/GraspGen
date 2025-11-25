import argparse
import os
import glob
import time
import numpy as np
import torch
import trimesh
import trimesh.transformations as tra
from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.robot import get_gripper_info
from grasp_gen.utils import meshcat_utils as mutils
from grasp_gen.utils import viser_utils as vutils
from grasp_gen.utils.point_cloud_utils import filter_colliding_grasps

DEFAULT_POINT_COLOR = np.array([200, 200, 200], dtype=np.uint8)

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Utopia dataset objects and generated grasps")
    parser.add_argument("--data_dir", type=str, default="GraspGenModels/obj_pc", help="Directory containing object folders")
    parser.add_argument("--gripper_config", type=str, default="GraspGenModels/checkpoints/graspgen_franka_panda.yml", 
                        help="Path to gripper configuration YAML file")
    parser.add_argument("--object_name", type=str, default=None, help="Specific object name to visualize")
    parser.add_argument("--use-viser", action="store_true", help="Use Viser for visualization")
    parser.add_argument("--grasp_threshold", type=float, default=0.5, help="Threshold for valid grasps")
    parser.add_argument("--num_grasps", type=int, default=50, help="Number of grasps to generate")
    parser.add_argument("--collision_filter", action="store_true", help="Enable collision filtering of predicted grasps")
    parser.add_argument("--collision_threshold", type=float, default=0.01, help="Collision distance threshold in meters")
    parser.add_argument("--max_scene_points", type=int, default=8192, help="Max scene points for collision checking (0 disables downsampling)")
    return parser.parse_args()

def load_ply_as_points(path):
    def _extract_points_and_colors(geometry):
        if not hasattr(geometry, "vertices"):
            return np.empty((0, 3), dtype=np.float32), None
        vertices = np.asarray(geometry.vertices, dtype=np.float32)
        if vertices.size == 0:
            return np.empty((0, 3), dtype=np.float32), None
        if vertices.ndim == 1:
            try:
                vertices = vertices.reshape(-1, 3)
            except Exception:
                vertices = vertices.reshape(1, -1)
        else:
            vertices = vertices.reshape(-1, vertices.shape[-1])
        if vertices.shape[1] > 3:
            vertices = vertices[:, :3]

        colors = None
        color_sources = []
        visual = getattr(geometry, "visual", None)
        if visual is not None:
            color_sources.append(getattr(visual, "vertex_colors", None))
        color_sources.append(getattr(geometry, "colors", None))

        for src in color_sources:
            if src is None:
                continue
            src_arr = np.asarray(src)
            if src_arr.ndim >= 2 and src_arr.shape[0] == vertices.shape[0]:
                colors = src_arr[:, :3]
                break
        return vertices, colors

    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            return np.empty((0, 3), dtype=np.float32), None
        pts_list = []
        color_list = []
        for geom in mesh.geometry.values():
            pts, cols = _extract_points_and_colors(geom)
            if pts.size == 0:
                continue
            pts_list.append(pts)
            color_list.append(cols)
        if not pts_list:
            return np.empty((0, 3), dtype=np.float32), None
        stacked_pts = np.vstack(pts_list)
        if any(cols is not None for cols in color_list):
            processed_cols = []
            for pts, cols in zip(pts_list, color_list):
                if cols is None:
                    processed_cols.append(np.tile(DEFAULT_POINT_COLOR, (pts.shape[0], 1)))
                else:
                    processed_cols.append(np.asarray(cols)[:, :3])
            stacked_cols = np.vstack(processed_cols)
        else:
            stacked_cols = None
        return stacked_pts, stacked_cols
    elif hasattr(mesh, "vertices"):
        return _extract_points_and_colors(mesh)
    else:
        return np.empty((0, 3), dtype=np.float32), None

def main():
    args = parse_args()
    
    vis_utils = vutils if args.use_viser else mutils
    vis = vis_utils.create_visualizer()
    
    grasp_cfg = load_grasp_cfg(args.gripper_config)
    gripper_name = grasp_cfg.data.gripper_name
    sampler = GraspGenSampler(grasp_cfg)
    gripper_info = get_gripper_info(gripper_name) if args.collision_filter else None
    # Ensure model and any stray tensor attributes are on the correct device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        sampler.model.to(device)
    except Exception:
        try:
            sampler.model.cuda()
        except Exception:
            pass

    # Best-effort: move parameters/buffers and any Tensor attributes stored on modules.
    try:
        for name, p in sampler.model.named_parameters():
            if p is not None:
                p.data = p.data.to(device)
        for name, b in sampler.model.named_buffers():
            if b is not None:
                try:
                    b.data = b.data.to(device)
                except Exception:
                    pass

        stray_attrs = []
        for m_name, m in sampler.model.named_modules():
            for attr_name, attr_val in list(vars(m).items()):
                if isinstance(attr_val, torch.Tensor) and attr_val.device != device:
                    stray_attrs.append((m_name or "<root>", type(m).__name__, attr_name, str(attr_val.device)))
                    try:
                        setattr(m, attr_name, attr_val.to(device))
                    except Exception:
                        pass
        if len(stray_attrs) > 0:
            print("Moved stray tensor attributes to device (or attempted):")
            for s in stray_attrs:
                print(" module:", s[0], "class:", s[1], "attr:", s[2], "orig_device:", s[3])
    except Exception:
        pass
    
    if args.object_name:
        object_dirs = [os.path.join(args.data_dir, args.object_name)]
    else:
        object_dirs = glob.glob(os.path.join(args.data_dir, "*"))
        object_dirs = [d for d in object_dirs if os.path.isdir(d)]
        object_dirs.sort()
        
    for obj_dir in object_dirs:
        obj_name = os.path.basename(obj_dir)
        print(f"Processing {obj_name}")
        
        vis_utils.clear_visualization(vis)
        
        local_pc_path = os.path.join(obj_dir, "local_pc.ply")
        global_pc_path = os.path.join(obj_dir, "global_pc.ply")
        
        if not os.path.exists(local_pc_path) or not os.path.exists(global_pc_path):
            print(f"Skipping {obj_name}: Missing ply files")
            continue
            
        # Load point clouds
        local_pc, local_colors = load_ply_as_points(local_pc_path)
        global_pc, global_colors = load_ply_as_points(global_pc_path)
        
        # Visualize scene (global)
        if global_pc.size > 0:
            # Use recorded colors if available; otherwise light gray for the scene
            scene_color = global_colors if global_colors is not None else np.array([180, 180, 180], dtype=np.uint8)
            vis_utils.visualize_pointcloud(vis, "scene", global_pc, size=0.002, color=scene_color)
            
        # Visualize object (local)
        if local_pc.size > 0:
            # Use recorded colors if available; fall back to green for the target object
            object_color = local_colors if local_colors is not None else np.array([0, 255, 0], dtype=np.uint8)
            vis_utils.visualize_pointcloud(vis, "object", local_pc, size=0.003, color=object_color)
            try:
                vis_utils.focus_camera_on_points(vis, local_pc)
            except AttributeError:
                pass
            
            print(f"Running inference on {local_pc.shape[0]} points...")
            try:
                local_pc_np = torch.as_tensor(local_pc, dtype=torch.float32, device=device)
            except Exception:
                local_pc_np = torch.tensor(local_pc, dtype=torch.float32, device=device)

            # Run inference. Disable internal outlier removal to avoid device-mismatch
            grasps, scores = GraspGenSampler.run_inference(
                local_pc_np,
                sampler,
                grasp_threshold=args.grasp_threshold,
                num_grasps=args.num_grasps,
                remove_outliers=False,
            )
            
            if len(grasps) > 0:
                grasps = grasps.cpu().numpy()
                scores = scores.cpu().numpy()
                grasps[:, 3, 3] = 1

                collision_mask = None
                removed_grasps = None
                if args.collision_filter:
                    if global_pc.size == 0:
                        print("Skipping collision filtering: global point cloud is empty.")
                    elif gripper_info is None:
                        print("Skipping collision filtering: gripper info unavailable.")
                    else:
                        scene_pc = global_pc
                        if args.max_scene_points > 0 and scene_pc.shape[0] > args.max_scene_points:
                            idx = np.random.choice(scene_pc.shape[0], args.max_scene_points, replace=False)
                            scene_pc = scene_pc[idx]
                        print("Running collision filtering...")
                        try:
                            collision_mask = filter_colliding_grasps(
                                scene_pc=scene_pc,
                                grasp_poses=grasps,
                                gripper_collision_mesh=gripper_info.collision_mesh,
                                collision_threshold=args.collision_threshold,
                            )
                        except Exception as exc:
                            print(f"Collision filtering failed: {exc}")
                            collision_mask = None
                        if collision_mask is not None:
                            kept = int(np.sum(collision_mask))
                            print(f"Collision filtering kept {kept}/{len(grasps)} grasps (threshold {args.collision_threshold} m).")
                            if kept == 0:
                                print("No collision-free grasps remain after filtering.")
                            if kept < len(grasps):
                                removed_grasps = grasps[~collision_mask]

                base_colors = vis_utils.get_color_from_score(scores, use_255_scale=True)
                grasps_to_show = grasps
                colors_to_show = base_colors
                if collision_mask is not None:
                    if collision_mask.any():
                        grasps_to_show = grasps[collision_mask]
                        colors_to_show = base_colors[collision_mask]
                    else:
                        grasps_to_show = np.empty((0, 4, 4))
                        colors_to_show = np.empty((0, 3))

                print(f"Visualizing {len(grasps_to_show)} grasps...")
                for i, grasp in enumerate(grasps_to_show):
                    vis_utils.visualize_grasp(
                        vis,
                        f"grasps/{i}",
                        grasp,
                        color=colors_to_show[i],
                        gripper_name=gripper_name
                    )

                if removed_grasps is not None and removed_grasps.size > 0:
                    max_colliding = min(20, removed_grasps.shape[0])
                    print(f"Visualizing {max_colliding} colliding grasps in red for reference.")
                    for i in range(max_colliding):
                        vis_utils.visualize_grasp(
                            vis,
                            f"colliding_grasps/{i}",
                            removed_grasps[i],
                            color=[255, 0, 0],
                            gripper_name=gripper_name
                        )
            else:
                print("No grasps found.")
                
        input("Press Enter for next object...")

if __name__ == "__main__":
    main()
