import argparse
import os
import glob
import time
import numpy as np
import torch
import trimesh
import imageio
import trimesh.transformations as tra
try:
    import viser
    from grasp_gen.utils import viser_utils as vutils
    VISER_AVAILABLE = True
except Exception:
    VISER_AVAILABLE = False
try:
    import open3d as o3d
    O3D_AVAILABLE = True
except Exception:
    O3D_AVAILABLE = False
from tqdm import tqdm
from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg

DEFAULT_POINT_COLOR = np.array([200, 200, 200], dtype=np.uint8)


def capture_viser_snapshot(server, save_path, viewport: str | None = None, wait_s: float = 3.0):
    """Capture a rendered image from an active Viser server and save to disk.

    Tries several common client APIs (`get_rendered_image`, `get_render`) and
    handles bytes/numpy return types. Assumes the scene has been populated
    already (points, grasps, etc.). This is non-blocking for the caller —
    we schedule a short wait to let the server render then fetch one frame.
    """
    if server is None:
        raise RuntimeError("Viser server is None — cannot capture snapshot")

    # Prefer a short delay so the server has time to update the frame
    time.sleep(wait_s)

    client = None
    # Try to obtain a client handle. Prefer get_clients() mapping if present
    # and wait briefly for clients to connect (non-blocking overall).
    try:
        if hasattr(server, "get_clients"):
            # Poll for connected clients for up to `wait_s` seconds.
            start_t = time.time()
            while time.time() - start_t < wait_s:
                clients = server.get_clients()
                # get_clients may return a dict of client_id -> handle
                if isinstance(clients, dict) and len(clients) > 0:
                    client = list(clients.values())[0]
                    break
                time.sleep(0.05)
        # fallback to server.get_client() or server.get_client_handle()
        if client is None and hasattr(server, "get_client"):
            try:
                client = server.get_client()
            except Exception:
                client = getattr(server, "client", None)
        if client is None:
            client = getattr(server, "client", None) or getattr(server, "get_client_handle", None)
    except Exception:
        client = getattr(server, "client", None) or getattr(server, "get_client_handle", None)

    if client is None:
        # Include helpful server URL in the message so the user can open it
        host = getattr(server, "get_host", lambda: "localhost")()
        port = getattr(server, "get_port", lambda: 8080)()
        raise RuntimeError(
            f"Could not obtain a Viser client handle — no clients connected to http://{host}:{port} within {wait_s}s"
        )

    # Try the most common capture methods used by Viser clients
    rendered = None
    try:
        if hasattr(client, "get_rendered_image"):
            rendered = client.get_rendered_image(viewport) if viewport is not None else client.get_rendered_image()
        elif hasattr(client, "get_render"):
            rendered = client.get_render(viewport) if viewport is not None else client.get_render()
        else:
            # Some clients expose a `render` method which returns image bytes
            if hasattr(client, "render"):
                rendered = client.render()
    except Exception:
        rendered = None

    if rendered is None:
        raise RuntimeError("Viser client did not return an image payload")

    # rendered can be bytes, numpy array, or dict depending on Viser version
    # Handle bytes first
    try:
        # bytes-like
        if isinstance(rendered, (bytes, bytearray)):
            with open(save_path, "wb") as fh:
                fh.write(rendered)
            return

        # numpy array
        import numpy as _np

        if isinstance(rendered, _np.ndarray):
            imageio.imwrite(save_path, rendered)
            return

        # dict-like payload might contain 'image'/'image_bytes'/'rgba' keys
        if isinstance(rendered, dict):
            if "image" in rendered:
                payload = rendered["image"]
            elif "image_bytes" in rendered:
                payload = rendered["image_bytes"]
            elif "rgba" in rendered:
                payload = rendered["rgba"]
            else:
                payload = None

            if payload is None:
                raise RuntimeError("Unrecognized render payload from Viser client")

            if isinstance(payload, (bytes, bytearray)):
                with open(save_path, "wb") as fh:
                    fh.write(payload)
                return

            if isinstance(payload, _np.ndarray):
                imageio.imwrite(save_path, payload)
                return

    except Exception as e:
        raise RuntimeError(f"Failed to write render result: {e}")


def render_open3d_snapshot(pc: np.ndarray, grasps: np.ndarray, scores: np.ndarray, save_path: str, size: int = 1024, pc_colors=None):
    """Render a headless screenshot using Open3D offscreen renderer.

    This function uses the OffscreenRenderer if available, otherwise falls
    back to the classic Visualizer capture approach.
    """
    if not O3D_AVAILABLE:
        raise RuntimeError("Open3D not available for headless rendering")

    # Build Open3D geometries
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    # Default color: light gray unless explicit colors are provided
    try:
        if pc_colors is not None and len(pc_colors) == len(pc):
            colors = np.asarray(pc_colors, dtype=np.float64)
            if colors.max() > 1.0:
                colors = colors / 255.0
        else:
            colors = np.ones((len(pc), 3), dtype=np.float64) * 0.7
        pcd.colors = o3d.utility.Vector3dVector(colors)
    except Exception:
        pass

    # Create grasp visuals as LineSets (small axis triads per grasp)
    line_sets = []
    if grasps is not None and len(grasps) > 0:
        for g in grasps[:20]:
            # Each g may be 4x4 transform (or flat 7-dof). Convert to 4x4
            T = np.array(g)
            if T.size == 7:
                trans = T[:3]
                quat = T[3:7]
                # trimesh expects [w,x,y,z]
                q_tm = [quat[3], quat[0], quat[1], quat[2]]
                Rm = tra.quaternion_matrix(q_tm)
                M = np.eye(4)
                M[:3, :3] = Rm[:3, :3]
                M[:3, 3] = trans
            else:
                M = T.reshape(4, 4)

            origin = M[:3, 3]
            x = origin + 0.05 * M[:3, 0]
            y = origin + 0.05 * M[:3, 1]
            z = origin + 0.05 * M[:3, 2]
            pts = [origin, x, origin, y, origin, z]
            lines = [[0, 1], [2, 3], [4, 5]]
            colors_ls = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(np.array(pts))
            ls.lines = o3d.utility.Vector2iVector(lines)
            ls.colors = o3d.utility.Vector3dVector(colors_ls)
            line_sets.append(ls)

    # Try OffscreenRenderer first
    try:
        from open3d.visualization.rendering import OffscreenRenderer, MaterialRecord

        r = OffscreenRenderer(size, size)
        mat = MaterialRecord()
        mat.shader = "defaultUnlit"
        r.scene.set_background([1, 1, 1, 1])
        r.scene.add_geometry("pc", pcd, mat)
        for i, ls in enumerate(line_sets):
            r.scene.add_geometry(f"grasp_{i}", ls, mat)

        img = r.render_to_image()
        o3d.io.write_image(save_path, img)
        return
    except Exception:
        pass

    # Fallback: use hidden Visualizer + capture_screen_float_buffer
    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=size, height=size)
        vis.add_geometry(pcd)
        for ls in line_sets:
            vis.add_geometry(ls)
        vis.poll_events()
        vis.update_renderer()
        img = vis.capture_screen_float_buffer(do_render=True)
        vis.destroy_window()
        img_np = (np.asarray(img) * 255).astype(np.uint8)
        imageio.imwrite(save_path, img_np)
        return
    except Exception as e:
        raise RuntimeError(f"Open3D render failure: {e}")

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

    try:
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
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return np.empty((0, 3), dtype=np.float32), None


def main():
    args = parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    print(f"Loading gripper config from {args.gripper_config}")
    grasp_cfg = load_grasp_cfg(args.gripper_config)
    sampler = GraspGenSampler(grasp_cfg)

    # Ensure model and all its parameters/buffers live on the correct device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        # Force move to device and double-check for any mismatched param/buffer
        sampler.model.to(device)
    except Exception:
        # best-effort; continue but warn
        print("Warning: failed to fully move model to device; attempting a cuda() call")
        try:
            sampler.model.cuda()
        except Exception:
            pass

    # Best-effort: force move all parameters and buffers to the target device
    try:
        for name, p in sampler.model.named_parameters():
            if p is not None:
                p.data = p.data.to(device)
                if p.grad is not None:
                    p.grad.data = p.grad.data.to(device)
        for name, b in sampler.model.named_buffers():
            if b is not None:
                try:
                    b.data = b.data.to(device)
                except Exception:
                    pass

        # Also try to move any tensor attributes on modules (some extensions store tensors directly)
        for m in sampler.model.modules():
            for attr_name, attr_val in list(vars(m).items()):
                if isinstance(attr_val, torch.Tensor) and attr_val.device != device:
                    try:
                        setattr(m, attr_name, attr_val.to(device))
                    except Exception:
                        pass
    except Exception as e:
        print(f"Warning: failed to move some model tensors: {e}")

    # Debugging info: print unique devices of parameters and buffers to detect mismatches
    param_devices = set([str(p.device) for p in sampler.model.parameters()])
    buffer_devices = set([str(b.device) for b in sampler.model.buffers()])
    print(f"Model parameter devices: {param_devices}")
    print(f"Model buffer devices: {buffer_devices}")

    # Scan for any tensor attributes on modules that are not registered as parameters/buffers
    mismatched_attrs = []
    for m_name, m in sampler.model.named_modules():
        for attr_name, attr_val in list(vars(m).items()):
            if isinstance(attr_val, torch.Tensor):
                if str(attr_val.device) != str(device):
                    mismatched_attrs.append((m_name or "<root>", type(m).__name__, attr_name, str(attr_val.device), tuple(attr_val.shape)))
                    # Best-effort move
                    try:
                        setattr(m, attr_name, attr_val.to(device))
                    except Exception:
                        pass

    if len(mismatched_attrs) > 0:
        print("Found non-parameter tensor attributes that were on the wrong device:")
        for item in mismatched_attrs:
            print(" module:", item[0], "class:", item[1], "attr:", item[2], "device:", item[3], "shape:", item[4])
    else:
        print("No stray tensor attributes found off-device")
    
    object_dirs = glob.glob(os.path.join(args.data_dir, "*"))
    object_dirs = [d for d in object_dirs if os.path.isdir(d)]
    object_dirs.sort()
    
    print(f"Found {len(object_dirs)} objects in {args.data_dir}")

    # Prepare a Viser server for visualization (single shared server)
    server = None
    if args.visualize:
        if not VISER_AVAILABLE:
            print("Viser not available; visualization disabled.")
        else:
            server = vutils.create_visualizer()
    
    for obj_dir in tqdm(object_dirs, desc="Processing objects"):
        obj_name = os.path.basename(obj_dir)
        local_pc_path = os.path.join(obj_dir, "local_pc.ply")
        
        if not os.path.exists(local_pc_path):
            # print(f"Skipping {obj_name}: local_pc.ply not found")
            continue
            
        # Load point cloud
        pc_points, pc_colors = load_ply_as_points(local_pc_path)
        
        if pc_points.size == 0:
            print(f"Skipping {obj_name}: Empty point cloud")
            continue
            
        # Ensure point cloud is a torch tensor on model device to avoid device mismatch
        try:
            if isinstance(pc_points, np.ndarray):
                pc_t = torch.from_numpy(pc_points).float().to(next(sampler.model.parameters()).device)
            elif isinstance(pc_points, torch.Tensor):
                pc_t = pc_points.float().to(next(sampler.model.parameters()).device)
            else:
                pc_t = torch.tensor(pc_points, dtype=torch.float32, device=next(sampler.model.parameters()).device)
        except Exception:
            pc_t = pc_points

        # Run inference (disable internal outlier-removal to avoid creating CPU tensors)
        grasps, scores = GraspGenSampler.run_inference(
            pc_t,
            sampler,
            grasp_threshold=args.grasp_threshold,
            num_grasps=args.num_grasps,
            topk_num_grasps=args.topk_num_grasps,
            remove_outliers=False,
        )
        
        # Save results
        output_path = os.path.join(args.output_dir, f"{obj_name}.npz")
        
        # Convert to numpy if needed
        grasps_np = grasps.cpu().numpy() if isinstance(grasps, torch.Tensor) else grasps
        scores_np = scores.cpu().numpy() if isinstance(scores, torch.Tensor) else scores
        
        np.savez(
            output_path,
            grasps=grasps_np,
            scores=scores_np,
            pc=pc_points
        )
        
        if args.visualize:
            # Create and reuse a Viser server if available. If Viser is not
            # available we'll skip rendering but still keep saved results.
            if not VISER_AVAILABLE:
                print("Viser is not available; skipping visualization for", obj_name)
            else:
                # server already created above; fallback to creating here
                if server is None:
                    server = vutils.create_visualizer()

                # Clear previous visualization and add scene / object
                vutils.clear_visualization(server)

                global_pc_path = os.path.join(obj_dir, "global_pc.ply")
                if os.path.exists(global_pc_path):
                    global_pc, global_colors = load_ply_as_points(global_pc_path)
                    if global_pc.size > 0:
                        scene_color = global_colors if global_colors is not None else np.array([180, 180, 180], dtype=np.uint8)
                        vutils.visualize_pointcloud(server, "scene", global_pc, color=scene_color, size=0.0025)

                # visualize object points (local)
                    if pc_points.size > 0:
                        object_color = pc_colors if pc_colors is not None else np.array([0, 255, 0], dtype=np.uint8)
                        vutils.visualize_pointcloud(server, "object", pc_points, color=object_color, size=0.003)
                        try:
                            vutils.focus_camera_on_points(server, pc_points)
                        except AttributeError:
                            pass

                # visualize predicted grasps (top-k colors)
                if isinstance(grasps_np, (list, tuple)):
                    # empty results may be lists
                    grasps_np = np.array(grasps_np)
                if grasps_np is None:
                    grasps_np = np.array([])

                if grasps_np.size != 0:
                    try:
                        colors = vutils.get_color_from_score(scores_np, use_255_scale=True)
                    except Exception:
                        colors = None

                    for i, g in enumerate(grasps_np):
                        col = [int(c) for c in colors[i]] if colors is not None and i < len(colors) else [255, 0, 0]
                        vutils.visualize_grasp(server, f"predicted_grasps/{i:03d}/grasp", g, color=col, gripper_name=grasp_cfg.data.gripper_name)

                # capture a rendered image
                snapshot_path = os.path.join(args.output_dir, f"{obj_name}.png")
                # Prefer Open3D for headless, non-blocking screenshots.
                if O3D_AVAILABLE:
                    try:
                        render_open3d_snapshot(pc_points, grasps_np, scores_np, snapshot_path, size=1024, pc_colors=pc_colors)
                    except Exception as e:
                        print(f"Open3D snapshot failed for {obj_name}: {e}")
                        # fall back to Viser capture if available
                        if VISER_AVAILABLE:
                            try:
                                capture_viser_snapshot(server, snapshot_path, wait_s=3.0)
                            except Exception as e2:
                                print(f"Viser snapshot fallback failed for {obj_name}: {e2}")
                else:
                    try:
                        capture_viser_snapshot(server, snapshot_path, wait_s=10.0)
                    except Exception:
                        # one final attempt with shorter wait before giving up
                        try:
                            capture_viser_snapshot(server, snapshot_path, wait_s=1.0)
                        except Exception as e:
                            print(f"Viser snapshot failed for {obj_name}: {e}")
                # snapshot attempts completed
        
    print(f"Inference complete. Results saved to {args.output_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description="Run GraspGen inference on Utopia dataset objects")
    parser.add_argument("--data_dir", type=str, default="GraspGenModels/obj_pc", help="Directory containing object folders")
    parser.add_argument("--output_dir", type=str, default="outputs/utopia_inference", help="Directory to save results")
    parser.add_argument("--gripper_config", type=str, default="GraspGenModels/checkpoints/graspgen_franka_panda.yml", help="Path to gripper configuration YAML file")
    parser.add_argument("--grasp_threshold", type=float, default=0.8, help="Threshold for valid grasps")
    parser.add_argument("--num_grasps", type=int, default=200, help="Number of grasps to generate")
    parser.add_argument("--topk_num_grasps", type=int, default=-1, help="Number of top grasps to return")
    parser.add_argument("--visualize", action="store_true", help="Render predicted grasps and save snapshot")
    return parser.parse_args()

if __name__ == "__main__":
    main()
