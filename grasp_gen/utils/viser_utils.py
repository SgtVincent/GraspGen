# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Utility functions for visualization using Viser.
This mirrors the API of `meshcat_utils` used by the demos so they can swap
visualizers by setting a flag.
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import trimesh
import trimesh.transformations as tra
try:
    import viser
    VISER_AVAILABLE = True
except Exception:
    VISER_AVAILABLE = False
from grasp_gen.robot import load_control_points_for_visualization
from grasp_gen.utils.logging_config import get_logger

logger = get_logger(__name__)


def get_color_from_score(labels, use_255_scale=False):
    scale = 255.0 if use_255_scale else 1.0
    if type(labels) in [np.float32, float]:
        return scale * np.array([1 - labels, labels, 0])
    else:
        score = scale * np.stack(
            [np.ones(labels.shape[0]) - labels, labels, np.zeros(labels.shape[0])],
            axis=1,
        )
        return score.astype(np.int32)


def create_visualizer(clear=True):
    """Create a Viser server instance and return it."""
    if not VISER_AVAILABLE:
        raise ImportError(
            "Viser is not installed. Install it or run without --use-viser flag."
        )
    logger.info("Starting Viser server for visualization...")
    server = viser.ViserServer()
    # Viser doesn't expose an explicit delete on scene; keep as is
    # Cache some server capabilities for later to avoid repeated introspection.
    try:
        server._supports_set_object_transform = _supports_set_object_transform(server)
    except Exception:
        server._supports_set_object_transform = False
    # No need to set additional pointcloud signature caches in the simplified viewer.
    return server


def _tinted_mesh(mesh: trimesh.Trimesh, rgb: Tuple[int, int, int]):
    colored = mesh.copy()
    rgba = np.clip(np.array([rgb[0], rgb[1], rgb[2], 255]), 0, 255).astype(np.uint8)
    colored.visual.vertex_colors = np.tile(rgba, (len(colored.vertices), 1))
    return colored


def _supports_set_object_transform(server: object) -> bool:
    try:
        return hasattr(server.scene, "set_object_transform")
    except Exception:
        return False


# NOTE: Detection helper for older Viser signatures removed to keep
# the code focused for the viewer and demo scripts. Use `add_point_cloud`
# with keyword args and explicit `point_size`/`point_shape` instead.


def visualize_mesh(
    server: Any,
    name: str,
    mesh: trimesh.Trimesh,
    color: Optional[List[int]] = None,
    transform: Optional[np.ndarray] = None,
):
    try:
        if server is None:
            return
        if color is not None:
            c = list(color)
            if len(c) < 3:
                c = (c + [255, 255, 255])[:3]
            rgb3: Tuple[int, int, int] = (int(c[0]), int(c[1]), int(c[2]))
            mesh = _tinted_mesh(mesh, rgb3)
        # If the server can't set object transforms, apply transform to the mesh itself
        if transform is not None and not getattr(server, "_supports_set_object_transform", False):
            mesh = mesh.copy()
            mesh.apply_transform(transform)

        server.scene.add_mesh_trimesh(
            f"/{name}", mesh=mesh, cast_shadow=False, receive_shadow=False
        )
        if transform is not None and getattr(server, "_supports_set_object_transform", False):
            server.scene.set_object_transform(f"/{name}", transform)
    except Exception as e:
        logger.error("Viser visualize_mesh failed: %s", e)


def visualize_pointcloud(
    server: Any,
    name: str,
    pc: np.ndarray,
    color: Optional[np.ndarray] = None,
    transform: Optional[np.ndarray] = None,
    size: float = 0.0025,
    **kwargs,
):
    try:
        if server is None:
            return
        if pc.ndim == 3:
            pc = pc.reshape(-1, pc.shape[-1])
        
        if color is not None:
            if isinstance(color, list):
                color = np.array(color)
            color = np.array(color)
            
            # If single color provided, broadcast it
            if color.ndim == 1:
                color = np.tile(color, (pc.shape[0], 1))
            elif color.ndim == 3:
                color = color.reshape(-1, color.shape[-1])
            
            # Handle float 0-1 vs uint8 0-255
            if color.dtype.kind in ('f', 'c') and color.max() <= 1.0:
                colors = (color * 255).astype(np.uint8)
            else:
                colors = color.astype(np.uint8)
        else:
            colors = np.ones((pc.shape[0], 3), dtype=np.uint8) * 255

        # Decide whether we should apply transform to the positions
        if transform is not None and not getattr(server, "_supports_set_object_transform", False):
            pc_to_add = tra.transform_points(pc, transform)
        else:
            pc_to_add = pc

        # Standardize on calling the keyword-based signature. Provide size/shape.
        point_size = float(size) if size is not None else 0.0025
        # Default shape chosen to be 'square' to match previous MeshCat style.
        point_shape = kwargs.get("point_shape", "square")
        try:
            server.scene.add_point_cloud(
                f"/{name}",
                points=pc_to_add.astype(float),
                colors=colors,
                point_size=point_size,
                point_shape=point_shape,
            )
        except Exception:
            try:
                server.scene.add_point_cloud(
                    f"/{name}",
                    points=pc_to_add.astype(float),
                    colors=colors,
                    point_size=point_size,
                )
            except Exception as e:
                logger.error("Viser add_point_cloud failed: %s", e)
        # NOTE: We intentionally avoid deprecated positional signatures
        # and always call keyword-based version supported by current Viser API.

        if transform is not None and getattr(server, "_supports_set_object_transform", False):
            server.scene.set_object_transform(f"/{name}", transform)
    except Exception as e:
        logger.error("Viser visualize_pointcloud failed: %s", e)


def _compute_focus_center_radius(points: np.ndarray):
    pts = np.asarray(points)
    if pts.size == 0:
        return None
    if pts.ndim == 1:
        try:
            pts = pts.reshape(-1, 3)
        except Exception:
            pts = pts.reshape(1, -1)
    elif pts.shape[1] > 3:
        pts = pts[:, :3]

    bbox_min = np.min(pts, axis=0)
    bbox_max = np.max(pts, axis=0)
    center = 0.5 * (bbox_min + bbox_max)
    radius = np.linalg.norm(bbox_max - bbox_min)
    if not np.isfinite(radius) or radius < 1e-4:
        try:
            radius = np.max(np.linalg.norm(pts - center, axis=1))
        except Exception:
            radius = 0.35
    if not np.isfinite(radius) or radius < 1e-4:
        radius = 0.35
    return center, radius


def _focus_client_camera(client: Any, position: np.ndarray, target: np.ndarray, up_vec: np.ndarray) -> bool:
    if client is None:
        return False
    camera = getattr(client, "camera", None)
    if camera is None:
        return False

    def _apply():
        try:
            if hasattr(camera, "position"):
                camera.position = tuple(position.tolist())
            if hasattr(camera, "look_at"):
                camera.look_at = tuple(target.tolist())
            if hasattr(camera, "up"):
                camera.up = tuple(up_vec.tolist())
        except Exception as exc:
            logger.debug("Viser client camera update failed: %s", exc)
            raise

    try:
        atomic_ctx = getattr(client, "atomic", None)
        if callable(atomic_ctx):
            with client.atomic():
                _apply()
        else:
            _apply()
        if hasattr(client, "flush"):
            try:
                client.flush()
            except Exception:
                pass
        return True
    except Exception:
        return False


def _apply_camera_focus(server: Any, position: np.ndarray, target: np.ndarray, up_vec: np.ndarray):
    scene = getattr(server, "scene", None)
    applied = False

    if scene is not None and hasattr(scene, "set_camera_look_at"):
        try:
            scene.set_camera_look_at(
                position=tuple(position.tolist()),
                look_at=tuple(target.tolist()),
                up=tuple(up_vec.tolist()),
            )
            applied = True
        except Exception as exc:
            logger.debug("Viser scene camera focus failed: %s", exc)

    clients_attr = getattr(server, "clients", None)
    client_handles: List[Any] = []
    if isinstance(clients_attr, dict):
        client_handles = list(clients_attr.values())
    elif isinstance(clients_attr, (list, tuple, set)):
        client_handles = list(clients_attr)
    elif clients_attr is not None:
        client_handles = [clients_attr]

    for handle in client_handles:
        applied = _focus_client_camera(handle, position, target, up_vec) or applied

    if hasattr(server, "on_client_connect") and not getattr(server, "_auto_focus_registered", False):
        try:
            def _apply_on_connect(client_handle: Any):
                params = getattr(server, "_last_focus_params", None)
                if not params:
                    return
                pos, tgt, up_inner = params
                _focus_client_camera(client_handle, pos, tgt, up_inner)

            server.on_client_connect(_apply_on_connect)
            server._auto_focus_registered = True
        except Exception:
            pass

    server._last_focus_params = (position, target, up_vec)
    if not applied:
        logger.debug("Viser camera focus deferred; waiting for clients")


def focus_camera_on_points(server: Any, points: np.ndarray, up: Optional[np.ndarray] = None):
    """Aim all connected Viser cameras at the bounding box of ``points``."""
    if server is None:
        return
    center_radius = _compute_focus_center_radius(points)
    if center_radius is None:
        return
    center, radius = center_radius
    up_vec = np.array([0.0, 0.0, 1.0]) if up is None else np.asarray(up, dtype=np.float32)
    distance = max(radius * 2.0, 0.35)
    position = center + np.array([distance, distance, max(distance * 0.7, 0.35)])
    _apply_camera_focus(server, position, center, up_vec)


def make_frame(server: Any, name: str, h: float = 0.15, radius: float = 0.01, o: float = 1.0, T: Optional[np.ndarray] = None):
    if server is None:
        return
    # Represent axes using boxes
    # X axis - red
    server.scene.add_box(f"/{name}/x", color=(255, 0, 0), dimensions=(h, radius, radius), position=(h / 2, 0, 0))
    server.scene.add_box(f"/{name}/y", color=(0, 255, 0), dimensions=(radius, h, radius), position=(0, h / 2, 0))
    server.scene.add_box(f"/{name}/z", color=(0, 0, 255), dimensions=(radius, radius, h), position=(0, 0, h / 2))
    if T is not None and getattr(server, "_supports_set_object_transform", False):
        server.scene.set_object_transform(f"/{name}", T)


def load_visualization_gripper_points(gripper_name: str = "franka_panda") -> List[np.ndarray]:
    ctrl_points = []
    for ctrl_pts in load_control_points_for_visualization(gripper_name):
        ctrl_pts = np.array(ctrl_pts, dtype=np.float32)
        ctrl_pts = np.hstack([ctrl_pts, np.ones([len(ctrl_pts), 1])])
        ctrl_pts = ctrl_pts.T
        ctrl_points.append(ctrl_pts)
    return ctrl_points


def visualize_grasp(
    server: Any,
    name: str,
    transform: np.ndarray,
    color: List[int] = [255, 0, 0],
    gripper_name: str = "franka_panda",
    linewidth: float = 0.6,
    **kwargs,
):
    try:
        if server is None:
            return
        grasp_vertices = load_visualization_gripper_points(gripper_name)
        # Compute Tmat once for this grasp
        T = np.array(transform)
        if T.ndim == 1 and T.shape[0] == 7:
            trans = T[:3]
            quat = T[3:7]
            Tmat = tra.quaternion_matrix([quat[-1], quat[0], quat[1], quat[2]])
            Tmat[:3, 3] = trans
        else:
            Tmat = np.array(T)
        supports_transform = getattr(server, "_supports_set_object_transform", False)
        # Determine color tuple
        if isinstance(color, (list, np.ndarray)):
            c_list = list(color)[:3]
            if len(c_list) < 3:
                c_list = (c_list + [0, 0, 0])[:3]
            col = (int(c_list[0]), int(c_list[1]), int(c_list[2]))
        elif isinstance(color, float) or isinstance(color, np.floating):
            # map score to RGB via existing helper
            c_arr = get_color_from_score(color, use_255_scale=True)
            col = (int(c_arr[0]), int(c_arr[1]), int(c_arr[2]))
        else:
            col = (255, 0, 0)
        pts_list = []
        for i, grasp_vertex in enumerate(grasp_vertices):
            # grasp_vertex may be shape (4, N) or (N, d) where d>=3. Convert to homogeneous Nx4
            gv = np.array(grasp_vertex)
            if gv.ndim == 2 and gv.shape[0] == 4:
                hom = gv
            elif gv.ndim == 2 and gv.shape[1] >= 3:
                coords = gv[:, :3]
                hom = np.hstack([coords, np.ones((coords.shape[0], 1))]).T
            else:
                # Fallback: attempt to reshape
                coords = gv.reshape(-1, 3)
                hom = np.hstack([coords, np.ones((coords.shape[0], 1))]).T
            pts = (Tmat @ hom).T[:, :3]
            pts_list.append(pts)
        # Try drawing a polyline if the Viser scene supports it
        try:
            # Use server.scene.add_line or add_polyline if available
            if hasattr(server.scene, "add_line"):
                for i, pts in enumerate(pts_list):
                    server.scene.add_line(f"/{name}/{i}", points=pts, color=col)
            elif hasattr(server.scene, "add_polyline"):
                for i, pts in enumerate(pts_list):
                    server.scene.add_polyline(f"/{name}/{i}", points=pts, color=col)
            else:
                raise AttributeError("no add_line or add_polyline on server.scene")
        except Exception:
            # Fallback: draw cylinders between consecutive points to represent lines
            for i, pts in enumerate(pts_list):
                for j in range(pts.shape[0] - 1):
                    p1 = pts[j]
                    p2 = pts[j + 1]
                    segment_vec = p2 - p1
                    length = float(np.linalg.norm(segment_vec))
                    if length <= 1e-6:
                        continue
                    # Create cylinder along z-axis with length 1.0 and apply transform
                    cyl = trimesh.creation.cylinder(radius=0.0015, height=1.0, sections=8)
                    # Compute transform: align z-axis to segment_vec and scale to length
                    z_axis = np.array([0.0, 0.0, 1.0])
                    axis_dir = segment_vec / length
                    rot_axis = np.cross(z_axis, axis_dir)
                    rot_axis_norm = np.linalg.norm(rot_axis)
                    if rot_axis_norm < 1e-6:
                        rot_mat = np.eye(4)
                    else:
                        rot_axis_unit = rot_axis / rot_axis_norm
                        angle = float(np.arccos(np.dot(z_axis, axis_dir)))
                        rot_mat = tra.rotation_matrix(angle, rot_axis_unit)
                    # Scale Z=length
                    scale_mat = np.diag([1.0, 1.0, length, 1.0])
                    # translation
                    center = (p1 + p2) / 2.0
                    trans_mat = tra.translation_matrix(tuple(center.tolist()))
                    Tseg = trans_mat @ rot_mat @ scale_mat
                    # Apply transform to cylinder
                    cyl.apply_transform(Tseg)
                    if col is not None:
                        cyl = _tinted_mesh(cyl, col)
                    try:
                        server.scene.add_mesh_trimesh(f"/{name}/{i}/segment_{j}", mesh=cyl, cast_shadow=False, receive_shadow=False)
                    except Exception as e:
                        logger.error("Viser failed to add cylinder segment for grasp: %s", e)
        if supports_transform:
            try:
                server.scene.set_object_transform(f"/{name}", Tmat)
            except Exception as e:
                logger.debug("Unable to set parent transform %s: %s", name, e)
    except Exception as e:
        logger.error("Viser visualize_grasp failed: %s", e)


def clear_visualization(server: Any):
    try:
        if server is None:
            return
        # Prefer the Viser API `reset()` which clears the scene and GUI.
        # Older/newer Viser versions may not expose `clear()` — use `reset`
        # and if that's unavailable fall back to removing handles individually.
        if hasattr(server, "scene") and hasattr(server.scene, "reset"):
            try:
                server.scene.reset()
                return
            except Exception:
                # fall through to handle-by-handle removal
                pass

        # Fallback: iterate through scene handles and call remove() where available.
        try:
            # Try to safely iterate and remove nodes/handles
            if hasattr(server.scene, "get_all_handles"):
                handles = server.scene.get_all_handles()
            elif hasattr(server.scene, "handles"):
                handles = list(server.scene.handles.values())
            else:
                handles = []

            for h in handles:
                try:
                    if hasattr(h, "remove"):
                        h.remove()
                    elif hasattr(server.scene, "remove"):
                        # Some versions expose a scene.remove(name)
                        name = getattr(h, "name", None)
                        if name is not None:
                            server.scene.remove(name)
                except Exception:
                    continue
        except Exception:
            # If removal fails, just log and continue - nothing else to do
            pass
    except Exception as e:
        logger.error("Viser clear_visualization failed: %s", e)
