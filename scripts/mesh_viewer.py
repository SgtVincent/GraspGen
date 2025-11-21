# %%
import argparse
import json
import yaml
import time
from pathlib import Path
from typing import Any, List, Sequence, Tuple, Union

import numpy as np
import trimesh
import viser
from viser.extras import ViserUrdf
from grasp_gen.dataset.eval_utils import (
    load_from_isaac_grasp_format,
    load_urdf_scene,
    dump_merged_urdf_mesh,
    find_base_config_size,
)
from grasp_gen.utils import viser_utils as vutils

DEFAULT_URDF = "/models/custom_objects/real_bottle_soso/color_0024_0.urdf"
DEFAULT_ANNOTATION = "/models/custom_objects/real_bottle_soso/annotation.json"

###################################
# Vscode remote debug configuration
###################################
# import debugpy

# debugpy.listen(5678)
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()
# print("Debugger attached.")
###################################

print(f"Viser version: {viser.__version__}")
print("Update `DEFAULT_URDF` and `DEFAULT_ANNOTATION` if you want to visualize another asset.")

def tinted_mesh(mesh: trimesh.Trimesh, rgb: Tuple[float, float, float]):
    colored = mesh.copy()
    rgba = np.clip(np.array([rgb[0], rgb[1], rgb[2], 255]), 0, 255).astype(np.uint8)
    colored.visual.vertex_colors = np.tile(rgba, (len(colored.vertices), 1))  # type: ignore[attr-defined]
    return colored


def compute_axis_aligned_box(points: Union[Sequence[Sequence[float]], np.ndarray]):
    pts = np.asarray(points)
    if pts.size == 0:
        return None
    min_corner = pts.min(axis=0)
    max_corner = pts.max(axis=0)
    center = (min_corner + max_corner) / 2.0
    dimensions = max_corner - min_corner
    return center, dimensions


def compute_oriented_box_mesh(points: Union[Sequence[Sequence[float]], np.ndarray]):
    pts = np.asarray(points)
    if pts.shape[0] < 3:
        return None
    point_cloud = trimesh.points.PointCloud(pts)
    return point_cloud.bounding_box_oriented.to_mesh()


def create_joint_control_sliders(
    server: viser.ViserServer, viser_urdf: ViserUrdf
) -> Tuple[List[Any], List[float]]:
    slider_handles: List[Any] = []
    initial_config: List[float] = []
    joint_limits = viser_urdf.get_actuated_joint_limits()
    if not joint_limits:
        server.gui.add_markdown("No actuated joints detected in the URDF.")
        return slider_handles, initial_config

    def _update_cfg(_: Any) -> None:
        if not slider_handles:
            return
        cfg = np.array([slider.value for slider in slider_handles], dtype=float)
        viser_urdf.update_cfg(cfg)

    for joint_name, (lower, upper) in joint_limits.items():
        lower_val = lower if lower is not None else -np.pi
        upper_val = upper if upper is not None else np.pi
        if lower_val > upper_val:
            lower_val, upper_val = upper_val, lower_val
        if lower_val < -0.1 and upper_val > 0.1:
            initial_pos = 0.0
        else:
            initial_pos = (lower_val + upper_val) / 2.0
        slider = server.gui.add_slider(
            label=joint_name,
            min=lower_val,
            max=upper_val,
            step=1e-3,
            initial_value=initial_pos,
        )
        slider_handles.append(slider)
        slider.on_update(_update_cfg)
        initial_config.append(initial_pos)
    return slider_handles, initial_config


def add_visibility_controls(
    server: viser.ViserServer,
    viser_urdf: ViserUrdf,
    load_meshes: bool,
    load_collision_meshes: bool,
) -> None:
    with server.gui.add_folder("Visibility"):
        show_meshes_cb = server.gui.add_checkbox(
            "Show meshes",
            initial_value=viser_urdf.show_visual,
        )
        show_collision_cb = server.gui.add_checkbox(
            "Show collision meshes",
            initial_value=viser_urdf.show_collision,
        )

    @show_meshes_cb.on_update
    def _(_: Any) -> None:
        viser_urdf.show_visual = show_meshes_cb.value

    @show_collision_cb.on_update
    def _(_: Any) -> None:
        viser_urdf.show_collision = show_collision_cb.value

    show_meshes_cb.visible = load_meshes
    show_collision_cb.visible = load_collision_meshes


def add_reset_button(
    server: viser.ViserServer,
    slider_handles: List[Any],
    initial_config: List[float],
) -> None:
    if not slider_handles:
        return
    reset_button = server.gui.add_button("Reset joints")

    @reset_button.on_click
    def _(_: Any) -> None:
        for slider, init_q in zip(slider_handles, initial_config):
            slider.value = init_q


def add_annotation_visuals(server: viser.ViserServer, annotation_path: Path, scale: float = 1.0) -> None:
    if not annotation_path.exists():
        print(f"Warning: annotation file not found at {annotation_path}")
        return
    with annotation_path.open('r') as fp:
        annotations = json.load(fp)
    affordable_points = annotations.get('affordable_points', [])
    if not affordable_points:
        return
    pts = np.asarray(affordable_points, dtype=float) * scale
    for point_idx, point in enumerate(pts):
        server.scene.add_icosphere(
            f"/affordance/point_{point_idx}",
            radius=0.008,
            color=(255, 80, 80),
            subdivisions=2,
            position=tuple(point.tolist()),
        )
    aabb = compute_axis_aligned_box(pts)
    if aabb is not None:
        center, dims = aabb
        server.scene.add_box(
            "/affordance/aabb",
            color=(80, 200, 120),
            dimensions=tuple(dims.tolist()),
            wireframe=True,
            position=tuple(center.tolist()),
            opacity=0.1,
        )
    obb_mesh = compute_oriented_box_mesh(pts)
    if obb_mesh is not None:
        tinted_obb = tinted_mesh(obb_mesh, (64, 128, 255))
        server.scene.add_mesh_trimesh(
            "/affordance/obb",
            mesh=tinted_obb,
            cast_shadow=False,
            receive_shadow=False,
        )


def configure_ground_grid(server: viser.ViserServer, viser_urdf: ViserUrdf) -> None:
    trimesh_scene = viser_urdf._urdf.scene or viser_urdf._urdf.collision_scene  # noqa: SLF001
    ground_z = trimesh_scene.bounds[0, 2] if trimesh_scene is not None else 0.0
    server.scene.add_grid(
        "/grid",
        width=2.0,
        height=2.0,
        cell_size=0.1,
        position=(0.0, 0.0, float(ground_z)),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Viser URDF viewer with affordances")
    parser.add_argument(
        "--urdf-path",
        type=Path,
        default=DEFAULT_URDF,
        help="Path to the URDF to visualize",
    )
    parser.add_argument(
        "--annotation-path",
        type=Path,
        default=DEFAULT_ANNOTATION,
        help="Path to the affordance annotation JSON",
    )
    parser.add_argument(
        "--predicted-grasps",
        type=Path,
        default=None,
        help="Path to an Isaac-format grasp YAML to visualize",
    )
    parser.add_argument(
        "--load-collision-meshes",
        action="store_true",
        help="Render collision meshes in addition to visuals",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = viser.ViserServer()
    # Do not load the original visual/collision meshes into the scene. We
    # will instead overlay a scaled mesh (controlled by base_config.size)
    # so that the scene only shows the canonical/scaled mesh while keeping
    # the underlying URDF model (for joint controls) available.
    viser_urdf = ViserUrdf(
        server,
        urdf_or_path=args.urdf_path,
        load_meshes=False,  # prevent original visuals from being added
        load_collision_meshes=False,  # prevent original collision visuals
        collision_mesh_color_override=(1.0, 0.0, 0.0, 0.4),
    )

    server.scene.configure_default_lights(enabled=True, cast_shadow=True)
    configure_ground_grid(server, viser_urdf)

    with server.gui.add_folder("Joint position control"):
        slider_handles, initial_config = create_joint_control_sliders(server, viser_urdf)

    # Keep the original show-controls but they now toggle whether the overlay
    # scaled mesh is visible. Since the original URDF visuals are not loaded,
    # we still expose the same UI and map it to the overlay mesh.
    add_visibility_controls(server, viser_urdf, load_meshes=True, load_collision_meshes=False)
    add_reset_button(server, slider_handles, initial_config)
    # Detect base_config.size and overlay a scaled mesh + scale annotations
    scale_factor = find_base_config_size(str(args.urdf_path)) or 1.0
    try:
        merged = dump_merged_urdf_mesh(str(args.urdf_path))
        if merged is not None and float(scale_factor) != 1.0:
            scaled_mesh = merged.copy()
            scaled_mesh.apply_scale(float(scale_factor))
            tinted = tinted_mesh(scaled_mesh, (180, 180, 180))
            server.scene.add_mesh_trimesh(
                "/scaled_object",
                mesh=tinted,
                cast_shadow=False,
                receive_shadow=False,
            )
    except Exception as e:
        print(f"Failed to apply base_config scale: {e}")

    add_annotation_visuals(server, args.annotation_path, scale=scale_factor)
    if args.predicted_grasps is not None and args.predicted_grasps.exists():
        try:
            grasps, confs = load_from_isaac_grasp_format(str(args.predicted_grasps))
        except Exception as e:
            print(f"Failed to load predicted grasps: {e}")
            grasps, confs = None, None
        if grasps is not None and len(grasps) > 0:
            colors = vutils.get_color_from_score(confs, use_255_scale=True)
            for i, g in enumerate(grasps):
                color = [int(c) for c in colors[i]] if colors is not None else [255, 0, 0]
                # The predicted grasps are expected to be in the same coordinate
                # frame as the scaled mesh (i.e. after base_config.size was
                # applied). Do not re-scale the transforms here or the gripper
                # will be positioned incorrectly.
                display_g = np.array(g).copy()

                vutils.visualize_grasp(
                    server,
                    f"predicted_grasps/{i:03d}/grasp",
                    display_g,
                    color=color,
                    gripper_name="franka_panda",
                )

    if slider_handles:
        viser_urdf.update_cfg(np.array(initial_config, dtype=float))

    viewer_url = getattr(server, "viewer_url", "http://localhost:8080")
    print(f"Viser server is running at {viewer_url}")
    print("Press Ctrl+C to exit")

    try:
        while True:
            time.sleep(10.0)
    except KeyboardInterrupt:
        print("Shutting down viewer")


if __name__ == "__main__":
    main()



