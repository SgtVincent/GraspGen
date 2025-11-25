import argparse
import time
from pathlib import Path
import numpy as np
import trimesh
import viser

# Default path relative to this script
DEFAULT_OBJECT_DIR = Path(__file__).parent.parent / "GraspGenModels/obj_pc/apple"

def load_point_cloud(path: Path) -> np.ndarray:
    """Load a point cloud from a file and return vertices as numpy array."""
    pc = trimesh.load(path)
    # trimesh.load can return a Scene, PointCloud, or Trimesh
    if isinstance(pc, trimesh.Scene):
        # If it's a scene, try to dump it to a single mesh/pointcloud or just take vertices if possible
        # For simple PLY files, it usually returns PointCloud or Trimesh
        if len(pc.geometry) > 0:
            # Combine vertices from all geometries
            vertices = []
            for g in pc.geometry.values():
                if hasattr(g, 'vertices'):
                    vertices.append(g.vertices)
            if vertices:
                return np.vstack(vertices)
        return np.array([])
    elif hasattr(pc, 'vertices'):
        return np.array(pc.vertices)
    else:
        raise ValueError(f"Could not load point cloud vertices from {path}")

def compute_axis_aligned_box(points: np.ndarray):
    """Compute the center and dimensions of an axis-aligned bounding box."""
    if points.size == 0:
        return None
    min_corner = points.min(axis=0)
    max_corner = points.max(axis=0)
    center = (min_corner + max_corner) / 2.0
    dimensions = max_corner - min_corner
    return center, dimensions

def main():
    parser = argparse.ArgumentParser(description="Viewer for Utopia objects (global PC and local PC bbox)")
    parser.add_argument(
        "--object-dir",
        type=Path,
        default=DEFAULT_OBJECT_DIR,
        help="Path to the object directory containing global_pc.ply and local_pc.ply",
    )
    args = parser.parse_args()

    if not args.object_dir.exists():
        print(f"Error: Directory {args.object_dir} does not exist.")
        return

    global_pc_path = args.object_dir / "global_pc.ply"
    local_pc_path = args.object_dir / "local_pc.ply"

    server = viser.ViserServer()
    server.scene.configure_default_lights(enabled=True, cast_shadow=True)
    server.scene.add_grid("/grid", width=2.0, height=2.0, cell_size=0.1, plane="xy")

    # Load and visualize global PC
    if global_pc_path.exists():
        print(f"Loading global PC from {global_pc_path}")
        try:
            global_points = load_point_cloud(global_pc_path)
            if global_points.size > 0:
                # Create colors array (N, 3) uint8
                colors = np.tile(np.array([200, 200, 200], dtype=np.uint8), (len(global_points), 1))
                server.scene.add_point_cloud(
                    "/global_pc",
                    points=global_points,
                    colors=colors,
                    point_size=0.002,
                )
                print(f"Loaded {len(global_points)} points for global PC")
            else:
                print("Global PC is empty")
        except Exception as e:
            print(f"Failed to load global PC: {e}")
    else:
        print(f"Warning: {global_pc_path} not found.")

    # Load local PC and visualize bbox
    if local_pc_path.exists():
        print(f"Loading local PC from {local_pc_path}")
        try:
            local_points = load_point_cloud(local_pc_path)
            if local_points.size > 0:
                aabb = compute_axis_aligned_box(local_points)
                if aabb is not None:
                    center, dims = aabb
                    print(f"Local PC BBox Center: {center}, Dims: {dims}")
                    server.scene.add_box(
                        "/local_pc_bbox",
                        position=center,
                        dimensions=dims,
                        color=(255, 100, 100),
                        wireframe=True,
                    )
                    # Also add a semi-transparent box to make it easier to see volume
                    server.scene.add_box(
                        "/local_pc_bbox_filled",
                        position=center,
                        dimensions=dims,
                        color=(255, 100, 100),
                        opacity=0.1,
                    )
            else:
                print("Local PC is empty")
        except Exception as e:
            print(f"Failed to load local PC: {e}")
    else:
        print(f"Warning: {local_pc_path} not found.")

    # Get URL properly
    # server.get_url() might not be available in older versions, checking mesh_viewer usage
    # mesh_viewer uses getattr(server, "viewer_url", ...)
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
