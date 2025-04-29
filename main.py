import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import measure
import trimesh
from trimesh.smoothing import filter_laplacian
from objects import Sphere, Cylinder, Tube
from utils import save_parameters  # Import only save_parameters
from concurrent.futures import ProcessPoolExecutor

def compute_sdf_capsule(args):
    """Compute the SDF for a single capsule."""
    A, B, X, Y, Z, r_tube = args
    AB = B - A
    AB_len2 = np.dot(AB, AB)

    # Vector from A to each voxel
    AP = np.stack((X - A[0], Y - A[1], Z - A[2]), axis=-1)  # shape (res,res,res,3)
    t = np.clip((AP @ AB) / AB_len2, 0.0, 1.0)              # projection factor
    proj = A + t[..., None] * AB                            # closest points on AB
    diff = np.stack((X, Y, Z), axis=-1) - proj
    dist = np.linalg.norm(diff, axis=-1)
    return dist - r_tube

def main(shape_type, distribution_type, connection_type, R_sphere, H_cylinder, R_cylinder,
         R_inner_tube, R_outer_tube, N_pts, k_nn, r_tube, res, sigma_blur, lap_iters, seed, output):
    np.random.seed(seed)  # Set seed for reproducibility
    print("Step 1: Sampling points...")
    if shape_type == "sphere":
        shape = Sphere(R_sphere)
    elif shape_type == "cylinder":
        shape = Cylinder(R_cylinder, H_cylinder)
    elif shape_type == "tube":
        shape = Tube(R_inner_tube, R_outer_tube, H_cylinder)

    if distribution_type == "inside":
        points = shape.generate_points_inside(N_pts, seed=seed)
    elif distribution_type == "surface":
        points = shape.generate_points_on_surface(N_pts, seed=seed)

    print("Step 2: Finding connection pairs...")
    pairs = shape.find_connection_pairs(points, connection_type, k_nn, seed=seed)

    print("Step 3: Building 3D grid...")
    if shape_type == "sphere":
        margin = R_sphere * 0.1
        lin = np.linspace(-R_sphere - margin, R_sphere + margin, res)
    elif shape_type == "cylinder":
        margin = max(H_cylinder, R_cylinder) * 0.1
        lin = np.linspace(-R_cylinder - margin, R_cylinder + margin, res)
    elif shape_type == "tube":
        margin = max(H_cylinder, R_outer_tube) * 0.3
        lin = np.linspace(-R_outer_tube - margin, R_outer_tube + margin, res)

    X, Y, Z = np.meshgrid(lin, lin, lin, indexing='ij')

    print("Step 4: Initializing field for union of capsules...")
    F_union = np.full((res, res, res), np.inf, dtype=np.float32)

    print("Step 5: Carving out each capsule SDF...")
    with ProcessPoolExecutor() as executor:
        sdf_results = list(executor.map(
            compute_sdf_capsule,
            [(points[i], points[j], X, Y, Z, r_tube) for (i, j) in pairs]
        ))

    for sdf_capsule in sdf_results:
        F_union = np.minimum(F_union, sdf_capsule)

    print("Step 6: Applying Gaussian blur...")
    F_smooth = gaussian_filter(F_union, sigma=sigma_blur)

    print("Step 7: Extracting mesh with marching cubes...")
    spacing = (lin[1] - lin[0],) * 3
    verts, faces, normals, _ = measure.marching_cubes(
        F_smooth, level=0.0, spacing=spacing)

    print("Step 8: Smoothing the mesh...")
    mesh = trimesh.Trimesh(vertices=verts, faces=faces,
                           vertex_normals=normals)
    filter_laplacian(mesh, iterations=lap_iters)

    print("Step 9: Exporting the mesh...")
    mesh.export(output)
    print(f"Exported → {output}")

    print("Step 10: Saving parameters...")
    save_parameters(
        output,
        shape_type=shape_type,
        distribution_type=distribution_type,
        connection_type=connection_type,
        R_sphere=R_sphere,
        H_cylinder=H_cylinder,
        R_cylinder=R_cylinder,
        R_inner_tube=R_inner_tube,
        R_outer_tube=R_outer_tube,
        N_pts=N_pts,
        k_nn=k_nn,
        r_tube=r_tube,
        res=res,
        sigma_blur=sigma_blur,
        lap_iters=lap_iters,
        seed=seed
    )