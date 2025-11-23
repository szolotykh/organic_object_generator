import numpy as np
import cupy as cp
import cupyx.scipy.ndimage
from skimage import measure
import trimesh
from trimesh.smoothing import filter_laplacian
from objects import Sphere, Cylinder, Tube
from utils import save_parameters

def compute_sdf_capsules_batch(grid_coords, A, B, r_tube):
    """
    Compute the minimum SDF for a batch of capsules on the GPU.
    grid_coords: (N_grid, 3)
    A, B: (N_batch, 3) start and end points of capsules
    """
    # Expand dimensions for broadcasting: (N_grid, N_batch, 3)
    P = grid_coords[:, None, :]
    A = A[None, :, :]
    B = B[None, :, :]
    
    AB = B - A
    AP = P - A
    
    AB_len2 = cp.sum(AB * AB, axis=-1)
    # Avoid division by zero
    AB_len2 = cp.maximum(AB_len2, 1e-8)
    
    t = cp.sum(AP * AB, axis=-1) / AB_len2
    t = cp.clip(t, 0.0, 1.0)
    
    # Closest point on segment: Proj = A + t * AB
    Proj = A + t[..., None] * AB
    
    dist_sq = cp.sum((P - Proj)**2, axis=-1)
    dist = cp.sqrt(dist_sq) - r_tube
    
    # Return the minimum distance across this batch of capsules for each grid point
    return cp.min(dist, axis=1)

def generate_mesh(shape_type, distribution_type, connection_type, R_sphere, H_cylinder, R_cylinder,
                  R_inner_tube, R_outer_tube, N_pts, k_nn, r_tube, res, sigma_blur, lap_iters, seed):
    """
    Generates the trimesh object based on parameters.
    Returns: trimesh.Trimesh object
    """
    np.random.seed(seed)
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

    print("Step 3: Building 3D grid on GPU...")
    if shape_type == "sphere":
        margin = R_sphere * 0.1
        lin = np.linspace(-R_sphere - margin, R_sphere + margin, res)
    elif shape_type == "cylinder":
        margin = max(H_cylinder, R_cylinder) * 0.1
        lin = np.linspace(-R_cylinder - margin, R_cylinder + margin, res)
    elif shape_type == "tube":
        margin = max(H_cylinder, R_outer_tube) * 0.3
        lin = np.linspace(-R_outer_tube - margin, R_outer_tube + margin, res)

    # Create grid on GPU
    lin_gpu = cp.array(lin)
    X, Y, Z = cp.meshgrid(lin_gpu, lin_gpu, lin_gpu, indexing='ij')
    grid_coords = cp.stack((X, Y, Z), axis=-1).reshape(-1, 3)

    print("Step 4: Initializing field for union of capsules...")
    F_union_flat = cp.full(grid_coords.shape[0], cp.inf, dtype=cp.float32)

    print("Step 5: Carving out each capsule SDF (GPU)...")
    points_gpu = cp.array(points)
    pair_indices = np.array(list(pairs))

    if len(pair_indices) > 0:
        A_all = points_gpu[pair_indices[:, 0]]
        B_all = points_gpu[pair_indices[:, 1]]
        
        # Process in batches to manage VRAM usage
        batch_size = 32 
        num_pairs = len(pair_indices)
        
        for i in range(0, num_pairs, batch_size):
            end = min(i + batch_size, num_pairs)
            A_batch = A_all[i:end]
            B_batch = B_all[i:end]
            
            sdf_batch = compute_sdf_capsules_batch(grid_coords, A_batch, B_batch, r_tube)
            F_union_flat = cp.minimum(F_union_flat, sdf_batch)

    F_union = F_union_flat.reshape(res, res, res)

    print("Step 6: Applying Gaussian blur (GPU)...")
    F_smooth = cupyx.scipy.ndimage.gaussian_filter(F_union, sigma=sigma_blur)

    print("Step 7: Extracting mesh with marching cubes...")
    # Transfer data back to CPU for marching cubes
    F_smooth_cpu = F_smooth.get()
    spacing = (lin[1] - lin[0],) * 3
    verts, faces, normals, _ = measure.marching_cubes(
        F_smooth_cpu, level=0.0, spacing=spacing)

    print("Step 8: Smoothing the mesh...")
    mesh = trimesh.Trimesh(vertices=verts, faces=faces,
                           vertex_normals=normals)
    filter_laplacian(mesh, iterations=lap_iters)
    
    return mesh

def main(shape_type, distribution_type, connection_type, R_sphere, H_cylinder, R_cylinder,
         R_inner_tube, R_outer_tube, N_pts, k_nn, r_tube, res, sigma_blur, lap_iters, seed, output):
    
    mesh = generate_mesh(shape_type, distribution_type, connection_type, R_sphere, H_cylinder, R_cylinder,
                         R_inner_tube, R_outer_tube, N_pts, k_nn, r_tube, res, sigma_blur, lap_iters, seed)

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