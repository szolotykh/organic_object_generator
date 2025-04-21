import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from skimage import measure
import trimesh
from trimesh.smoothing import filter_laplacian

# ───── PARAMETERS ─────
R_sphere   = 1.0    # for sampling points; bounding box is [-R_sphere,R_sphere]^3
N_pts      = 120     # number of random points
k_nn       = 6      # connect each point to its k nearest neighbors
r_tube     = 0.04   # radius of each tube
res        = 100    # grid resolution (res³ voxels)
sigma_blur = 0.5    # smooth the implicit field
lap_iters  = 15     # mesh Laplacian smoothing
connection_type = "nearest"  # Options: "nearest" or "random"
distribution_type = "surface"  # Options: "inside" or "surface"

# ───── 1) SAMPLE POINTS inside the sphere (rejection sampling) ─────
points = []
if distribution_type == "inside":
    while len(points) < N_pts:
        p = np.random.uniform(-R_sphere, R_sphere, size=3)
        if p.dot(p) <= R_sphere**2:
            points.append(p)

elif distribution_type == "surface":
    while len(points) < N_pts:
        p = np.random.normal(size=3)
        p /= np.linalg.norm(p)  # Normalize to unit vector
        p *= R_sphere          # Scale to sphere surface
        points.append(p)

points = np.array(points)

# ───── 2) FIND CONNECTION PAIRS ─────
if connection_type == "nearest":
    tree = cKDTree(points)
    _, idxs = tree.query(points, k=k_nn+1)
    pairs = set()
    for i in range(N_pts):
        for j in idxs[i,1:]:
            pairs.add(tuple(sorted((i,j))))
    pairs = list(pairs)

elif connection_type == "random":
    pairs = set()
    while len(pairs) < N_pts * k_nn:
        i, j = np.random.choice(N_pts, size=2, replace=False)
        pairs.add(tuple(sorted((i, j))))
    pairs = list(pairs)

# ───── 3) BUILD 3D GRID ─────
margin = R_sphere * 0.1
lin = np.linspace(-R_sphere-margin, R_sphere+margin, res)
X, Y, Z = np.meshgrid(lin, lin, lin, indexing='ij')

# ───── 4) INITIALIZE FIELD for union of capsules ─────
# We want F(x)<0 inside any tube, >0 outside all tubes.
F_union = np.full((res,res,res), np.inf, dtype=np.float32)

# ───── 5) CARVE OUT EACH CAPSULE SDF ─────
for (i,j) in pairs:
    A = points[i]
    B = points[j]
    AB = B - A
    AB_len2 = np.dot(AB,AB)

    # Vector from A to each voxel
    AP = np.stack((X - A[0], Y - A[1], Z - A[2]), axis=-1)  # shape (res,res,res,3)
    t  = np.clip((AP @ AB) / AB_len2, 0.0, 1.0)              # projection factor
    proj = A + t[...,None] * AB                             # closest points on AB
    diff = np.stack((X,Y,Z),axis=-1) - proj
    dist = np.linalg.norm(diff, axis=-1)
    sdf_capsule = dist - r_tube

    # union = pointwise min over all capsules
    F_union = np.minimum(F_union, sdf_capsule)

# ───── 6) OPTIONAL: GAUSSIAN BLUR ─────
F_smooth = gaussian_filter(F_union, sigma=sigma_blur)

# ───── 7) EXTRACT MESH with MARCHING CUBES ─────
spacing = (lin[1]-lin[0],)*3
verts, faces, normals, _ = measure.marching_cubes(
    F_smooth, level=0.0, spacing=spacing)

# ───── 8) MESH SMOOTHING ─────
mesh = trimesh.Trimesh(vertices=verts, faces=faces,
                       vertex_normals=normals)
filter_laplacian(mesh, iterations=lap_iters)

# ───── 9) EXPORT ─────
mesh.export('point_web.stl')
print("Exported → point_web.stl")