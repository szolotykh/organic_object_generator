import argparse
from main import main  # Assuming the main logic is encapsulated in a `main` function in main.py

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate organic 3D objects.")
    parser.add_argument("--shape_type", type=str, default="sphere", choices=["sphere", "cylinder", "tube"],
                        help="Type of shape to generate.")
    parser.add_argument("--distribution_type", type=str, default="inside", choices=["inside", "surface"],
                        help="Distribution type for points.")
    parser.add_argument("--connection_type", type=str, default="nearest", choices=["nearest", "random"],
                        help="Connection type for points.")
    parser.add_argument("--R_sphere", type=float, default=1.0, help="Radius of the sphere.")
    parser.add_argument("--H_cylinder", type=float, default=3.0, help="Height of the cylinder.")
    parser.add_argument("--R_cylinder", type=float, default=1.0, help="Radius of the cylinder.")
    parser.add_argument("--R_inner_tube", type=float, default=0.2, help="Inner radius of the tube.")
    parser.add_argument("--R_outer_tube", type=float, default=1.0, help="Outer radius of the tube.")
    parser.add_argument("--N_pts", type=int, default=180, help="Number of random points.")
    parser.add_argument("--k_nn", type=int, default=4, help="Number of nearest neighbors for connections.")
    parser.add_argument("--r_tube", type=float, default=0.07, help="Radius of each tube.")
    parser.add_argument("--res", type=int, default=100, help="Grid resolution.")
    parser.add_argument("--sigma_blur", type=float, default=0.4, help="Gaussian blur sigma.")
    parser.add_argument("--lap_iters", type=int, default=15, help="Number of Laplacian smoothing iterations.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random operations.")
    parser.add_argument("--output", type=str, default="output/point_web.stl", help="Output file path.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(
        shape_type=args.shape_type,
        distribution_type=args.distribution_type,
        connection_type=args.connection_type,
        R_sphere=args.R_sphere,
        H_cylinder=args.H_cylinder,
        R_cylinder=args.R_cylinder,
        R_inner_tube=args.R_inner_tube,
        R_outer_tube=args.R_outer_tube,
        N_pts=args.N_pts,
        k_nn=args.k_nn,
        r_tube=args.r_tube,
        res=args.res,
        sigma_blur=args.sigma_blur,
        lap_iters=args.lap_iters,
        seed=args.seed,
        output=args.output
    )
