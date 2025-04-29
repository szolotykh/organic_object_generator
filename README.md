# Organic Objects Generator

This project generates organic 3D objects based on user-defined parameters. The generated objects can be exported as STL files for further use in 3D modeling or printing.

## Features

- Generate shapes: Sphere, Cylinder, Tube
- Customize point distribution: Inside or Surface
- Define connection types: Nearest or Random
- Adjustable parameters for shape dimensions, resolution, and smoothing
- Export results to STL format

## Requirements

- Python 3.7 or higher
- Install required Python packages using the `requirements.txt` file:
  ```bash
  pip install -r requirements.txt
  ```

## Usage

Run the script with the desired parameters:

```bash
python generator.py --shape_type sphere --distribution_type inside --connection_type nearest --output output/point_web.stl
```

### Available Arguments

| Argument             | Default Value       | Description                                   |
|----------------------|---------------------|-----------------------------------------------|
| `--shape_type`       | `sphere`           | Type of shape to generate (`sphere`, `cylinder`, `tube`) |
| `--distribution_type`| `inside`           | Distribution type for points (`inside`, `surface`) |
| `--connection_type`  | `nearest`          | Connection type for points (`nearest`, `random`) |
| `--R_sphere`         | `1.0`              | Radius of the sphere                         |
| `--H_cylinder`       | `3.0`              | Height of the cylinder                       |
| `--R_cylinder`       | `1.0`              | Radius of the cylinder                       |
| `--R_inner_tube`     | `0.2`              | Inner radius of the tube                     |
| `--R_outer_tube`     | `1.0`              | Outer radius of the tube                     |
| `--N_pts`            | `180`              | Number of random points                      |
| `--k_nn`             | `4`                | Number of nearest neighbors for connections  |
| `--r_tube`           | `0.07`             | Radius of each tube                          |
| `--res`              | `100`              | Grid resolution                              |
| `--sigma_blur`       | `0.4`              | Gaussian blur sigma                          |
| `--lap_iters`        | `15`               | Number of Laplacian smoothing iterations     |
| `--seed`             | `42`               | Seed for random operations                   |
| `--output`           | `output/point_web.stl` | Output file path                          |

## Example

To generate a tube with surface point distribution and nearest neighbor connections:

```bash
python generator.py --shape_type tube --distribution_type surface --connection_type nearest --R_outer_tube 1.5 --N_pts 200 --output output/tube_model.stl
```

## Output

The generated STL file will be saved to the specified output path. Additionally, a `.md` file containing the parameters used for generation will be created alongside the STL file.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
