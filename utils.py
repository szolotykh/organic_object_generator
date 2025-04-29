import numpy as np

def save_parameters(output_path, **params):
    """Save the parameters to a .md file."""
    param_file = output_path.replace('.stl', '.md')
    with open(param_file, 'w') as f:
        f.write("# Parameters\n\n")
        for key, value in params.items():
            f.write(f"- **{key}**: {value}\n")
    print(f"Parameters saved → {param_file}")
