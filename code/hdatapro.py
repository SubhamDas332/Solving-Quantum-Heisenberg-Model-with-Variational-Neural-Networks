import numpy as np
import glob

# Load all npz files in a specified folder
file_paths = glob.glob("heisenberg data plot/*.npz")
all_data = [np.load(file) for file in file_paths]

# Helper function to find a non-NaN replacement value
def get_non_nan_value(arrays, index, key):
    for data in arrays:
        value = data[key][index]
        if not np.isnan(value):
            return value
    return np.nan  # If all files have NaN, fallback to NaN

# List of keys to process
keys = ["Transverse", "energy_per_site", "magnetization_z", "magnetization_x",
        "susceptibility_z", "susceptibility_x", "exact_gs_energiespersite", "seed"]

# Iterate over each file and replace NaNs
for i, data in enumerate(all_data):
    modified_data = {}
    for key in keys:
        modified_data[key] = data[key].copy()
        # Replace NaNs with values from other files
        for idx, value in enumerate(data[key]):
            if np.isnan(value):
                replacement = get_non_nan_value(all_data[:i] + all_data[i+1:], idx, key)
                modified_data[key][idx] = replacement

    # Save modified data back to a new npz file or overwrite
    np.savez(f"Heisenberg data modified/modified_run_{i}.npz", **modified_data)

print("NaN replacement complete.")
