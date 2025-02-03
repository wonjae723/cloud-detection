import os
import xarray as xr

def load_file_list(folder1_path, folder2_path):
    folder_files = {
        'folder1': [os.path.join(folder1_path, f) for f in os.listdir(folder1_path) if f.endswith('.nc')],
        'folder2': [os.path.join(folder2_path, f) for f in os.listdir(folder2_path) if f.endswith('.nc')],
    }
    return folder_files

def load_nc_file(file_path):
    try:
        dataset = xr.open_dataset(file_path)
        return dataset
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None
