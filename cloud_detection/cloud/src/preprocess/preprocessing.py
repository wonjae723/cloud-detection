import numpy as np
import xarray as xr
import os

def calculate_difference(array1, array2):
    diff_array = array1 - array2
    return diff_array

def save_as_netcdf(output_folder, file_suffix, array):
    try:
        output_file_path = os.path.join(output_folder, f"{file_suffix}.nc")
        ds = xr.Dataset(
            {
                "image_pixel_value": (["x", "y"], array)
            },
            coords={
                "x": np.arange(array.shape[0]),
                "y": np.arange(array.shape[1])
            }
        )
        ds.to_netcdf(output_file_path)
        print(f"Saved file: {output_file_path}")
    except Exception as e:
        print(f"Error saving NetCDF file: {e}")
