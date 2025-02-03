import numpy as np
import math

# Constants
C = 299792458.  # Speed of light (m/s)
H = 6.62607015e-34  # Planck's constant (m^2 kg / s)
K = 1.3806488e-23  # Boltzmann constant (m^2 kg / s^2 / K)

def load_calibration_coefficients(file_path, channel_index):
    try:
        coefficients = np.loadtxt(file_path, skiprows=9, unpack=True)[:, channel_index]
        return coefficients
    except Exception as e:
        print(f"Error loading calibration coefficients: {e}")
        return None

def apply_calibration(img_array, coefficients):
    cw, gain, offset, c1, c2, c3 = coefficients
    rad = gain * img_array + offset
    t_eff = (H * C / K * (cw * 100)) / np.log(2 * H * C ** 2 * (cw * 100) ** 3 / (rad * 1e-5) + 1)
    tbb = c1 + c2 * t_eff + c3 * t_eff ** 2
    return tbb
