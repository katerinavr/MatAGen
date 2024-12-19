import matplotlib.pyplot as plt
import json
import numpy as np
from scipy.interpolate import make_interp_spline

def load_data(filename):
    """Load data from a JSON file."""
    with open(filename, 'r') as file:
        return json.load(file)

def create_smooth_line(x, y, num_points=1000):
    """Create a smooth line through the data points."""
    x = np.array(x)
    y = np.array(y)
    
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]
    
    spl = make_interp_spline(x, y, k=3)
    x_smooth = np.linspace(x.min(), x.max(), num_points)
    y_smooth = spl(x_smooth)
    
    return x_smooth, y_smooth

def plot_spectra(data, figsize=(12, 7)):
    """
    Plot spectra for any number of samples with different numbers of data points.
    
    Parameters:
        data (dict): Dictionary containing the spectral data
        figsize (tuple): Figure size in inches
    """
    plt.figure(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(data)))
    
    for (sample_name, sample_data), color in zip(data.items(), colors):
        wavelength = sample_data['wavelength']
        try:
            intensity = sample_data['intensity']
        except:
            intensity = sample_data['absorbance']
        
        x_smooth, y_smooth = create_smooth_line(wavelength, intensity)
        
        plt.plot(x_smooth, y_smooth, '-', 
                label=sample_name, 
                color=color,
                linewidth=2)
        
        plt.plot(wavelength, intensity, 'o', 
                color=color,
                markersize=6,
                alpha=0.5)

    all_wavelengths = [val for sample in data.values() 
                      for val in sample['wavelength']]
    try: 
        all_intensities = [val for sample in data.values() 
                        for val in sample['intensity']]
    except:
        all_intensities = [val for sample in data.values() 
                        for val in sample['absorbance']]
    
    x_padding = (max(all_wavelengths) - min(all_wavelengths)) * 0.05
    y_padding = (max(all_intensities) - min(all_intensities)) * 0.05
    
    plt.xlim(min(all_wavelengths) - x_padding, 
            max(all_wavelengths) + x_padding)
    plt.ylim(max(0, min(all_intensities) - y_padding),
            max(all_intensities) + y_padding)

    plt.xlabel('Wavelength (nm)', fontsize=12)
    plt.ylabel('Intensity (a.u.)', fontsize=12)
    plt.title('Absorption Spectra', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    return plt.gcf()

def plot_smoothed_absorbance(data, points=200):
    """
    Plot smoothed absorbance spectra from input data.

    Args:
        data (dict): Dictionary containing compound names as keys and their
                     wavelength and absorbance data as subkeys.
        points (int): Number of points for smoothing.
    """
    plt.figure(figsize=(10, 6))

    for compound, values in data.items():
        wavelengths = np.array(values["wavelength"])
        absorbances = np.array(values["absorbance"])

        # Smooth data using cubic spline
        spline = make_interp_spline(wavelengths, absorbances, k=3)
        smooth_wavelengths = np.linspace(wavelengths.min(), wavelengths.max(), points)
        smooth_absorbances = spline(smooth_wavelengths)

        plt.plot(smooth_wavelengths, smooth_absorbances, label=f"{compound.capitalize()} (Smoothed)")

    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Absorbance')
    plt.title('Smoothed Absorbance Spectra')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    return plt


