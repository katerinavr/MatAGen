import json
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path

# Load the JSON data
with open('html_scraping/retrieved_image_caption_pairs.json', 'r') as file:
    data = json.load(file)

# Create output directory if it doesn't exist
output_dir = "absorption_spectra_plots"
os.makedirs(output_dir, exist_ok=True)

# Find all records classified as absorption spectra
absorption_records = [record for record in data['records'] if record.get('classification') == 'absorption spectra']

# Function to extract base filename
def get_base_filename(path):
    return Path(path).name

# Plot each absorption spectrum and save it
for record in absorption_records:
    if 'metadata' not in record:
        continue
    
    image_path = record.get('image', '')
    base_filename = get_base_filename(image_path)
    
    # Create a new figure for each record
    plt.figure(figsize=(12, 8))
    
    # Plot each spectrum in the metadata
    for label, data_dict in record['metadata'].items():
        if 'wavelength' in data_dict and 'absorbance' in data_dict:
            plt.plot(data_dict['wavelength'], data_dict['absorbance'], label=label)
    
    # Add plot details
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Absorbance (a.u.)')
    plt.title(f"Absorption Spectra - {record.get('caption', 'Unknown')}")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save the figure
    output_filename = os.path.join(output_dir, f"{base_filename.split('.')[0]}_absorption_spectra.png")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_filename}")

print(f"\nAll absorption spectra have been plotted and saved to the '{output_dir}' directory.")