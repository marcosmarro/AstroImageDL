import os
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from pathlib import Path
from utils import do_aperture_photometry
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from photutils.centroids import centroid_sources, centroid_quadratic
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)    # Astropy throws fixing errors that can be ignored

parser = argparse.ArgumentParser(description='SNR and CNR evaluation on denoised images')
parser.add_argument('--data_directory', '-d', default=os.getcwd(), help='Directory where denoised files live.')
parser.add_argument('--denoising_model', '-m', default='n2v', help='Denoising model to use [n2v/standard]. Defualt: n2v')
args = parser.parse_args()

directory = Path(args.data_directory)
file_list = sorted(directory.glob(f'{args.denoising_model}*.fits'))

cnr_values = []
snr_values = []

for file in file_list:
    print(file.stem)

    science      = fits.open(file)
    science_data = science[0].data
    science_wcs  = WCS(science[0].header)
    if 'CTYPE1' not in science[0].header:   # Some files don't have celestial coordinates due to technical errors
        continue

    target_ra    = science[0].header['RA']
    target_dec   = science[0].header['DEC']
    target_coord = SkyCoord(ra=target_ra, dec=target_dec, frame='icrs', unit=(u.hourangle, u.deg))
     
    x, y = target_coord.to_pixel(science_wcs)
    position = centroid_sources(science_data - np.median(science_data), xpos=x, ypos=y, box_size=35, centroid_func=centroid_quadratic)
    position = (int(position[0][0]), int(position[1][0]))

    # Photometry parameters inferred by first looking at data
    photometry_radius = 12
    annulus_radius    = 18
    annulus_width     = 4

    snr, cnr = do_aperture_photometry(science_data, position, photometry_radius, annulus_radius, annulus_width)
    
    snr_values.append(snr)
    cnr_values.append(cnr)

np.save(f'Plotting/{args.denoising_model}_cnr.npy', np.array(cnr_values))
np.save(f'Plotting/{args.denoising_model}_snr.npy', np.array(snr_values))