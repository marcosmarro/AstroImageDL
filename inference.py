import os
import gc
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)    # Astropy throws fixing errors that can be ignored

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='ML training script on astronomical images.')
parser.add_argument('--denoising_model', '-m', default='n2v', help='Denoising model to use [n2v]. Defualt: n2v')
parser.add_argument('--data_directory', '-d', default=os.getcwd(), help='Directory where training files live.')
args = parser.parse_args()

batchsize = 1


if args.denoising_model == 'n2v':
    model = torch.load('N2V.pth', weights_only=False)
    model.eval()
    

directory = Path(args.data_directory)
file_list = sorted(directory.glob('*.fits'))


for i, file in tqdm(enumerate(file_list)):

    # Reading data from files, rememebring to discard outer pixels due to overscan regions
    science      = fits.open(file)
    science_data = science[0].data[100:-100, 100:-100]

    train_loader = torch.from_numpy(science_data).float().to(DEVICE)
    train_loader = train_loader.unsqueeze(0).unsqueeze(0)

    with torch.inference_mode():
        input_sequence = train_loader
   
        output_seq = model(input_sequence)
    
    # Scaling back to 16-bit values
    denoised = output_seq.squeeze().cpu().numpy()

    # Ensuring all the information from the science header is written to the denoised file to preserve 
    # all information including RA/DEC
    original_header = science[0].header
    original_wcs    = WCS(original_header)
    cropped_wcs     = original_wcs.slice((slice(100, -100), slice(100, -100)))
    cropped_header  = cropped_wcs.to_header()

    for key in original_header:
      # breakpoint()
      if key in ('HISTORY', 'COMMENT'):
         continue
      if key not in cropped_header:
        cropped_header[key] = original_header[key]
   
    # Writing the data into the file where it's saved in directory named DenoisedScience
    science_hdu = fits.PrimaryHDU(data=denoised, header=cropped_header)
    hdu_list    = fits.HDUList([science_hdu])
    hdu_list.writeto(f'DenoisedScience/{args.denoising_model}_{file.stem}.fits', overwrite=True)

    print(f'Denoised {file}')
    del file, train_loader
    gc.collect()
    

del model
torch.cuda.empty_cache()
       
        