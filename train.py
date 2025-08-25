import os
import gc
import torch
import argparse
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.io import fits
from network import UNetN2V, UNetN2N
from utils import random_crop, apply_n2v_mask

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='ML training script on astronomical images.')
parser.add_argument('--denoising_model', '-m', default='n2v', help='Denoising model to use [n2v]. Defualt: n2v')
parser.add_argument('--data_directory', '-d', default=os.getcwd(), help='Directory where training files live.')
args = parser.parse_args()

batchsize = 2
num_steps = 50 // batchsize

# Selecting model
if args.denoising_model == 'n2v':   # Noise2Void
    model      = UNetN2V(in_ch=1, depth=3).to(DEVICE)
    criterion  = nn.SmoothL1Loss(reduction='none')
    model_name = 'N2V.pth'
    optimizer  = torch.optim.Adam(model.parameters(), lr=0.0004)

if args.denoising_model == 'n2n':   # Noise2Noise
    model      = UNetN2N(in_ch=1, depth=4).to(DEVICE)
    criterion  = nn.SmoothL1Loss(reduction='none')
    model_name = 'N2N.pth'
    optimizer  = torch.optim.Adam(model.parameters(), lr=0.001)


# Grabbing
directory = Path(args.data_directory)
file_list = sorted(directory.glob('*.fits'))


for i, file in enumerate(file_list):
    science_data = fits.getdata(file)[100:-100, 100:-100]

    train_loader = torch.from_numpy(science_data).float().to(DEVICE)
    train_loader = train_loader.unsqueeze(0).unsqueeze(0)

    for ii in range(num_steps):
        input_sequence     = random_crop(train_loader, batchsize)   # crops random 128x128 from image  
        input_masked, mask = apply_n2v_mask(input_sequence)         # masks ~1% of pixels to be trained on (only N2V)

        output_seq = model(input_masked)

        loss_map = criterion(output_seq, input_sequence)
        loss     = loss_map[mask].mean()                           # only care about loss of masked values

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if ii % 5 == 0:
            print('Epoch: {} | Batch: {} | Loss: {}'.format(i, ii, loss))
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(input_sequence[0, 0].detach().cpu().numpy(), cmap="gray")
            axs[0].set_title("Noisy Input")
            axs[0].axis("off")

            axs[1].imshow(output_seq[0, 0].detach().cpu().numpy(), cmap="gray")
            axs[1].set_title("Denoised Output")
            axs[1].axis("off")

            plt.savefig('denoise_sample.pdf', dpi=300)

    del file, train_loader
    gc.collect()

torch.save(model, model_name)

del model, criterion, optimizer
torch.cuda.empty_cache()