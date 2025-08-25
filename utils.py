import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry


def random_crop(data: np.ndarray, num_patches: int, patch_size: Optional[int] = 128) -> np.ndarray:
    """Crop a random patch from [B, C, H, W] image.
    
    Args:
        data: 4D array of data with shape [B, C, H, W].
        num_patches: number of patches to crop from image.
        patch_size: size of patch, in pixels, to be cropped from image H & W.
            - default = 128
    
    Returns:
        patches: 4D array of cropped patch(es) with shape [B, C, patch_size, patch_size].
    """
    B, C, H, W = data.shape
    patches = []

    for _ in range(num_patches):
        i = torch.randint(0, H - patch_size + 1, (1,))      # -patchsize ensures the patch won't be outside image limits
        j = torch.randint(0, W - patch_size + 1, (1,))
        patches.append(data[:, :, i:i+patch_size, j:j+patch_size])

    patches = torch.cat(patches)
    return patches


def apply_n2v_mask(data: np.ndarray, mask_fraction: Optional[int] = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Randomly mask a fraction of pixels in data.

    Args:
        data: 4D array of data with shape [B, C, H, W].
        mask_fraction: fraction of pixels to be masked out in the patch.
            - default = 0.01

    Returns:
        A tuple[x_masked, mask] containing:
            data_masked: 4D masked array with shape [B, C, H, W].
            mask: masked array.
    """
    B, C, H, W  = data.shape
    data_masked = data.clone()
    mask        = torch.rand(B, 1, H, W) < mask_fraction

    data_masked[mask] = 0.0        # Replace masked pixels with 0

    return data_masked, mask


def do_aperture_photometry(
    data: np.ndarray,
    position: tuple[int, int],
    radius: int,
    sky_radius_in: int,
    sky_annulus_width: int
) -> tuple:
    """Performs aperture photometry and returns a tuple of values needed to calculate SNR/CNR.

    Args:
        data: 2D data of reduced science image.
        position: a tuple of position location with integers (x, y).
        radius: aperture radius in pixels.
        sky_radius_in: pixel radius at which to measure the sky background.
        sky_annulus_width: pixel width of the annulus.

    Returns:
        A tuple[SNR, CNR] containing:
            SNR: signal-to-noise ratio.
            CNR: contrast-to-noise ratio.
    """
    x, y = position[0], position[1]
    
    # Makes a circular aperture and caclulates the total flux in the area
    aperture = CircularAperture(position, radius)
    raw_flux = aperture_photometry(data, aperture)['aperture_sum'][0]

    # Makes a circular annulus and calculates the total background flux in that area
    annulus        = CircularAnnulus(position, sky_radius_in, sky_radius_in + sky_annulus_width)
    raw_background = aperture_photometry(data, annulus)['aperture_sum'][0]

    # Grabs the background's mean in the annulus and multiplies it by aperture's area to grab background in only annulus
    noise = (raw_background / annulus.area).item()
    noise_std  = data[x - 10: x + 10, y - 100: y - 50].flatten()
    noise_std  = np.std(noise_std)

    # Background count in the aperture
    background = noise * aperture.area

    # Calculates total flux
    signal = raw_flux - background

    # Read-out noise that was calculated prior to denoising
    RON = 17.26    

    cnr = (signal - noise) / noise_std
    snr = signal / np.sqrt(signal + aperture.area * RON ** 2)

    return snr, cnr 



def plot_comparisons(models: list):
    """Plots SNR and CNR for different models.

    Args:
        models: list of strings of models wished to be plot
            - example: ['n2v', 'standard']
    """
    # Create SNR figure
    fig_snr, ax_snr = plt.subplots()
    ax_snr.set_title("SNR")
    ax_snr.grid(True)

    # Create CNR figure
    fig_cnr, ax_cnr = plt.subplots()
    ax_cnr.set_title("CNR")
    ax_cnr.grid(True)

    for model in models:
        model_snr = np.load(f'Plotting/{model}_snr.npy')
        model_cnr = np.load(f'Plotting/{model}_cnr.npy')

        ax_snr.plot(model_snr, label=model)
        ax_cnr.plot(model_cnr, label=model)

    # Add legends and save after all models are plotted
    ax_snr.legend()
    fig_snr.savefig("Plotting/SNR.pdf", dpi=300)
    plt.close(fig_snr)

    ax_cnr.legend()
    fig_cnr.savefig("Plotting/CNR.pdf", dpi=300)
    plt.close(fig_cnr)
