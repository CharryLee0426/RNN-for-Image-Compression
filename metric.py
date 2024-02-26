import argparse
import numpy as np
from scipy.ndimage import convolve
from PIL import Image
from scipy import signal

"""
This python code is used to define the perceptual 
metric for the image quality assessment.

The ms-ssim implementation is devrived from the tensorflow's
implementation.
paper: https://www.cns.nyu.edu/pub/eero/wang03b.pdf

Usage:
python metric.py --original_image=<path_to_original_image> --compared_image=<path_to_compared_image>
"""

parser = argparse.ArgumentParser()

# add parser arguments
parser.add_argument("--original-image", "-o", type=str, required=True, help="original image file path")
parser.add_argument("--compared-image", "-c", type=str, required=True, help="compared image file path")
parser.add_argument("--metric", "-m", type=str, help="metric", default="all")

args = parser.parse_args()

def psnr(original, compared):
    """
    PSNR Function

    Args:
        original: original image
        compared: compared image    
    """

    if isinstance(original, str):
        original = np.array(Image.open(original).convert("RGB"), dtype=np.float32)
    if isinstance(compared, str):
        compared = np.array(Image.open(compared).convert("RGB"), dtype=np.float32)

    mse = np.mean(np.square(original - compared))
    psnr = np.clip(np.multiply(np.log10(255. * 255. / mse[mse > 0.]), 10.), 0., 99.99)[0]

    return psnr

def _FSpecialGauss(size, sigma):
    """
    2-D Special Gauss Function

    Args:
        size:  the number of elements in the Gaussian kernel;
        sigma: the standard deviation parameter of the Gaussian distribution;
    """
    radius = size // 2
    offset = 0.0
    start, stop = -radius, radius + 1

    if size % 2 == 0:
        offset = 0.5
        stop -= 1
    
    x, y = np.mgrid[offset + start:stop, offset + start:stop]

    assert len(x) == size

    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))

    return g / g.sum()

def _SSIMForMultipleScale(img1,
                          img2,
                          max_val=255,
                          filter_size=11,
                          filter_sigma=1.5,
                          k1=0.01,
                          k2=0.03):
    """
    SSIM for Multiple Scale SSIM

    Args:
        img1 & img2: image1 and image2
        max_val:  the dynamic range of the pixel values (255 for 8bits/pixel gray scale images)
        filter_size: the size of gauss kernel;
        filter_sigma: the standard deviation of gauss kernel;
        k1, k2: sclar vectors used, set as 0.01 and 0.03 based on the paper
    
    Error:
        input images don't have the same shape or not 4-d [batch_size, height, width, depth]
    
    Based on section 2: SINGLE-SCALE STRUCTURAL SIMILARITY
    """

    # some expections handling before starting calculating ssim
    if img1.shape != img2.shape:
        raise RuntimeError("Two input images must have the same shape, but got (%s, %s)", img1.shape, img2.shape)
    
    if img1.ndim != 4:
        raise RuntimeError("The input image must be 4-dimentional, but got %d", img1.ndim)
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    _, height, width, _ = img1.shape

    # handle filter size in order to fit input images' size
    # filter size can't be larger than input images' size
    size = min(filter_size, height, width)

    # scale down the sigma if the smaller filter size is used
    sigma = size * filter_sigma / filter_size if filter_size else 0 # avoid division by 0

    if filter_size:
        window = np.reshape(_FSpecialGauss(size, sigma), (1, size, size, 1))
        mu1 = signal.fftconvolve(img1, window, mode="valid")
        mu2 = signal.fftconvolve(img2, window, mode="valid")
        sigma11 = signal.fftconvolve(img1 * img1, window, mode="valid")
        sigma12 = signal.fftconvolve(img1 * img2, window, mode="valid")
        sigma22 = signal.fftconvolve(img2 * img2, window, mode="valid")
    else:
        # Empty gauss blur kernel, so no need to convolve
        mu1, mu2 = img1, img2
        sigma11 = img1 * img1
        sigma22 = img2 * img2
        sigma12 = img1 * img2

    mu11 = mu1 * mu1
    mu22 = mu2 * mu2
    mu12 = mu1 * mu2
    sigma11 -= mu11
    sigma22 -= mu22
    sigma12 -= mu12
    
    # Calculate intermediate values used by both ssim and contrast structure measure.
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    v1 = 2.0 * sigma12 + c2
    v2 = sigma11 + sigma22 + c2
    ssim = np.mean((((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2)))
    cs = np.mean(v1 / v2)
    return ssim, cs

def MultiScaleSSIM(img1,
                   img2,
                   max_val=255,
                   filter_size=11,
                   filter_sigma=1.5,
                   k1=0.01,
                   k2=0.03,
                   weights=None
):
    """
    MS-SSIM score between img1 and img2

    Args:
        img1 & img2: image1 and image2's array
        max_val: the dynamic range of the pixel values (255 for 8bits/pixel gray scale images)
        filter_size: the size of gauss kernel;
        filter_sigma: the standard deviation of gauss kernel;
        k1, k2: sclar vectors used, set as 0.01 and 0.03 based on the paper
        weights: list of weights for each level, if none, use five levels and the original paper's weights
    
    Error:
        input images don't have the same shape or not 4-d [batch_size, height, width, depth]
    """

    # Handling error first
    if img1.shape != img2.shape:
        raise RuntimeError("Two input images must have the same shape, but got (%s, %s)", img1.shape, img2.shape)
    
    if img1.ndim != 4:
        raise RuntimeError("The input image must be 4-dimentional, but got %d", img1.ndim)
    
    # default weights are shown in the end of section 3 of based paper
    weights = np.array(weights if weights else [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])

    levels = weights.size
    downsample_filter = np.ones((1, 2, 2, 1)) / 4.0
    im1, im2 = [x.astype(np.float64) for x in [img1, img2]]
    mssim = np.array([])
    mcs = np.array([])

    for _ in range(levels):
        ssim, cs = _SSIMForMultipleScale(
            im1,
            im2,
            max_val=max_val,
            filter_size=filter_size,
            filter_sigma=filter_sigma,
            k1=k1,
            k2=k2,
        )

        mssim = np.append(mssim, ssim)
        mcs = np.append(mcs, cs)

        filtered = [
            convolve(im, downsample_filter, mode='reflect')
            for im in [im1, im2]
        ]
        im1, im2 = [x[:, ::2, ::2, :] for x in filtered]


    return (np.prod(mcs[0:levels-1] ** weights[0:levels-1]) * (mssim[levels-1] ** weights[levels-1]))

def msssim(original, compared):
    """
    the mssim function that can be invoked by outer users
    
    Args:
        original: the original image file path;
        compared: the compared image file path;
    """
    if isinstance(original, str):
        original = np.array(Image.open(original).convert("RGB"), dtype=np.float32)
    if isinstance(compared, str):
        compared = np.array(Image.open(compared).convert("RGB"), dtype=np.float32)

    # convert original and compared to 4-d
    original = original[None, ...] if original.ndim == 3 else original
    compared = compared[None, ...] if compared.ndim == 3 else compared

    return MultiScaleSSIM(original, compared, max_val=255)

def main():
    if args.metric != "psnr":
        print(msssim(args.original_image, args.compared_image), end="")
    if args.metric != "ssim":
        print(psnr(args.original_image, args.compared_image), end="")

if __name__ == "__main__":
    main()