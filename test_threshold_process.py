from net import skip,skip_mask
from net.losses import ExclusionLoss, plot_image_grid, StdLoss, GradientLoss,MS_SSIM,tv_loss
from net.noise import get_noise
from utils.image_io import *
from utils.segamentation import k_means
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple
import scipy.signal as signal
from net.downsampler import Downsampler

import cv2
matplotlib.use('TkAgg')

def cmp_PSF(A,B,psf_r=5,k1=3,k2=3,k3=3,threshold=0.018,if_largest_region=1):
    fA = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(A)))

    fB = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(B)))
    fC = fA / fB
    #fC = cv2.GaussianBlur(fC, ksize=(3, 3,), sigmaX=0, sigmaY=0)
    C = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(fC)))
    PSF = abs(C[C.shape[0] // 2 -psf_r+1:C.shape[0] // 2 + psf_r, C.shape[1] // 2 - psf_r+1:C.shape[1] // 2 + psf_r])
    sC = cv2.filter2D(A, -1, PSF)
    plt.imshow(sC, 'gray')
    plt.show()
    F = abs(A - sC)
    D = clear_segmetation(F,k1,k2,k3,threshold)
    if if_largest_region==1:
        m = D
        q = generate_largest_region(m)
        D = fillHole(q)
    plt.imshow(PSF, 'gray')
    plt.show()
    """
    plt.subplot(1, 3, 1)
    plt.imshow(PSF, 'gray')
    plt.subplot(1, 3, 2)
    plt.imshow(F, 'gray')
    plt.subplot(1, 3, 3)
    plt.imshow(D, 'gray')
    plt.show()
    """
    return D




if __name__ == "__main__":
    # Separation from two images
    import os

    i = 0
    dict = [[3, 1, 1, 0.008, 0],
            [5, 3, 3, 0.010, 0],
            [3, 1, 1, 0.018, 0],
            [5, 1, 1, 0.010, 0],
            [5, 3, 3, 0.010, 1],
            [5, 3, 3, 0.010, 1],
            [5, 3, 3, 0.010, 1],
            [5, 3, 3, 0.010, 1],
            [3, 3, 3, 0.010, 1],
            [5, 3, 3, 0.010, 1],
            [3, 1, 1, 0.010, 0],
            [5, 3, 3, 0.010, 0],
            [5, 3, 3, 0.018, 0],
            [3, 3, 3, 0.010, 0],
            [5, 3, 3, 0.010, 1],
            [5, 3, 3, 0.010, 0],
            [5, 3, 3, 0.008, 1],
            [3, 3, 3, 0.018, 1],
            [5, 3, 3, 0.018, 0],
            [5, 3, 3, 0.010, 0],
            [3, 3, 3, 0.018, 1],
            [3, 3, 3, 0.010, 0],
            [5, 3, 3, 0.010, 0],
            [5, 3, 3, 0.010, 0],
            [5, 3, 3, 0.018, 1],
            [5, 3, 3, 0.010, 0],
            [5, 3, 3, 0.018, 0],
            [7, 3, 3, 0.010, 1],
            [5, 3, 3, 0.010, 1],
            [9, 3, 3, 0.010, 1]]
    for dirs in os.listdir('./output'):
        outpath = dirs
        input1 = prepare_image('./output/' + outpath + '/' + outpath + '_A.jpg')
        input2 = prepare_image('./output/' + outpath + '/' + outpath + '_B.jpg')

        f = 1


        input1_pil = np_to_pil(input1)
        input1_down = input1_pil.resize((input1_pil.size[0] // f, input1_pil.size[1] // f), Image.BICUBIC)
        image1=pil_to_np(input1_down)
        input2_pil = np_to_pil(input2)
        input2_down = input2_pil.resize((input2_pil.size[0] // f, input2_pil.size[1] // f), Image.BICUBIC)
        image2 = pil_to_np(input2_down)

        print(dict[i][0], dict[i][1],
                dict[i][2], dict[i][3], dict[i][4])
        cmp_PSF(rgb2y_CWH_nol(image1), rgb2y_CWH_nol(image2),5, int(dict[i][0]), int(dict[i][1]),
                int(dict[i][2]), dict[i][3], dict[i][4])
        i+=1

"""



    histogram = np.histogram(output,50)

    mask=clear_segmetation(pil_to_np(input1_down)[0,:,:],threshold=histogram[1][1])
    contour_index = 1
    from scipy.io import loadmat


    m = mask
    q = generate_largest_region(m)
    im_out = fillHole(q)
"""