import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from skimage import data

def dct_coeff(height, width):
    # Generate coefficients for DCT
    coeff = np.zeros((height, width))
    PI = np.math.atan(1.0) * 4.0

    # Set coefficients
    for i in range(height):
        for j in range(width):
            if i == 0:
                coeff[i, j] = 1.0 / np.sqrt(width)
            else:
                coeff[i, j] = np.sqrt(2.0 / width) * np.cos(((2 * j + 1) * i * PI) / (2.0 * width))

    return coeff
    
def forward_dct(block, coeff, dct_len):
    pass

def inverse_dct(block, coeff, dct_len):
    pass

if __name__ == '__main__':
    # Get an image
    img = data.moon()

    # Calculate coeffcients for DCT
    dct_len = 8
    coeff = dct_coeff(dct_len, dct_len)

    # Do DCT processes
    inv = np.zeros(img.shape)
    for i in range(0, img.shape[0], dct_len):
        for j in range(0, img.shape[1], dct_len):
            # Get a patch
            patch = img[i:i+dct_len, j:j+dct_len]

            # Do forward process
            forward_dct(patch, coeff, dct_len)

            # Do inverse process
            inverse_dct(patch, coeff, dct_len)

            # Set a patch
            inv[i:i+dct_len, j:j+dct_len] = patch

    # Show results
    plt.figure('image')
    plt.imshow(img, cmap=plt.cm.gray)
    plt.figure('inverse')
    plt.imshow(inv, cmap=plt.cm.gray)
    plt.show()
