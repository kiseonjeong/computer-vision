import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from skimage import data

def get_image():
    # Get a test image
    img = data.astronaut()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    return gray.astype(dtype=np.float64)

def __forward_haar_wavelet_filter1D(img):
    # Do forward filtering haar wavelet
    height, width = img.shape
    half_width = int(width / 2)
    filt = np.zeros((height, width))
    even, odd = img[:, 0::2], img[:, 1::2]
    L, H = (even + odd) / 2, (even - odd) / 2
    filt[:, 0:half_width], filt[:, half_width:width] = L, H

    return filt

def __forward_haar_wavelet_filter2D(img):
    # Do forward filtering on column axis
    height, width = img.shape
    half_width = int(width / 2)
    L_H = __forward_haar_wavelet_filter1D(img)
    L, H = L_H[:, 0:half_width], L_H[:, half_width:width]

    # Do forward filtering on row axis
    LL_LH = __forward_haar_wavelet_filter1D(L.T).T
    HL_HH = __forward_haar_wavelet_filter1D(H.T).T
    DWT = np.concatenate((LL_LH, HL_HH), axis=1)

    return DWT

def __forward_dwt(img, index = 0, level = 1):
    # Check an index
    if index < level:
        # Iteratively, do forward transforms
        height, width = img.shape
        half_height, half_width = int(height / 2), int(width / 2)
        curr_dwt = __forward_haar_wavelet_filter2D(img)
        next_dwt = __forward_dwt(curr_dwt[0:half_height, 0:half_width], index + 1, level)
        curr_dwt[0:half_height, 0:half_width] = next_dwt

        return curr_dwt
    else:
        return img

def forward_dwt(img, level):
    # Do forward transforms
    return __forward_dwt(img, 0, level)

def __inverse_haar_wavelet_filter1D(img):
    # Do inverse filtering haar wavelet
    height, width = img.shape
    half_width = int(width / 2)
    filt = np.zeros((height, width))
    L, H = img[:, 0:half_width], img[:, half_width:width]
    even, odd = L + H, L - H
    filt[:, 0::2], filt[:, 1::2] = even, odd

    return filt

def __inverse_haar_wavelet_filter2D(img):
    # Do inverse filtering on row axis
    height, width = img.shape
    half_width = int(width / 2)
    L, H = img[:, 0:half_width], img[:, half_width:width]
    iL = __inverse_haar_wavelet_filter1D(L.T).T
    iH = __inverse_haar_wavelet_filter1D(H.T).T
    iL_iH = np.concatenate((iL, iH), axis=1)

    # Do inverse filtering on column axis
    iDWT = __inverse_haar_wavelet_filter1D(iL_iH)

    return iDWT

def __inverse_dwt(img, index = 0, level = 1):
    # Check an index
    if index < level:
        # Iteratively, do inverse transforms
        height, width = img.shape
        half_height, half_width = int(height / 2), int(width / 2)
        curr_dwt = img.copy()
        next_dwt = curr_dwt[0:half_height, 0:half_width]
        next_idwt = __inverse_dwt(next_dwt, index + 1, level)
        curr_dwt[0:half_height, 0:half_width] = next_idwt
        curr_idwt = __inverse_haar_wavelet_filter2D(curr_dwt)

        return curr_idwt
    else:
        return img

def inverse_dwt(img, level):
    # Do inverse transforms
    return __inverse_dwt(img, 0, level)

def main():
    # Get a test image
    img = get_image()

    # Do forward transform
    dwt = forward_dwt(img, 2)

    # Do inverse transform
    idwt = inverse_dwt(dwt, 2)

    # Compare the reuslts
    compare = img == idwt
    equal_arr = compare.all()
    print('compare result :', equal_arr)

    # Show results
    plt.figure('img')
    plt.imshow(img, cmap=plt.cm.gray)
    plt.figure('dwt')
    plt.imshow(dwt, cmap=plt.cm.gray)
    plt.figure('idwt')
    plt.imshow(idwt, cmap=plt.cm.gray)
    plt.show()

if __name__ == '__main__':
    # Do main process
    main()