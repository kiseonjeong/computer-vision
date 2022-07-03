import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from skimage import data

def get_image():
    # Get a test image
    img = data.astronaut()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    return gray.astype(dtype=np.float64)

def error_diffusion(img):
    # Copy the input image
    temp = img.copy()
    
    # Calculate error terms
    ipart = np.floor(temp)
    ppart = temp - ipart
    
    # Do padding the error terms
    sz = temp.shape
    pad = np.zeros((sz[0] + 2, sz[1] + 2))
    pad[1:sz[0]+1, 1:sz[1]+1] = ppart           # data region
    pad[0, 1:sz[1]+1] = ppart[sz[0]-1, :]         # top region
    pad[sz[0]+1, 1:sz[1]+1] = ppart[0, :]         # bottom region
    pad[:, 0] = pad[:, sz[1]]           # left region
    pad[:, sz[1]+1] = pad[:, 1]         # right region
    
    # Apply the error diffusion
    for i in np.arange(0, sz[0] + 1):
        if i % 2 == 1:           # odd lines
            for j in np.arange(0, sz[1] + 1):
                round_val = np.round(pad[i, j])
                delta_val = pad[i, j] - round_val
                pad[i+0, j+0] = round_val
                pad[i+0, j+1] = pad[i+0, j+1] + delta_val * (7 / 16)
                pad[i+1, j+1] = pad[i+1, j+1] + delta_val * (1 / 16)
                pad[i+1, j+0] = pad[i+1, j+0] + delta_val * (5 / 16)
                pad[i+1, j-1] = pad[i+1, j-1] + delta_val * (3 / 16)                
        else:           # even lines
            for j in np.arange(sz[1], 0, -1):
                round_val = np.round(pad[i, j])
                delta_val = pad[i, j] - round_val
                pad[i+0, j+0] = round_val
                pad[i+0, j-1] = pad[i+0, j-1] + delta_val * (7 / 16)
                pad[i+1, j+1] = pad[i+1, j+1] + delta_val * (3 / 16)
                pad[i+1, j+0] = pad[i+1, j+0] + delta_val * (5 / 16)
                pad[i+1, j-1] = pad[i+1, j-1] + delta_val * (1 / 16)
                
    # Merge and crop the diffusion result
    return ipart + pad[1:sz[0]+1, 1:sz[1]+1]

def main():
    # Get a test image
    img = get_image()
    
    # Do image smoothing
    loss_nbit = 4            # 4-bit lossy
    lossy = img / 2**loss_nbit
    
    # Do error diffusion on input image
    dither = error_diffusion(lossy) * 2**loss_nbit
    
    # Do quantizations
    img = img.astype('uint32')
    lossy = lossy.astype('uint32')
    dither = dither.astype('uint32')

    # Show results
    plt.figure('img')
    plt.imshow(img, cmap=plt.cm.gray)
    plt.figure('lossy')
    plt.imshow(lossy, cmap=plt.cm.gray)
    plt.figure('dither')
    plt.imshow(dither, cmap=plt.cm.gray)
    plt.show()

if __name__ == '__main__':
    # Do main process
    main()