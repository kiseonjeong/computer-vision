import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from skimage import data

def get_image():
    # Get a test image
    img = data.astronaut()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    return gray.astype(dtype=np.float64)

def __generate_haar_matrix(length, level):
    # Generate haar matrix
    W = np.identity(length)
    for i in np.arange(level):
        Wi = np.zeros((length, length))
        size = int(length / 2**i)
        for j in np.arange(0, size, 2):
            Wi[j, int(j/2)] = 0.5
            Wi[j+1, int(j/2)] = 0.5
        for j in np.arange(0, size, 2):
            Wi[j, int(size/2)+int(j/2)] = 0.5
            Wi[j+1, int(size/2)+int(j/2)] = -0.5
        for j in np.arange(size, length):
            Wi[j, j] = 1
        W = np.dot(W, Wi)          # haar-matrix
        
    return W

def forward_dwt(img, level):
    # Set data size information
    sz = img.shape
    length = sz[1]
    
    # Generate a haar matrix
    haar_mat = __generate_haar_matrix(length, level)
    
    # Do forward transforms
    return np.dot(np.dot(img, haar_mat).T, haar_mat).T         # row filter -> column filter

def inverse_dwt(img, level):
    # Set data size information
    sz = img.shape
    length = sz[1]
    
    # Generate an inverse haar matrix
    haar_mat = __generate_haar_matrix(length, level)
    haar_mat_inv = np.linalg.inv(haar_mat)
    
    # Do inverse transforms
    return np.dot(np.dot(img, haar_mat_inv).T, haar_mat_inv).T         # row filter -> column filter

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