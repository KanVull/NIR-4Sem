import os
import cv2
import numpy as np

level = 7
size = (600, 600)

image = cv2.imread('Pics/mdb095.bmp', 0)
G = image.copy()
image = cv2.resize(image, size)
cv2.imshow('img', image)

# generate Gaussian pyramid
gpImg = [G]
for i in range(level):
    G = cv2.pyrDown(G)
    gpImg.append(G)

# generate Laplacian Pyramid
lpImg = [gpImg[level - 1]]
for i in range(level - 1, 0, -1):
    GE = cv2.pyrUp(gpImg[i])
    L = cv2.subtract(gpImg[i - 1], GE)
    lpImg.append(L)

ls_ = lpImg[0]
for i in range(1,level):
    ls_ = cv2.pyrUp(ls_)
    ls_ = cv2.add(ls_, lpImg[i])

ls_ = cv2.resize(ls_, size)
cv2.imshow('imgL', ls_)

#k-avarage
#4 layers


cv2.waitKey(0)