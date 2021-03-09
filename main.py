import os
import cv2
import numpy as np
from sklearn.cluster import KMeans

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

img = cv2.resize(ls_, size)
# cv2.imshow('imgL', img)

#k-avarage
#4 layers
img_r = (img / 255.0).reshape(-1,3)
k_colors = KMeans(n_clusters=4).fit(img_r)
img1 = k_colors.cluster_centers_[k_colors.labels_]
img1 = np.reshape(img1, (img.shape))

cv2.imshow('k', img1)

cv2.waitKey(0)