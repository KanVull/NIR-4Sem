import os
import cv2
import pathlib
import numpy as np
from sklearn.cluster import KMeans

level = 7
size = (600, 600)
sizeShow = (300, 300)

def GetPyrImg(image):
    G = image.copy()
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
    return img

def GetKMeans(image):
    img_r = (image / 255.0).reshape(-1,3)
    k_colors = KMeans(n_clusters=4).fit(img_r)
    img1 = k_colors.cluster_centers_[k_colors.labels_]
    img1 = np.reshape(img1, (image.shape))
    return (img1 * 255).astype(np.uint8)

path = 'Pics/'

for imageName in pathlib.Path(path).iterdir():
    if imageName.is_file():
        image = cv2.imread( str(imageName), 0)
        os.chdir('Pics/Mod/')
        PyrImage = GetPyrImg(image)
        KImage = GetKMeans(PyrImage)
        image = cv2.resize(image, sizeShow)
        k_image = cv2.resize(KImage, sizeShow)
        saveImage = np.concatenate([image, k_image], axis=1)
        cv2.imwrite(str(imageName).split('\\')[-1], saveImage)
        os.chdir('../../')