import os
from patchify import patchify
import numpy as np

import matplotlib.pyplot as plt
import cv2
######################################
path = "images_cont"
os.makedirs("rough", exist_ok=True)

images_path = os.listdir(path)

for idx, img in enumerate(images_path):
    print(idx)
    images = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)

    patches = patchify(images, (64, 64), step=64)
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            img = patches[i, j, :, :]
            cv2.imwrite(f"rough/patches_{idx}_{i}{j}.png", img)


