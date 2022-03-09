import os
import matplotlib.pyplot as plt
import numpy as np

from celluloid import Camera


def create_movie(input_dir, target_dir, filename, fps, figsize=(9, 6)):
    os.makedirs(target_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    camera = Camera(fig)

    for imgs in os.listdir(input_dir):
        img = plt.imread(os.path.join(input_dir, imgs))
        ax.imshow(img, cmap="gray")
        plt.axis("off")
        camera.snap()
    animation = camera.animate(repeat=True)
    if fps:
        animation.save(f"{target_dir}/{filename}.mp4", writer="ffmpeg", fps=fps)
    else:
    	animation.save(f"{target_dir}/{filename}.mp4", writer="ffmpeg")

if __name__ == "__main__":
    INPUT_DIR = "training/saved_images"
    TARGET_DIR = "training/movies"
    FILENAME = "trained_v1"
    create_movie(INPUT_DIR, TARGET_DIR, FILENAME)
