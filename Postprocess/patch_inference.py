import os
import numpy as np
import cv2
import torch

import matplotlib.pyplot as plt
import pandas as pd

from celluloid import Camera

from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.preprocessing import normalize
from network import BIMSNet, BIMSNet_Attn
from dataset import BIMSDataset
from metrics import *


def get_model(model_type="basic"):
	if model_type == "basic":
		return BIMSNet()
	elif model_type == "attention":
		return BIMSNet_Attn()


def prediction(test_dir="data/test/", stop=None, verbose=None):
	test_img_dir = test_dir + "/images"
	test_mask_dir = test_dir + "/masks"

	test_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize([0.6414], [0.0655])
	])

	test_ds = BIMSDataset(image_dir=test_img_dir, mask_dir=test_mask_dir, transform=test_transform)
	test_dataloader = DataLoader(test_ds, batch_size=1, num_workers=8, pin_memory=True, shuffle=False)

	ckpt_path = "training/saved_models/bimsnet.pth"
	device = torch.device("cuda")
	model = get_model()
	model.to(device)
	model.eval()
	state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
	model.load_state_dict(state["model_state"])

	os.makedirs("prediction/saved_images", exist_ok=True)
	os.makedirs("prediction/movies", exist_ok=True)

	fig, ax = plt.subplots(1, 3, figsize=(8, 6))
	camera = Camera(fig)

	for i, batch in enumerate(test_dataloader):
		image, target_mask = batch
		segmented_batch = torch.sigmoid(model(image.to(device)))
		segmented_batch = segmented_batch.detach().cpu().numpy()
		segmented_image = (segmented_batch > 0.5).astype("float32")

		ax[0].imshow(np.squeeze(image), cmap="gray")
		ax[1].imshow(np.squeeze(target_mask), cmap="gray")
		ax[2].imshow(np.squeeze(segmented_image), cmap="gray")
		camera.snap()

		ax[0].set_title("Original")
		ax[1].set_title("Groundtruth")
		ax[2].set_title("Predicted")
		ax[0].axis("off")
		ax[1].axis("off")
		ax[2].axis("off")

		if verbose:
			fig.savefig(f"prediction/saved_images/segmented_{i}.png", dpi=300)

		if stop and i == stop:
			break
		else:
			continue
	animation = camera.animate(interval=200, repeat=True, repeat_delay=200)
	animation.save("prediction/movies/prediction_v1.mp4", writer="ffmpeg", fps=2)


def main():
	prediction(stop=200, verbose=True)


if __name__ == '__main__':
	main()
