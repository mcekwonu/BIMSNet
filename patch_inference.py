import os
import torch
import csv
import math
import numpy as np
import pandas as pd

from collections import defaultdict
from network import BIMSNet
from metrics import *
from utils import *

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
batch_size = 1
sample_interval = 10
test_dir = "data"
TEST_IMG_DIR = test_dir + "/images"
TEST_MASK_DIR = test_dir + "/masks"

os.makedirs("predictions/segmented_stacks", exist_ok=True)
os.makedirs("predictions/segmented_images", exist_ok=True)
os.makedirs("predictions/segmented_images2", exist_ok=True)
os.makedirs("predictions/results", exist_ok=True)

path = "training/logs/checkpoints"
restore_path = sorted(os.listdir(path))
checkpoint = torch.load(os.path.join(path, restore_path[-1]), map_location=lambda storage, loc: storage)

# Compute mean and std from raw dataset
t_ds = get_dataset(image_dir=TEST_IMG_DIR, mask_dir=TEST_MASK_DIR)
t_loader = get_loader(t_ds, batch_size=batch_size, shuffle=False)
t_mean, t_std = compute_mean_and_std(t_loader)
test_transform = composed_transform(t_mean, t_std)

# Normalized dataset and loader for prediction
test_ds = get_dataset(image_dir=TEST_IMG_DIR, mask_dir=TEST_MASK_DIR, transform=test_transform)
test_loader = get_loader(dataset=test_ds, batch_size=batch_size, num_workers=4, shuffle=False, pin_memory=True)

bimsnet = BIMSNet(in_channels=1, out_channels=1)
bimsnet.to(device)
bimsnet.load_state_dict(checkpoint["model_state"])
# bimsnet.train(False)
bimsnet.eval()

result_path = "predictions/results"
result = {"acc": [], "SE": [], "SP": [], "PC": [], "F1": [], "JS": [], "DC": []}

for step, (inp_image, target_mask) in enumerate(test_loader):
    batch_done = int((step + 1) / inp_image.size(0))

    inp_image = inp_image.to(device)
    GT = target_mask.to(device)
    SR = torch.sigmoid(bimsnet(inp_image))

    result["acc"].append(accuracy(SR, GT).cpu().detach().item())
    result["SE"].append(accuracy(SR, GT).cpu().detach().item())
    result["SP"].append(accuracy(SR, GT).cpu().detach().item())
    result["PC"].append(accuracy(SR, GT).cpu().detach().item())
    result["F1"].append(accuracy(SR, GT).cpu().detach().item())
    result["JS"].append(accuracy(SR, GT).cpu().detach().item())
    result["DC"].append(accuracy(SR, GT).cpu().detach().item())

    pred_mask = (SR > 0.5).float()

    if batch_done % sample_interval == 0:
        save_tensor_images(filename="predictions/segmented_stacks", index=batch_done, input_image=inp_image,
                           target_mask=GT, pred_mask=SR, nrow=1, normalize=True)
        save_segmented_images(filename="predictions/segmented_images", pred_mask=pred_mask, index=batch_done)
        save_segmented_images2(filename="predictions/segmented_images2", pred_mask=pred_mask, index=batch_done)

# Average the result
col_names = list(result.keys())
avg_result = pd.DataFrame.from_dict(result)
avg_result = avg_result.mean(axis=0, skipna=True)
np.savetxt(f"{result_path}/avg_result.txt", avg_result.values, fmt="%.6f", header=f"{col_names}", comments=" ")

with open(os.path.join(result_path, "result.csv"), "w") as f:
    writer = csv.writer(f, delimiter=",")
    writer.writerow(col_names)
    writer.writerows(zip(*[result[col_name] for col_name in col_names]))
