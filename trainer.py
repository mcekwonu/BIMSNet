"""This code was inspired by Johannes Schmidt: https://johschmidt42.medium.com/.
Written by Michael Chukwuemeka Ekwonu: https://github.com/mcekwonu/bimsnet
"""

import os
import time
import torch
import warnings
import numpy as np
import torch.optim as optim

from torchvision import transforms
from datetime import datetime

from network import BIMSNet
from metrics import BCELoss, dice_score
from utils import *

warnings.filterwarnings("ignore")
seed = 24
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


class Trainer:
    """BIMS Trainer class

    Parameters:
        model (nn.Module): Model, deep neural network
        train_dir (str): Training directory containing images and masks
        val_dir (str): Validation directory containing images and masks
        loss_fn (nn.Module): Loss function (criterion) for computing network loss
        optimizer (torch.optim): Optimizer
        batch_size (int): mini-batch of image-mask pair. Default=8
        device (str): device. Default="cuda"
        num_epochs (int): Number of epochs. Default=100
        num_workers (int): Number of processors to use. Default=8
        best_loss (float): Initial best loss. Default=inf
        pin_memory (bool): Default=True
        scheduler (bool): Learning rate scheduler (uses ReduceLROnPlateau method). Default=True
        display_step (int): Iteration number to print. Default=100
        sample_interval (int): Saves stack of input image, mask and predicted mask.Default=1000
        verbose (bool): To display target and predicted image on image grid. Default=None
    """

    def __init__(self,
                 model,
                 optimizer,
                 loss_fn,
                 batch_size=8,
                 lr=2e-6,
                 num_epochs=100,
                 train_dir="data/train",
                 val_dir="data/val",
                 device=torch.device("cuda"),
                 num_workers=8,
                 best_loss=float("inf"),
                 display_step=100,
                 sample_interval=1000,
                 pin_memory=True,
                 scheduler=True,
                 verbose=False,
                 ):
        self.device = device
        self.model = model.to(self.device)
        self.num_epochs = num_epochs
        self.lr = lr
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.display_step = display_step
        self.sample_interval = sample_interval
        self.verbose = verbose
        self.best_loss = best_loss
        self.phases = ["train", "valid"]

        self.losses = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}

        TRAIN_IMG_DIR = train_dir + "/images"
        TRAIN_MASK_DIR = train_dir + "/masks"
        VAL_IMG_DIR = val_dir + "/images"
        VAL_MASK_DIR = val_dir + "/masks"

        os.makedirs("training/saved_models", exist_ok=True)
        os.makedirs("training/saved_images", exist_ok=True)
        os.makedirs("training/logs/checkpoints", exist_ok=True)
        os.makedirs("training/logs/history", exist_ok=True)

        # Compute mean and std for normalization of datasets
        train_ds = BIMSDataset(image_dir=TRAIN_IMG_DIR, mask_dir=TRAIN_MASK_DIR)
        train_loader = get_loader(train_ds)
        train_mean, train_std = compute_mean_and_std(train_loader)
        train_transform = composed_transform(mean=train_mean, std=train_std)

        valid_ds = BIMSDataset(image_dir=VAL_IMG_DIR, mask_dir=VAL_MASK_DIR)
        valid_loader = get_loader(valid_ds)
        val_mean, val_std = compute_mean_and_std(valid_loader)
        val_transform = composed_transform(mean=val_mean, std=val_std)

        # Prepare normalized datasets for training and validation
        self.train_ds = get_dataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, transform=train_transform)
        self.val_ds = get_dataset(VAL_IMG_DIR, VAL_MASK_DIR, transform=val_transform)
        self.train_loader = get_loader(self.train_ds, batch_size, num_workers, shuffle=True, pin_memory=pin_memory)
        self.val_loader = get_loader(self.train_ds, batch_size, num_workers, shuffle=False, pin_memory=pin_memory)

        self.dataloaders = {"train": self.train_loader, "valid": self.val_loader}

        if scheduler:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", patience=5)

    def fit(self):
        model_path = f"training/saved_models/{self.model.__class__.__qualname__.lower()}.pth"
        date = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_path = f"training/logs/checkpoints/ckpt_{date}.pth"

        start = time.time()

        for epoch in range(self.num_epochs):
            print("\nEpoch {}/{}: ".format(epoch + 1, self.num_epochs))
            print("-" * 15)

            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
            }

            for phase in self.phases:
                if phase == "train":
                    self.model.train()
                    dataloaders = self.dataloaders[phase]
                else:
                    self.model.eval()
                    dataloaders = self.dataloaders[phase]

                running_loss = 0.0

                for step, (inp_image, target_mask) in enumerate(dataloaders):
                    batches_done = epoch * len(self.dataloaders[phase]) + step

                    inp_image = inp_image.to(self.device)
                    target_mask = target_mask.to(self.device)

                    if phase == "train":
                        self.optimizer.zero_grad()
                        pred_mask = self.model(inp_image)
                        loss = self.loss_fn(pred_mask, target_mask)
                        loss.backward()
                        self.optimizer.step()
                    else:
                        with torch.no_grad():
                            pred_mask = self.model(inp_image)
                            loss = self.loss_fn(pred_mask, target_mask)

                    dice = dice_score(pred_mask, target_mask)
                    running_loss += loss.item() * self.dataloaders.__len__()
                    curr_lr = self.optimizer.param_groups[0]["lr"]

                    if step % self.display_step == 0:
                        print(
                            "Current step: {}  Loss: {:.6f}  Dice score: {:.6f}  LR: {:.1e}".format(
                                step, loss.item(), dice, curr_lr)
                        )
                        torch.save(state, ckpt_path)

                    if batches_done % self.sample_interval == 0:
                        save_tensor_images(filename="training/saved_images", index=batches_done, input_image=inp_image,
                                           target_mask=target_mask, pred_mask=pred_mask, nrow=1, normalize=True)

                epoch_loss = running_loss / len(dataloaders.dataset)
                self.scheduler.step(epoch_loss)
                self.losses[phase].append(epoch_loss)
                self.dice_scores[phase].append(dice.cpu().numpy())
                torch.cuda.empty_cache()

                print("\n{} Loss: {:.6f}\n".format(phase.capitalize(), epoch_loss))

                # log epoch of metric: losses and dice scores
                epoch_log("training/logs/history/loss.txt", epoch, phase, epoch_loss)
                epoch_log("training/logs/history/dice_score.txt", epoch, phase, dice)

                if phase == "valid" and epoch_loss < self.best_loss:
                    print("***** New optimal found, saving state... *****")
                    state["best_loss"] = self.best_loss = epoch_loss
                    torch.save(state, model_path)

        np.savez(f"training/logs/history/losses_{epoch + 1}", **self.losses)
        np.savez(f"training/logs/history/dice_scores_{epoch + 1}", **self.dice_scores)

        time_elapsed = time.time() - start
        msg = convert_to_time(time_elapsed)
        print("Best Validation Loss: {:.6f}".format(self.best_loss))

        timer_log("training/logs/history/training_time.txt", msg)

    def resume_fit(self):
        model_path = f"training/saved_models/{self.model.__class__.__qualname__.lower()}.pth"
        date = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpts_path = f"training/logs/checkpoints/ckpt_{date}.pth"

        print("\nResuming training from last saved checkpoint ...")
        path = "training/logs/checkpoints"
        restore_path = sorted(os.listdir(path))
        ckpt = torch.load(os.path.join(path, restore_path[-1]))

        self.best_loss = ckpt["best_loss"]
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.model.load_state_dict(ckpt["model_state"])
        epochs = ckpt["epoch"]

        start = time.time()

        for epoch in range(epochs, self.num_epochs):
            print("\nEpoch {}/{}: ".format(epoch + 1, self.num_epochs))
            print("-" * 15)

            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
            }

            for phase in self.phases:
                if phase == "train":
                    self.model.train()
                    dataloaders = self.dataloaders[phase]
                else:
                    self.model.eval()
                    dataloaders = self.dataloaders[phase]

                running_loss = 0.0

                for step, (inp_image, target_mask) in enumerate(dataloaders):
                    batches_done = epoch * len(self.dataloaders[phase]) + step

                    inp_image = inp_image.to(self.device)
                    target_mask = target_mask.to(self.device)

                    if phase == "train":
                        self.optimizer.zero_grad()
                        pred_mask = self.model(inp_image)
                        loss = self.loss_fn(pred_mask, target_mask)
                        loss.backward()
                        self.optimizer.step()
                    else:
                        with torch.no_grad():
                            pred_mask = self.model(inp_image)
                            loss = self.loss_fn(pred_mask, target_mask)

                    dice = dice_score(pred_mask, target_mask)

                    running_loss += loss.item() * self.dataloaders.__len__()
                    curr_lr = self.optimizer.param_groups[0]["lr"]

                    if step % self.display_step == 0:
                        print(
                            "Current step: {}  Loss: {:.6f}  Dice score: {:.6f}  LR: {:.1e}".format(
                                step, loss.item(), dice, curr_lr)
                        )
                        torch.save(state, ckpts_path)

                    if batches_done % self.sample_interval == 0:
                        save_tensor_images(filename="training/saved_images", index=batches_done, input_image=inp_image,
                                           target_mask=target_mask, pred_mask=pred_mask, nrow=1, normalize=True)

                epoch_loss = running_loss / len(dataloaders.dataset)
                self.scheduler.step(epoch_loss)
                self.losses[phase].append(epoch_loss)
                self.dice_scores[phase].append(dice.cpu().numpy())
                torch.cuda.empty_cache()

                print("\n{} Loss: {:.4f}\n".format(phase.capitalize(), epoch_loss))

                # log loss and dice score for each epoch
                epoch_log("training/logs/history/loss.txt", epoch + 1, phase, epoch_loss)
                epoch_log("training/logs/history/dice_score.txt", epoch + 1, phase, dice)

                if phase == "valid" and epoch_loss < self.best_loss:
                    print("***** New optimal found, saving state... *****")
                    state["best_loss"] = self.best_loss = epoch_loss
                    torch.save(state, model_path)

        np.savez(f"training/logs/history/losses_{epoch + 1}", **self.losses)
        np.savez(f"training/logs/history/dice_scores_{epoch + 1}", **self.dice_scores)

        time_elapsed = time.time() - start
        msg = convert_to_time(time_elapsed)
        print("Best Validation Loss: {:.6f}".format(self.best_loss))

        timer_log("training/logs/history/training_time.txt", msg)


def main(RESUME=None):
    model = BIMSNet(in_channels=1, out_channels=1)
    LEARNING_RATE = 2e-6
    BATCH_SIZE = 10
    NUM_EPOCHS = 100
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = BCELoss()

    print("Training BIMSNet with BCE Loss")
    trainer = Trainer(model, optimizer, loss_fn, batch_size=BATCH_SIZE, lr=LEARNING_RATE, num_epochs=NUM_EPOCHS,
                      train_dir="../databank/data_containers/train", val_dir="../databank/data_containers/val",
                      display_step=200, sample_interval=10_000)

    if RESUME:
        trainer.resume_fit()
    else:
        trainer.fit()


if __name__ == '__main__':
    main()
    # main(RESUME=True)
