"""This code was copied from Johannes Schmidt: https://johschmidt42.medium.com/, and
modified by Michael Chukwuemeka Ekwonu: https://github.com/mcekwonu/bimsnet"""
import os

import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
import math


class LearningRateFinder:
    """
    Train a model using different learning rates within a range to find the optimal learning rate

    Parameters:
        model (nn.Module): Model
        loss_fn (nn.criterion): Loss function
        optimizer (torch.optim): Optimizer
        device (str): Device. Default="cuda"
    """

    def __init__(self,
                 model,
                 loss_fn,
                 optimizer,
                 device="cuda"
                 ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.loss_history = {}
        self._model_init = model.state_dict()
        self._optimizer_init = optimizer.state_dict()

    def fit(self,
            train_dataloader,
            steps=100,
            init_lr=1e-10,
            max_lr=1,
            constant_increment=False
            ):
        """
        Trains the model for number of steps using varied learning rate and store the statistics

        Parameters:
            train_dataloader (torch.utils.dataloader): Train dataloader
            steps (int): Number of steps to train model
            init_lr (float): Initial learning rate
            max_lr (float): Final learning rate
            constant_increment (bool): Use constant increment for learning rate. Default=False
        """

        self.model.train()
        curr_lr = init_lr
        step = 0
        epochs = math.ceil(steps / len(train_dataloader))

        for epoch in range(epochs):
            for i, (x, y) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Training"):
                x, y = x.to(self.device), y.to(self.device)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = curr_lr
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss_fn(output, y)
                loss.backward()
                self.optimizer.step()
                self.loss_history[curr_lr] = loss.item()

                step += 1
                if step > steps:
                    break

                if constant_increment:
                    curr_lr += (max_lr - init_lr) / steps
                else:
                    curr_lr = curr_lr * (max_lr / init_lr) ** (1 / steps)

    def plot(self,
             save_dir,
             smoothing=True,
             clipping=True,
             smoothing_factor=0.1,
             verbose=None
             ):
        """
        Displays the loss vs learning rate (log scale)

        Parameters:
            save_dir: Output save directory
            smoothing (bool): Applies exponential smoothing to loss data. Default=True
            clipping (bool): Skips the first 10 values and the last 5 values. It removes
                            the initial and final high losses
            smoothing_factor (float): Exponential smoothing factor. Default=0.1
            verbose (bool): To display figure. Default=None
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        loss_data = pd.Series(list(self.loss_history.values()))
        lr = list(self.loss_history.keys())
        if smoothing:
            loss_data = loss_data.ewm(alpha=smoothing_factor).mean()
            loss_data = loss_data.divide(pd.Series(
                [1 - (1.0 - smoothing_factor) ** i for i in range(1, loss_data.shape[0] + 1)]
            ))
        if clipping:
            loss_data = loss_data[10:-5]
            lr = lr[10:-5]

        font = {"family": "sans-serif", "color": "black",
                "weight": "normal", "size": 13}

        plt.figure(figsize=(8, 6))
        plt.plot(lr, loss_data, color="green")
        plt.xscale("log")
        plt.xlabel("Learning rate (log scale)", fontdict=font)
        plt.ylabel("Loss (exponential moving average)", font)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/loss_lr.png", dpi=300)
        if verbose:
            plt.show()

    @property
    def reset(self):
        """Restore model and optimizer to its initial state"""
        self.model.load_state_dict(self._model_init)
        self.optimizer.load_state_dict(self._optimizer_init)
        print("Model and optimizer in initial state")
