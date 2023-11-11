import argparse

# make sure to install promptsource, transformers, and datasets!
import copy
import datetime
import os
import os as os
from typing import List, Optional

import fire
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

import api.util as util

# from src.interventions.ccs.utils import get
from api.data_classes import Distribution
from src.interventions.ccs.train import get_hs


def main(
    distribution_dirs: List[str],
    model_dir: str = None,
    output_paths: Optional[List[str]] = None,
    max_examples: int = None,
):
    generation_args = util.load_json(model_dir + "/configuration.json")
    for distribution_dir, output_path in zip(distribution_dirs, output_paths):
        test_dataset = Distribution(distribution_dir).test_dataset
        test_dataset.convert_to_pairs(one_pair_per_instruction=True)
        if max_examples != None:
            test_dataset.set_max_examples(max_examples)

        def load_single_generation(save_dir, generation_type="hidden_states"):
            # use the same filename as in save_generations
            filename = generation_type + ".npy"
            return np.load(os.path.join(save_dir, filename))

        def load_all_generations(save_dir):
            # load all the saved generations: neg_hs, pos_hs, and labels
            neg_hs = load_single_generation(
                save_dir, generation_type="negative_hidden_states"
            )
            pos_hs = load_single_generation(
                save_dir, generation_type="positive_hidden_states"
            )
            labels = load_single_generation(save_dir, generation_type="labels")

            return neg_hs, pos_hs, labels

        neg_hs_train, pos_hs_train, y_train = load_all_generations(model_dir)
        neg_hs_test, pos_hs_test, y_test = get_hs(
            generation_args["base_model_dir"], test_dataset
        )

        class MLPProbe(nn.Module):
            def __init__(self, d):
                super().__init__()
                self.linear1 = nn.Linear(d, 100)
                self.linear2 = nn.Linear(100, 1)

            def forward(self, x):
                h = F.relu(self.linear1(x))
                o = self.linear2(h)
                return torch.sigmoid(o)

        class CCS(object):
            def __init__(
                self,
                x0,
                x1,
                nepochs=1000,
                ntries=10,
                lr=1e-3,
                batch_size=-1,
                verbose=False,
                device="cuda",
                linear=True,
                weight_decay=0.01,
                var_normalize=False,
            ):
                # data
                self.var_normalize = var_normalize
                self.x0 = self.normalize(x0)
                self.x1 = self.normalize(x1)
                self.d = self.x0.shape[-1]

                # training
                self.nepochs = nepochs
                self.ntries = ntries
                self.lr = lr
                self.verbose = verbose
                self.device = device
                self.batch_size = batch_size
                self.weight_decay = weight_decay

                # probe
                self.linear = linear
                self.initialize_probe()
                self.best_probe = copy.deepcopy(self.probe)

            def initialize_probe(self):
                if self.linear:
                    self.probe = nn.Sequential(nn.Linear(self.d, 1), nn.Sigmoid())
                else:
                    self.probe = MLPProbe(self.d)
                self.probe.to(self.device)

            def normalize(self, x):
                """
                Mean-normalizes the data x (of shape (n, d))
                If self.var_normalize, also divides by the standard deviation
                """
                normalized_x = x - x.mean(axis=0, keepdims=True)
                if self.var_normalize:
                    normalized_x /= normalized_x.std(axis=0, keepdims=True)

                return normalized_x

            def get_tensor_data(self):
                """
                Returns x0, x1 as appropriate tensors (rather than np arrays)
                """
                x0 = torch.tensor(
                    self.x0, dtype=torch.float, requires_grad=False, device=self.device
                )
                x1 = torch.tensor(
                    self.x1, dtype=torch.float, requires_grad=False, device=self.device
                )
                return x0, x1

            def get_loss(self, p0, p1):
                """
                Returns the CCS loss for two probabilities each of shape (n,1) or (n,)
                """
                informative_loss = (torch.min(p0, p1) ** 2).mean(0)
                consistent_loss = ((p0 - (1 - p1)) ** 2).mean(0)
                return informative_loss + consistent_loss

            def get_acc(self, x0_test, x1_test, y_test):
                """
                Computes accuracy for the current parameters on the given test inputs
                """
                x0 = torch.tensor(
                    self.normalize(x0_test),
                    dtype=torch.float,
                    requires_grad=False,
                    device=self.device,
                )
                x1 = torch.tensor(
                    self.normalize(x1_test),
                    dtype=torch.float,
                    requires_grad=False,
                    device=self.device,
                )
                with torch.no_grad():
                    p0, p1 = self.best_probe(x0), self.best_probe(x1)
                avg_confidence = 0.5 * (p0 + (1 - p1))
                predictions = (avg_confidence.detach().cpu().numpy() < 0.5).astype(int)[
                    :, 0
                ]
                acc = (predictions == y_test).mean()
                acc = max(acc, 1 - acc)

                return acc

            def train(self):
                """
                Does a single training run of nepochs epochs
                """
                x0, x1 = self.get_tensor_data()
                permutation = torch.randperm(len(x0))
                x0, x1 = x0[permutation], x1[permutation]

                # set up optimizer
                optimizer = torch.optim.AdamW(
                    self.probe.parameters(), lr=self.lr, weight_decay=self.weight_decay
                )

                batch_size = len(x0) if self.batch_size == -1 else self.batch_size
                nbatches = len(x0) // batch_size

                # Start training (full batch)
                for epoch in range(self.nepochs):
                    for j in range(nbatches):
                        x0_batch = x0[j * batch_size : (j + 1) * batch_size]
                        x1_batch = x1[j * batch_size : (j + 1) * batch_size]

                        # probe
                        p0, p1 = self.probe(x0_batch), self.probe(x1_batch)

                        # get the corresponding loss
                        loss = self.get_loss(p0, p1)

                        # update the parameters
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                return loss.detach().cpu().item()

            def repeated_train(self):
                best_loss = np.inf
                for train_num in range(self.ntries):
                    self.initialize_probe()
                    loss = self.train()
                    if loss < best_loss:
                        self.best_probe = copy.deepcopy(self.probe)
                        best_loss = loss

                return best_loss

        # Train CCS without any labels
        ccs = CCS(neg_hs_train, pos_hs_train)
        ccs.repeated_train()

        # Evaluate
        ccs_acc = ccs.get_acc(neg_hs_test, pos_hs_test, y_test)
        print("CCS accuracy: {}".format(ccs_acc))
        result = {
            "model_dir": model_dir,
            "distribution_dir": distribution_dir,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "eval_accuracy": ccs_acc,
            "eval_score": ccs_acc,
        }
        util.save_json(result, output_path)


if __name__ == "__main__":
    fire.Fire(main)
