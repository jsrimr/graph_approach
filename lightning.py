import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from sklearn.model_selection import KFold
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torch.utils.data import DataLoader, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn.inits import glorot_orthogonal

from dataset import FakeQM9
# from encoding import MolecularEncoder
# from modeling import MoTConfig, MoTLayerNorm, MoTModel

try:
    from apex.optimizers import FusedAdam as AdamW
except ModuleNotFoundError:
    from torch.optim import AdamW


class FineTuningModule(pl.LightningModule):
    def __init__(self, config: DictConfig, model):
        super().__init__()
        self.config = config
        # self.Eg, self.Eex = models
        self.model = model
        self.mlp = nn.Sequential(
            nn.Linear(256*2, 512), nn.SiLU(), nn.Dropout(0.5),
            nn.Linear(512, 2, bias=False)
        )

    def forward(
        self, g_data, ex_data, labels
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        g = self.model(*g_data)
        ex = self.model(*ex_data)

        # logits = self.mlp(torch.cat([g - ex], dim=1))  # g = (B,128 * 4)
        out = []
        for i in range(6):
            o = self.mlp(torch.cat([g[i]+ex[i], g[i]-ex[i]], dim=1))  # g = (B,128 * 4)
            out.append(o)
        logits = torch.stack(out, dim=1)  # (B, 6, 2)
        logits = torch.mean(logits, dim=1)  # (B, 2)

        labels = labels.view(-1, 2)  # TODO : 하드코딩 되어있음. 수정 필요.
        mse_loss = F.mse_loss(logits, labels.type_as(logits))
        mae_loss = F.l1_loss(logits, labels.type_as(logits))
        return mse_loss, mae_loss

    def training_step(self, batch: Dict[str, torch.Tensor], idx: int) -> torch.Tensor:
        # batch_g, batch_ex, labels = batch.g_data, batch.ex_data, batch.y
        batch_g, batch_ex = batch
        # implicit 하게 cuda로 옮겨지므로 미리 .to() 할 수 있는 형태로 쪼개 놓아야 함.
        mse_loss, mae_loss = self([batch_g.z, batch_g.pos, batch_g.batch], [
                                  batch_ex.z, batch_ex.pos, batch_ex.batch],
                                  batch_g.y)
        self.log("train/mse_loss", mse_loss)
        self.log("train/mae_loss", mae_loss)
        self.log("train/score", mae_loss)
        return mse_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], idx: int):
        # batch_g, batch_ex, labels = batch.g_data, batch.ex_data, batch.y
        batch_g, batch_ex = batch
        # implicit 하게 cuda로 옮겨지므로 미리 .to() 할 수 있는 형태로 쪼개 놓아야 함.
        mse_loss, mae_loss = self([batch_g.z, batch_g.pos, batch_g.batch], [
                                  batch_ex.z, batch_ex.pos, batch_ex.batch],
                                  batch_g.y)
        # mse_loss, mae_loss = self(batch.z, batch.pos, batch.batch)  # implicit 하게 cuda로 옮겨지므로 미리 .to() 할 수 있는 형태로 쪼개 놓아야 함.
        self.log("val/mse_loss", mse_loss)
        self.log("val/mae_loss", mae_loss)
        self.log("val/score", mae_loss)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        batch_g, batch_ex = batch
        logits = self.model(*[batch_g.z, batch_g.pos, batch_g.batch], *[
            batch_ex.z, batch_ex.pos, batch_ex.batch])
        return logits

    # def adjust_learning_rate(self, steps: int) -> float:
        
    #     if steps < self.config.train.warmup_steps:
    #         return float(steps) / float(max(1, self.config.train.warmup_steps))
        
    #     total_steps = self.config.train.epochs * len(self.trainer.train_dataloader())
    #     return max(
    #         0.0,
    #         float(total_steps - steps)
    #         / float(max(1, total_steps - self.config.train.warmup_steps)),
    #     )

    def create_param_groups(self) -> List[Dict[str, Any]]:
        """Create parameter groups for the optimizer.

        Transformer-based models are usually optimized by AdamW (weight-decay decoupled
        Adam optimizer). And weight decaying are applied to only weight parameters, not
        bias and layernorm parameters. Hence, this method creates parameter groups which
        contain parameters for weight-decay and ones for non-weight-decay. Using this
        paramter groups, you can separate which parameters should not be decayed from
        entire parameters in this model.

        Returns:
            A list of parameter groups.
        """
        do_decay_params, no_decay_params = [], []
        n_param = 0
        for layer in self.modules():
            for name, param in layer.named_parameters(recurse=False):
                print(name, param.numel())
                n_param += param.numel()
                if name == "bias":
                    no_decay_params.append(param)
                else:
                    do_decay_params.append(param)

        return [
            {"params": do_decay_params},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[Dict[str, Any]]]:
        optimizer = AdamW(self.create_param_groups(), **
                          self.config.train.optimizer)
        # optimizer = AdamW(self.parameters(), **self.config.train.optimizer)
        # optimizer = Adam(self.parameters())
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self.adjust_learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, threshold=0.01, min_lr=1e-6, threshold_mode="rel")
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/score"}


class FineTuningDataModule(pl.LightningDataModule):
    def __init__(self, config: DictConfig, g_dataset, ex_dataset):
        super().__init__()
        self.config = config
        # self.dataset = dataset
        self.g_dataset = g_dataset
        self.ex_dataset = ex_dataset

    def setup(self, stage: Optional[str] = None):

        self.dataset = list(zip(self.g_dataset, self.ex_dataset))

        # TODO : 전부 train_dataset 으로 활용하는 방법 찾기
        # Split the structure file list into k-folds. Note that the splits will be same
        # because of the random seed fixing.
        kfold = KFold(self.config.data.num_folds,
                      shuffle=True, random_state=42)
        train_val_sets = list(kfold.split(self.dataset))[
            self.config.data.fold_index]

        self.train_dataset = [self.dataset[idx] for idx in train_val_sets[0]]
        self.val_dataset = [self.dataset[idx] for idx in train_val_sets[1]]
        # train_dataset = self.data_group[train_val_sets[0]]
        # val_dataset = self.data_group[train_val_sets[1]]

        # self.train_dataset = MyDataset( train_dataset )
        # self.val_dataset = MyDataset( val_dataset )

    @property
    def num_dataloader_workers(self) -> int:
        """Get the number of parallel workers in each dataloader."""
        if self.config.data.dataloader_workers >= 0:
            return self.config.data.dataloader_workers
        return os.cpu_count()


    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train.batch_size,
            num_workers=self.num_dataloader_workers,
            # collate_fn=self.dataloader_collate_fn,
            persistent_workers=True,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.train.batch_size,
            num_workers=self.num_dataloader_workers,
            # collate_fn=self.dataloader_collate_fn,
            # drop_last=True,
            persistent_workers=True,
        )
