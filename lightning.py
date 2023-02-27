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
    from apex.optimizers import FusedAdam as Adam
    from apex.optimizers import FusedSGD as SGD
except ModuleNotFoundError:
    from torch.optim import AdamW
    from torch.optim import Adam
    from torch.optim import SGD


class FineTuningModule(pl.LightningModule):
    def __init__(self, config: DictConfig, model: nn.Module):
        super().__init__()
        self.config = config
        # self.Eg, self.Eex = models
        self.model = model

        self.q = nn.Linear(128, 128)
        self.k = nn.Linear(128, 128)
        self.v = nn.Linear(128, 128)
        
        self.mlp = nn.Sequential(
            # nn.Linear(256*2, 512), nn.SiLU(), nn.Dropout(0.5),
            nn.Linear(128*7, 512), nn.SiLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.SiLU(), nn.Dropout(0.5),
            nn.Linear(256, 64), nn.SiLU(), nn.Dropout(0.5),
            nn.Linear(64, 2, bias=False)
        )

    def forward(
        self, g_data, ex_data, labels=None, mode="train"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        g = self.model(*g_data)
        ex = self.model(*ex_data)

        g = torch.stack(g).transpose(0,1)  # (B, L, 128)
        ex = torch.stack(ex).transpose(0,1) # (B, L, 128)
        f = g - ex  # (B, L, 128)
        # self attention
        q_f = self.q(f)  # (B, L, 128)
        k_f = self.k(f)  # (B, L, 128)
        v_f = self.v(f)  # (B, L, 128)
        attn = torch.bmm(q_f, k_f.transpose(1,2))  # (B, L, L)
        attn = F.softmax(attn, dim=2)  # (B, L, L)
        features = torch.bmm(attn, v_f)  # (B, L, 128)
        features = features.view(features.size(0), -1)  # (B, 128 * L)
        logits = self.mlp(features)  # (B, 2)

        if mode == "test":
            return logits
        # attn = torch.bmm(g, ex.transpose(1,2))  # (B, L, L)
        # attn = F.softmax(attn, dim=2)  # (B, L, L)

        # features = torch.bmm(attn, g-ex)  # (B, L, 128)
        # features = features.view(features.size(0), -1)  # (B, 128 * L)
        # logits = self.mlp(features)  # (B, 2)

        # out = []
        # for i in range(6):
        #     # g = (B,128 * 4)
        #     o = self.mlp(torch.cat([g[i]+ex[i], g[i]-ex[i]], dim=1))
        #     out.append(o)
        # logits = torch.stack(out, dim=1)  # (B, 6, 2)
        # logits = torch.mean(logits, dim=1)  # (B, 2)

        labels = labels.view(-1, 2)  # TODO : 하드코딩 되어있음. 수정 필요.
        # mse_loss = torch.mean((logits[:,0]- labels[:,0])**2)
        # mae_loss = torch.mean(torch.abs(logits[:,0]- labels[:,0]))
        # compute mse_loss and mae_loss not using torch.nn.MSELoss and torch.nn.L1Loss
        # because of the difference of the reduction method.
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
        # print("val/score", mae_loss)

        # log learning rate
        for param_group in self.trainer.optimizers[0].param_groups:
            self.log("lr", param_group['lr'])


    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        batch_g, batch_ex = batch
        logits = self([batch_g.z, batch_g.pos, batch_g.batch], [
            batch_ex.z, batch_ex.pos, batch_ex.batch], mode="test")
        return logits


    def configure_optimizers(self) -> Tuple[List[Optimizer], List[Dict[str, Any]]]:
        # optimizer = AdamW(self.create_param_groups(), ** self.config.train.optimizer)
        # SGD optimizer
        # optimizer = SGD(self.create_param_groups(), **self.config.train.optimizer)
        # optimizer = SGD(self.parameters(
        # ), lr=self.config.train.optimizer.lr, nesterov=True, momentum=0.9, weight_decay=0.0001)
        optimizer = AdamW(self.parameters())
        # optimizer = Adam(self.parameters())
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self.adjust_learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1, threshold=0.01, min_lr=1e-6, threshold_mode="rel")
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=self.config.train.epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/score"}
        # return {"optimizer": optimizer, "monitor": "val/score"}


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
                      shuffle=True, random_state=self.config.data.random_seed)
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
    
    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.config.train.batch_size,
            num_workers=self.num_dataloader_workers,
            # collate_fn=self.dataloader_collate_fn,
            # drop_last=True,
            persistent_workers=True,
        )