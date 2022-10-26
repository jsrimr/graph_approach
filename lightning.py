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
from torch.optim.lr_scheduler import LambdaLR
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
            nn.Linear(1024, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
        )

    def forward(
        self, g_data, ex_data, labels
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        g = self.model(*g_data)
        ex = self.model(*ex_data)

        logits = self.mlp(torch.cat([g+ex, g - ex], dim=1))  # 왜 370,128 이지?

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

    def adjust_learning_rate(self, current_step: int) -> float:
        """Calculate a learning rate scale corresponding to current step.

        MoT pretraining uses a linear learning rate decay with warmups. This method
        calculates the learning rate scale according to the linear warmup decaying
        schedule. Using this method, you can create a learning rate scheduler through
        `LambdaLR`.

        Args:
            current_step: A current step of training.

        Returns:
            A learning rate scale corresponding to current step.
        """
        training_steps = self.get_total_training_steps()
        warmup_steps = int(training_steps * self.config.train.warmup_ratio)

        if current_step < warmup_steps:
            return current_step / warmup_steps
        return max(0, (training_steps - current_step) / (training_steps - warmup_steps))

    def get_total_training_steps(self) -> int:
        """Calculate the total training steps from the trainer.

        If you are using epochs to limit training on pytorch lightning, you cannot
        directly get the entire training steps. It requires some complicated ways to get
        the training steps. This method uses the number of samples in the dataloader,
        distributed devices and accumulations to get approximately correct training
        steps.

        Returns:
            The total training steps.
        """
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = (
            min(batches, limit_batches)
            if isinstance(limit_batches, int)
            else int(limit_batches * batches)
        )

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[Dict[str, Any]]]:
        optimizer = AdamW(self.create_param_groups(), **
                          self.config.train.optimizer)
        # optimizer = AdamW(self.parameters(), **self.config.train.optimizer)
        # optimizer = Adam(self.parameters())
        scheduler = LambdaLR(optimizer, self.adjust_learning_rate)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


class FineTuningDataModule(pl.LightningDataModule):
    def __init__(self, config: DictConfig, g_dataset, ex_dataset):
        super().__init__()
        self.config = config
        # self.dataset = dataset
        self.g_dataset = g_dataset
        self.ex_dataset = ex_dataset

    def setup(self, stage: Optional[str] = None):
        # fine same idx data and concat
        # data_group = [[] for _ in range(len(self.dataset))]
        # for data in self.dataset:
        #     data_group[data.idx].append(data)
        # for data in self.dataset:
        #     data_group[data.idx].append(data.y)
        # self.data_group = np.array(data_group)
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

    # def dataloader_collate_fn(
    #     self, features: List[Dict[str, Union[torch.Tensor, int, str]]]
    # ) -> Tuple[List[str], Dict[str, torch.Tensor]]:
    #     """Simple datacollate binder for dataloaders."""
    #     uids = [dict_['uid'] for dict_ in features]
    #     encoding_gs = [dict_['encoding_g'] for dict_ in features]
    #     encoding_ex = [dict_['encoding_ex'] for dict_ in features]
    #     labels = [dict_['labels'] for dict_ in features]

    #     encoding_gs = self.encoder.collate(
    #         encoding_gs,
    #         max_length=self.config.data.max_length,
    #         pad_to_multiple_of=8,
    #     )
    #     encoding_ex = self.encoder.collate(
    #         encoding_ex,
    #         max_length=self.config.data.max_length,
    #         pad_to_multiple_of=8,
    #     )
    #     # list of numpy to tensor
    #     labels = torch.tensor(labels, dtype=torch.float32)
    #     return uids, (encoding_gs, encoding_ex, labels)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train.batch_size,
            num_workers=self.num_dataloader_workers,
            # collate_fn=self.dataloader_collate_fn,
            persistent_workers=True,
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
