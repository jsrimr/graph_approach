import argparse
import os.path as osp

import torch

# from torch_geometric.datasets import QM9
from dataset import FakeQM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DimeNet, DimeNetPlusPlus

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from lightning import FineTuningDataModule, FineTuningModule

try:
    import apex

    amp_backend = apex.__name__
except ModuleNotFoundError:
    amp_backend = "native"

# parser = argparse.ArgumentParser()
# parser.add_argument('--use_dimenet_plus_plus', action='store_true')
# args = parser.parse_args()

# Model = DimeNetPlusPlus if args.use_dimenet_plus_plus else DimeNet

def main(config: DictConfig):
    # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9')
    dataset = FakeQM9(config.data.path)
    # target = None

    model = DimeNetPlusPlus(
            hidden_channels=128,
            out_channels=1,
            num_blocks=4,
            int_emb_size=64,
            basis_emb_size=8,
            out_emb_channels=256,
            num_spherical=7,
            num_radial=6,
            cutoff=5.0,
            max_num_neighbors=32,
            envelope_exponent=5,
            num_before_skip=1,
            num_after_skip=2,
            num_output_layers=3,
        )
    model.load_state_dict(torch.load('data/dimenet_pretrained.pth'))
    model_name = f"{config.train.name}-fold{config.data.fold_index}"
    model_checkpoint = ModelCheckpoint(monitor="val/score", save_weights_only=True)

    Trainer(
        gpus=config.train.gpus,
        # logger=WandbLogger(project="mot-finetuning", name=model_name),
        callbacks=[model_checkpoint, LearningRateMonitor("step")],
        precision=config.train.precision,
        max_epochs=config.train.epochs,
        amp_backend=amp_backend,
        gradient_clip_val=config.train.max_grad_norm,
        val_check_interval=config.train.validation_interval,
        accumulate_grad_batches=config.train.accumulate_grads,
        # resume_from_checkpoint=config.train.resume_from,
        # progress_bar_refresh_rate=1,
        log_every_n_steps=10,
    ).fit(FineTuningModule(config, model), datamodule=FineTuningDataModule(config, dataset))

    model = FineTuningModule.load_from_checkpoint(
        model_checkpoint.best_model_path, config=config
    )
    torch.save(model.state_dict(), model_name + ".pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args, unknown_args = parser.parse_known_args()

    config = OmegaConf.load(args.config)
    config.merge_with_dotlist(unknown_args)

    main(config)
