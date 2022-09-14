import argparse
import os.path as osp

import torch

from torch_geometric.datasets import QM9
from dataset import ExDataset, FakeQM9, GDataset
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
from torch_scatter import scatter
from torch_geometric.nn import radius_graph
class MyDimenet(DimeNetPlusPlus):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.lin = torch.nn.Linear(256, 1)

    def preprocess(self, z, pos, batch):
         # ground block
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)

        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = self.triplets(
            edge_index, num_nodes=z.size(0))

        # Calculate distances.
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        # Calculate angles.
        pos_i = pos[idx_i]
        pos_ji, pos_ki = pos[idx_j] - pos_i, pos[idx_k] - pos_i
        a = (pos_ji * pos_ki).sum(dim=-1)
        b = torch.cross(pos_ji, pos_ki).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)

        # Embedding block.
        x = self.emb(z, rbf, i, j)
        P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))

        return x, rbf, sbf, idx_kj, idx_ji, P, i

    def forward(self, z, pos, batch, z2=None, pos2=None, batch2=None):
        # Preprocess.
        x, rbf, sbf, idx_kj, idx_ji, P, i = self.preprocess(z, pos, batch)
        # x2, rbf2, sbf2, idx_kj2, idx_ji2, P2 = self.preprocess(z2, pos2, batch2)

        # Interaction blocks.
        for interaction_block, output_block in zip(self.interaction_blocks,
                                                   self.output_blocks[1:]):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)  # 
            
            P += output_block(x, rbf, i)


        return P.sum(dim=0) if batch is None else scatter(P, batch, dim=0)

def main(config: DictConfig):
    # dataset = QM9(osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9'))
    g_dataset = GDataset(config.data.path)
    ex_dataset = ExDataset(config.data.path)

    model = MyDimenet(
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
        # amp_backend=amp_backend,
        gradient_clip_val=config.train.max_grad_norm,
        val_check_interval=config.train.validation_interval,
        accumulate_grad_batches=config.train.accumulate_grads,
        # resume_from_checkpoint=config.train.resume_from,
        # progress_bar_refresh_rate=1,
        log_every_n_steps=10,
    ).fit(FineTuningModule(config, model), datamodule=FineTuningDataModule(config, g_dataset, ex_dataset))

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
