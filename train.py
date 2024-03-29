import argparse
import os.path as osp
import pandas as pd

import torch
from torch.nn import Linear, SiLU

from torch_geometric.datasets import QM9
from torch_geometric.nn.inits import glorot_orthogonal
from tqdm import tqdm
from dataset import ExDataset, FakeQM9, GDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DimeNet, DimeNetPlusPlus
# from torch_geometric.nn.models.dimenet import OutputPPBlock

import torch.nn.functional as F
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

from torch_scatter import scatter
from torch_geometric.nn import radius_graph


class MyOutputPPBlock(torch.nn.Module):
    def __init__(self, num_radial, hidden_channels, out_emb_channels,
                 out_channels, num_layers, act):
        super().__init__()
        self.act = act

        self.lin_rbf = Linear(num_radial, hidden_channels, bias=False)

        # The up-projection layer:
        self.lin_up = Linear(hidden_channels, out_emb_channels, bias=False)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(Linear(out_emb_channels, out_emb_channels))
        # self.lin = Linear(out_emb_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.data.fill_(0)

    def forward(self, x, rbf, i, num_nodes=None):
        x = self.lin_rbf(rbf) * x
        x = scatter(x, i, dim=0, dim_size=num_nodes)
        x = self.lin_up(x)
        for lin in self.lins:
            x = self.act(lin(x))
        return x
        # return self.lin(x)


class MyDimenet(DimeNetPlusPlus):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.output_blocks = torch.nn.ModuleList([
            MyOutputPPBlock(kwargs['num_radial'], kwargs['hidden_channels'], kwargs['out_emb_channels'],
                          kwargs['out_channels'], kwargs['num_output_layers'], SiLU())
            for _ in range(kwargs['num_blocks'] + 1)
        ])
            

        self.reset_parameters()

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
        x = self.emb(z, rbf, i, j)  # 여기서 왜 370개 노드가 4238개가 되는걸까 -> edge 개수만큼
        P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0)) # 다시 노드개수로 압축

        return x, rbf, sbf, idx_kj, idx_ji, P, i  # ex x, P) (E=8060, 128), (N=710, 256)

    def forward(self, z, pos, batch, z2=None, pos2=None, batch2=None):
        # return lambda_g or lambda_e

        # Preprocess.
        x, rbf, sbf, idx_kj, idx_ji, P, i = self.preprocess(z, pos, batch)

        # Interaction blocks.  After scatter, (Graph=32, H)
        result = [scatter(P, batch, dim=0)]
        for interaction_block, output_block in zip(self.interaction_blocks,
                                                   self.output_blocks[1:]):
            # x는 각 edge의 feature를 나타낸다.
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            # dropout
            # x = F.dropout(x, p=0.5, training=self.training)
            # P += output_block(x, rbf, i)
            result.append(scatter(output_block(x, rbf, i), batch, dim=0))
        
        return result #torch.cat(result, dim=-1)  # scatter(P, batch, dim=0)


def main(config: DictConfig):
    # dataset = QM9(osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9'))
    g_dataset = GDataset(config.data.path)
    ex_dataset = ExDataset(config.data.path)

    model = MyDimenet(
        hidden_channels=128,
        out_channels=1, # 상관없음. 안쓰임.
        num_blocks=6,
        int_emb_size=64,
        basis_emb_size=8,
        out_emb_channels=128,
        num_spherical=7,
        num_radial=6,
        cutoff=5.0,
        max_num_neighbors=32,
        envelope_exponent=5,
        num_before_skip=1,
        num_after_skip=2,
        num_output_layers=3,
    )
    # model.load_state_dict(torch.load('data/dimenet_pretrained.pth'))
    # models.append(model)

    model_name = f"{config.train.name}-fold{config.data.fold_index}"
    model_checkpoint = ModelCheckpoint(monitor="val/score", save_on_train_epoch_end=False, save_top_k=1, dirpath=f"checkpoints/{model_name}", filename="{epoch}-{val/score:.4f}")

    # logger=WandbLogger(project="mot-finetuning", name=model_name)
    # logger.watch(model, log="all", log_freq=100)
    trainer = Trainer(
        gpus=config.train.gpus,
        # logger=logger,
        callbacks=[model_checkpoint, LearningRateMonitor("epoch")],
        precision=config.train.precision,
        max_epochs=config.train.epochs,
        amp_backend=amp_backend,
        gradient_clip_val=config.train.max_grad_norm,
        # val_check_interval=config.train.validation_interval,
        # accumulate_grad_batches=config.train.accumulate_grads,
        # accumulate_grad_batches={0: 8, 4: 4, 8: 1},
        # resume_from_checkpoint=config.train.resume_from,
        # progress_bar_refresh_rate=1,
        log_every_n_steps=10,
    )
    trainer.fit(FineTuningModule(config, model), datamodule=FineTuningDataModule(config, g_dataset, ex_dataset))

    # model = FineTuningModule.load_from_checkpoint('checkpoints/dimenet++-fold0/epoch=1-val/score=0.2848.ckpt', config=config, model=model)
    model = FineTuningModule.load_from_checkpoint(model_checkpoint.best_model_path, config=config, model=model)
    model.eval()
    
    preds = trainer.predict(model, datamodule=FineTuningDataModule(config, GDataset(config.data.path, mode="test"), ExDataset(config.data.path, mode="test")))
    # flatten list
    preds = [item.cpu().numpy() for sublist in preds for item in sublist]
    # preds = model.predictions
    submission = pd.read_csv("sample_submission.csv")
    submission.iloc[:, 1:] = preds
    submission.to_csv(f"submission/{model_name}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args, unknown_args = parser.parse_known_args()

    config = OmegaConf.load(args.config)
    config.merge_with_dotlist(unknown_args)

    main(config)
