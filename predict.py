import argparse
import os.path as osp

import torch
import pandas as pd

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
from train import MyDimenet

@torch.no_grad()
def main(config: DictConfig):
    # dataset = QM9(osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9'))
    g_dataset = GDataset(config.data.path, mode='test')
    ex_dataset = ExDataset(config.data.path, mode='test')

    dataset = list(zip(g_dataset, ex_dataset))
    test_dataloader = DataLoader(dataset, batch_size=config.train.batch_size, shuffle=False, num_workers=4)
    
    model = MyDimenet(
            hidden_channels=128,
            # out_channels=1,
            out_channels=2,
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
    ckpt = torch.load(config.infer.predict_from)['state_dict']
    try:
        model.load_state_dict(ckpt)
    except:
        for k in list(ckpt.keys()):
            if 'model.' in k:
                ckpt[k.replace('model.', '')] = ckpt.pop(k)
        model.load_state_dict(ckpt)

    module = FineTuningModule(config, model)
    predictions = Trainer(
        gpus=config.train.gpus,
        precision=config.train.precision,
    ).predict(module, dataloaders=test_dataloader)

    # export predictions
    preds = torch.cat(predictions).cpu().numpy()
    df = pd.read_csv('data/raw/sample_submission.csv')
    df.iloc[:, 1:] = preds
    df.to_csv(config.infer.filename, index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args, unknown_args = parser.parse_known_args()

    config = OmegaConf.load(args.config)
    config.merge_with_dotlist(unknown_args)

    main(config)
