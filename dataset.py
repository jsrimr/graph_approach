from pathlib import Path
from typing import Dict, List, Optional
import os
import os.path as osp
import sys
from typing import Callable, List, Optional

import pandas as pd
import torch
import torch.nn.functional as F
from torch_scatter import scatter
from tqdm import tqdm

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)

MOLECULAR_ATOMS = ['Br', 'I', 'N', 'P', 'C',
                   'Cl', 'F', 'Si', 'S', 'B', 'H', 'O']
HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

# target 만들때 사용되는데, 의미는 모르겠음
conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])

# # target 만들 때 사용
# atomrefs = {
#     6: [0., 0., 0., 0., 0.],
#     7: [
#         -13.61312172, -1029.86312267, -1485.30251237, -2042.61123593,
#         -2713.48485589
#     ],
#     8: [
#         -13.5745904, -1029.82456413, -1485.26398105, -2042.5727046,
#         -2713.44632457
#     ],
#     9: [
#         -13.54887564, -1029.79887659, -1485.2382935, -2042.54701705,
#         -2713.42063702
#     ],
#     10: [
#         -13.90303183, -1030.25891228, -1485.71166277, -2043.01812778,
#         -2713.88796536
#     ],
#     11: [0., 0., 0., 0., 0.],
# }
class GDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, mode='train'):
        self.mode = mode
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        if self.mode == 'train':
            return 'train_g.pt'
        else:
            return 'test_g.pt'

class ExDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, mode='train'):
        self.mode = mode
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        if self.mode == 'train':
            return 'train_g.pt'
        else:
            return 'test_g.pt'
    
class FakeQM9(InMemoryDataset):
    r"""The QM9 dataset from the `"MoleculeNet: A Benchmark for Molecular
    Machine Learning" <https://arxiv.org/abs/1703.00564>`_ paper, consisting of
    about 130,000 molecules with 19 regression targets.
    Each molecule includes complete spatial information for the single low
    energy conformation of the atoms in the molecule.
    In addition, we provide the atom features from the `"Neural Message
    Passing for Quantum Chemistry" <https://arxiv.org/abs/1704.01212>`_ paper.

    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | Target | Property                         | Description                                                                       | Unit                                        |
    +========+==================================+===================================================================================+=============================================+
    | 0      | :math:`\mu`                      | Dipole moment                                                                     | :math:`\textrm{D}`                          |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 1      | :math:`\alpha`                   | Isotropic polarizability                                                          | :math:`{a_0}^3`                             |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 2      | :math:`\epsilon_{\textrm{HOMO}}` | Highest occupied molecular orbital energy                                         | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 3      | :math:`\epsilon_{\textrm{LUMO}}` | Lowest unoccupied molecular orbital energy                                        | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 4      | :math:`\Delta \epsilon`          | Gap between :math:`\epsilon_{\textrm{HOMO}}` and :math:`\epsilon_{\textrm{LUMO}}` | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 5      | :math:`\langle R^2 \rangle`      | Electronic spatial extent                                                         | :math:`{a_0}^2`                             |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 6      | :math:`\textrm{ZPVE}`            | Zero point vibrational energy                                                     | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 7      | :math:`U_0`                      | Internal energy at 0K                                                             | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 8      | :math:`U`                        | Internal energy at 298.15K                                                        | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 9      | :math:`H`                        | Enthalpy at 298.15K                                                               | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 10     | :math:`G`                        | Free energy at 298.15K                                                            | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 11     | :math:`c_{\textrm{v}}`           | Heat capavity at 298.15K                                                          | :math:`\frac{\textrm{cal}}{\textrm{mol K}}` |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 12     | :math:`U_0^{\textrm{ATOM}}`      | Atomization energy at 0K                                                          | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 13     | :math:`U^{\textrm{ATOM}}`        | Atomization energy at 298.15K                                                     | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 14     | :math:`H^{\textrm{ATOM}}`        | Atomization enthalpy at 298.15K                                                   | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 15     | :math:`G^{\textrm{ATOM}}`        | Atomization free energy at 298.15K                                                | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 16     | :math:`A`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 17     | :math:`B`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 18     | :math:`C`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)

    Stats:
        .. list-table::
            :widths: 10 10 10 10 10
            :header-rows: 1

            * - #graphs
              - #nodes
              - #edges
              - #features
              - #tasks
            * - 130,831
              - ~18.0
              - ~37.3
              - 11
              - 19
    """  # noqa: E501

    # raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/'
    #            'molnet_publish/qm9.zip')
    # raw_url2 = 'https://ndownloader.figshare.com/files/3195404'
    # processed_url = 'https://data.pyg.org/datasets/qm9_v3.zip'

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 mode='train'):
        self.mode = mode
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])  # TODO : g, ex 파일 나눠서 저장해서 한번에 가져오는 상황이 안됨.

    @property
    def raw_file_names(self) -> List[str]:
        try:
            import rdkit  # noqa
            if self.mode == "train":
                return ['mol_files', 'train_set.ReorgE.csv']
            elif self.mode == "test":
                return ['mol_files']
        except ImportError:
            return
            # return ['qm9_v3.pt']

    @property
    def processed_file_names(self) -> str: # TODO : g, ex 파일 나눠서 저장해서 한번에 가져오는 상황이 안됨.
        if self.mode == "train":
            return 'data_v3.pt'
        else:
            return 'test_data.pt'

    def process(self):
        try:
            import rdkit
            from rdkit import Chem, RDLogger
            from rdkit.Chem.rdchem import BondType as BT
            from rdkit.Chem.rdchem import HybridizationType
            RDLogger.DisableLog('rdApp.*')

        except ImportError:
            rdkit = None

        if rdkit is None:
            print(("Using a pre-processed version of the dataset. Please "
                   "install 'rdkit' to alternatively process the raw data."),
                  file=sys.stderr)

            data_list = torch.load(self.raw_paths[0])
            data_list = [Data(**data_dict) for data_dict in data_list]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            torch.save(self.collate(data_list), self.processed_paths[0])
            return
        def extract_mol(file :str, i:int) -> Data:
            mol = Chem.MolFromMolFile(file)

            N = mol.GetNumAtoms()

            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)

            type_idx = []
            atomic_number = []
            aromatic = []
            sp = []
            sp2 = []
            sp3 = []
            num_hs = []
            for atom in mol.GetAtoms():
                type_idx.append(types[atom.GetSymbol()])  # C, N, O, F, H
                atomic_number.append(atom.GetAtomicNum())  # 원자번호
                aromatic.append(1 if atom.GetIsAromatic() else 0)  # 고리화
                hybridization = atom.GetHybridization()  # sp, sp2, sp3
                sp.append(1 if hybridization ==
                          HybridizationType.SP else 0)  # sp
                sp2.append(1 if hybridization ==
                           HybridizationType.SP2 else 0)  # sp2
                sp3.append(1 if hybridization ==
                           HybridizationType.SP3 else 0)  # sp3

            z = torch.tensor(atomic_number, dtype=torch.long) # 원자번호

            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [bonds[bond.GetBondType()]]

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = F.one_hot(edge_type,
                                  num_classes=len(bonds)).to(torch.float)

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_type = edge_type[perm]
            edge_attr = edge_attr[perm]

            row, col = edge_index
            hs = (z == 1).to(torch.float)
            num_hs = scatter(hs[row], col, dim_size=N).tolist()

            x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(types))  # 원자 종류
            x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs],
                              dtype=torch.float).t().contiguous() # 원자번호, 고리화, sp, sp2, sp3, num_hs
            x = torch.cat([x1.to(torch.float), x2], dim=-1)

            
            # name = mol.GetProp('_Name')

            if self.mode == "train":
                data = Data(x=x, z=z, pos=pos, edge_index=edge_index,y=target[i], edge_attr=edge_attr)
            else:
                data = Data(x=x, z=z, pos=pos, edge_index=edge_index, edge_attr=edge_attr)

            if self.pre_filter is not None and not self.pre_filter(data):
                return None
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            
            return data

        types = {el: i for i, el in enumerate(MOLECULAR_ATOMS)}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        if self.mode == "test":
            g_data_list = []
            ex_data_list = []
            g_list = sorted(Path(self.raw_paths[0]+'/test_set').glob("*_g.mol"))
            ex_list = sorted(Path(self.raw_paths[0]+'/test_set').glob("*_ex.mol"))

            for i, (g_file, ex_file) in enumerate(zip(g_list, ex_list)):
                g_data = extract_mol(str(g_file), i)
                g_data_list.append(g_data)
                ex_data = extract_mol(str(ex_file), i)
                ex_data_list.append(ex_data)

            torch.save(self.collate(g_data_list), os.path.dirname(self.processed_paths[0]) + '/test_g.pt')
            torch.save(self.collate(ex_data_list), os.path.dirname(self.processed_paths[0]) + '/test_ex.pt')
        
        else:           
            df = pd.read_csv(self.raw_paths[1], index_col=0)
            target = df[['Reorg_g', 'Reorg_ex']]
            target = torch.tensor(target.values, dtype=torch.float)

            # read from mol_files
            # data_list = []
            g_data_list = []
            ex_data_list = []
            g_list = sorted(Path(self.raw_paths[0]+'/train_set').glob("*_g.mol"))
            ex_list = sorted(Path(self.raw_paths[0]+'/train_set').glob("*_ex.mol"))


            for i, (g_file, ex_file) in enumerate(zip(g_list, ex_list)):
                g_data = extract_mol(str(g_file), i)
                g_data_list.append(g_data)
                ex_data = extract_mol(str(ex_file), i)
                ex_data_list.append(ex_data)

                # y = target[i].unsqueeze(0)
                # data_list.append(Data(g_data=g_data, ex_data=ex_data, y=y, idx=i))
            
            torch.save(self.collate(g_data_list), os.path.dirname(self.processed_paths[0]) + '/train_g.pt')
            torch.save(self.collate(ex_data_list), os.path.dirname(self.processed_paths[0]) + '/train_ex.pt')