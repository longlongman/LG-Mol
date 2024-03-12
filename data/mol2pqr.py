import os
import ray
import pickle
import numpy as np
from tqdm import tqdm

import lmdb

def mkdir(directory_path):
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            print(f"目录 '{directory_path}' 创建成功")
        except OSError as e:
            print(f"创建目录 '{dWirectory_path}' 失败：{e}")

def count_lmdb_data(lmdb_path):
    # 检查LMDB文件是否存在
    if not os.path.exists(lmdb_path):
        print(f"LMDB file not found at {lmdb_path}")
        return -1

    # 打开LMDB文件
    env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
    txn = env.begin()
    # datapoint_pickled = env.begin().get(f"{0}".encode("ascii"))
    # data = pickle.loads(datapoint_pickled)
    # print(data)
    # 获取数据库的统计信息
    stats = txn.stat()
    num_entries = stats['entries']

    env.close()

    return num_entries



import json
with open('info.json') as f:
    config = json.load(f)

dataset_name = config['dataset_name']
split_name = config['split_name']
start_index = config['start_index'][split_name]
end_index = config['end_index'][split_name]
print(dataset_name)
print(split_name)
print(start_index)
print(end_index)

input_path = f"/sharefs/longsiyu/projects/shape4classify/data_mol/molecular_property_prediction/{dataset_name}/{split_name}.lmdb"
print(input_path)
num_data = count_lmdb_data(input_path)
print(f'Total number of entries in LMDB: {num_data}')
split_name = dataset_name + "_" + split_name
pqr_save_dir = f"/sharefs/longsiyu/projects/shape4classify/data_mol/convert_data/{split_name}/pqr/"
log_save_dir = f'/sharefs/longsiyu/projects/shape4classify/data_mol/convert_data/{split_name}/log'
mkdir(pqr_save_dir)
mkdir(log_save_dir)

ray.init(num_cpus=32, _temp_dir='/sharefs/longsiyu/projects/ray_tmp')

@ray.remote(num_cpus=1)
def mol2pqr(mol, i):
    mol = AllChem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol, numConfs=1, randomSeed=42)
    try:
        AllChem.MMFFOptimizeMoleculeConfs(mol)
    except Exception as e:
        print(f"{e}")
    import sys
    sys.path.insert(0, '/sharefs/longsiyu/projects/shape2mol/my_io')
    from mol_prepare import MolPreparator
    from antechamber import Antechamber
    from pqr import raw_info_to_pqr_block

    outfile = open(os.path.join(pqr_save_dir, str(i)+'.pqr'), 'w')
    try:
        preparator = MolPreparator(ionize=True, pH=7.4, align=False, add_hydrogens=True)
        antechamber = Antechamber()

        prepared_mol = preparator(mol)
        charges, radii = antechamber.get_charges_and_radii(prepared_mol)
        
        coords = []
        elements = []
        conf = prepared_mol.GetConformer()
        for atom_index, atom in enumerate(prepared_mol.GetAtoms()):
            positions = conf.GetAtomPosition(atom_index)
            positions = np.array([positions.x, positions.y, positions.z])
            elements.append(atom.GetSymbol())
            coords.append(positions)
        coords = np.stack(coords)
        pqr_block = raw_info_to_pqr_block(coords, elements, charges, radii)
        outfile.write(pqr_block)
        return "success"
    
    except Exception as e:
        if os.path.exists(os.path.join(pqr_save_dir, str(i)+'.pqr')):
            os.remove(os.path.join(pqr_save_dir, str(i)+'.pqr'))
        with open(os.path.join(log_save_dir, str(i)+'.txt'), 'w') as f:
            f.write(str(i)+"th molecule fail\n")
            f.write(e.__class__.__name__+"\n")
            f.write(f"{e}")
        return str(i)+"th molecule fail"






import lmdb
import os
import pickle
from functools import lru_cache

class LMDBDataset:
    def __init__(self, db_path):
        self.db_path = db_path
        assert os.path.isfile(self.db_path), "{} not found".format(self.db_path)
        env = self.connect_db(self.db_path)
        with env.begin() as txn:
            self._keys = list(txn.cursor().iternext(values=False))

    def connect_db(self, lmdb_path, save_to_self=False):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        if not save_to_self:
            return env
        else:
            self.env = env

    def __len__(self):
        return len(self._keys)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if not hasattr(self, "env"):
            self.connect_db(self.db_path, save_to_self=True)
        datapoint_pickled = self.env.begin().get(f"{idx}".encode("ascii"))
        data = pickle.loads(datapoint_pickled)
        return data


dataset = LMDBDataset(input_path)

from rdkit import Chem
from rdkit.Chem import AllChem


def coords_to_str(coords):
    return '\n'.join([' '.join(map(str, coord)) for coord in coords])

def to_iterator(obj_ids):
    while obj_ids:
        done, obj_ids = ray.wait(obj_ids)
        yield ray.get(done)

futures = [mol2pqr.remote(Chem.MolFromSmiles(dataset.__getitem__(i)['smi']), i) for i in range(start_index, end_index)]
t = tqdm(total=len(futures))
for _ in to_iterator(futures):
    t.update(len(_))

for i in range(dataset.__len__()):
    break
    if i > -1000:
        continue
    # molecule_info = dataset.__getitem__(i)
    # mol = Chem.MolFromSmiles(molecule_info['smi'])
    # print(molecule_info['smi'])
    # print(mol.GetNumAtoms())
    # print(molecule_info)
    # if mol is not None:
    #     # 添加3D坐标信息
    #     conf = Chem.Conformer(mol.GetNumAtoms())
    #     for i, coord in enumerate(molecule_info['coordinates'][0]):
    #         conf.SetAtomPosition(i, tuple(coord))
    #     mol.AddConformer(conf)
    # print(mol)
    # print(mol_block)
    smi = dataset.__getitem__(i)['smi']
    m=Chem.MolFromSmiles(smi)
    AllChem.EmbedMultipleConfs(m, numConfs=1, randomSeed=42)
    mol2pqr(m, i)
    
    
