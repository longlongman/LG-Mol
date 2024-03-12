import os
import ray
import pickle
from rdkit import Chem
from tqdm import tqdm

def mkdir(directory_path):
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            print(f"目录 '{directory_path}' 创建成功")
        except OSError as e:
            print(f"创建目录 '{dWirectory_path}' 失败：{e}")

import json
with open('info.json') as f:
    config = json.load(f)

dataset_name = config['dataset_name']
split_name = config['split_name']
start_index = config['start_index'][split_name]
end_index = config['end_index'][split_name]

input_path = f"/sharefs/longsiyu/projects/shape4classify/data_mol/molecular_property_prediction/{dataset_name}/{split_name}.lmdb"
split_name = dataset_name + "_" + split_name
xyzrn_save_dir = f"/sharefs/longsiyu/projects/shape4classify/data_mol/convert_data/{split_name}/xyzrn/"
log_save_dir = f'/sharefs/longsiyu/projects/shape4classify/data_mol/convert_data/{split_name}/log/xyzrn_log/'
mkdir(xyzrn_save_dir)
mkdir(log_save_dir)

radii = {}
radii["N"] = "1.540000"
radii["O"] = "1.400000"
radii["C"] = "1.740000"
radii["H"] = "1.200000"
radii["S"] = "1.800000"
radii["P"] = "1.800000"
radii["Z"] = "1.390000"
radii["X"] = "0.770000"
radii["Cl"]= "1.750000"
radii["F"] = "1.470000"
radii["Br"]= "1.850000"


ray.init(num_cpus=32, _temp_dir='/sharefs/longsiyu/projects/ray_tmp')


@ray.remote(num_cpus=1)
def mol2xyzrn(mol, i):
    mol = AllChem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol, numConfs=1, randomSeed=42)
    try:
        AllChem.MMFFOptimizeMoleculeConfs(mol)
    except Exception as e:
        print(f"{e}")
    import sys
    sys.path.insert(0, '/sharefs/longsiyu/projects/shape2mol/my_io')
    from mol_prepare import MolPreparator
    preparator = MolPreparator(ionize=True, pH=7.4, align=False, add_hydrogens=True)

    outfile = open(os.path.join(xyzrn_save_dir, str(i)+'.xyzrn'), "w")
    try:
        prepared_mol = preparator(mol)
        conf = prepared_mol.GetConformer()
        
        for a_index, atom in enumerate(prepared_mol.GetAtoms()):
            name = atom.GetSymbol()
            resname = 'UNK'
            chain = 'L'
            atomtype = name

            color = "Green"
            coords = None
            
            if atomtype in radii:
                if atomtype == "O":
                    color = "Red"
                if atomtype == "N":
                    color = "Blue"
                if atomtype == "H":
                    color = "Blue"
                pos = conf.GetAtomPosition(a_index)
                coords = "{:.06f} {:.06f} {:.06f}".format(
                    pos.x, pos.y, pos.z
                )
                insertion = "x"
                full_id = "{}_{:d}_{}_{}_{}_{}".format(
                    chain, a_index, insertion, resname, name, color
                )
            if coords is not None:
                outfile.write(coords + " " + radii[atomtype] + " 1 " + full_id + "\n")
        return "success"
    
    except Exception as e:
        if os.path.exists(os.path.join(xyzrn_save_dir, str(i)+'.xyzrn')):
            os.remove(os.path.join(xyzrn_save_dir, str(i)+'.xyzrn'))
        with open(os.path.join(log_save_dir, str(i)+'.txt'), 'w') as f:
            f.write(str(i)+"th molecule fail\n")
            f.write(e.__class__.__name__+"\n")
            f.write(f"{e}")
        return str(i)+"th molecule fail"


import os
# import ray
import pickle
import numpy as np
from tqdm import tqdm

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

def to_iterator(obj_ids):
    while obj_ids:
        done, obj_ids = ray.wait(obj_ids)
        yield ray.get(done)

futures = [mol2xyzrn.remote(Chem.MolFromSmiles(dataset.__getitem__(i)['smi']), i) for i in range(start_index, end_index)]
t = tqdm(total=len(futures))
for _ in to_iterator(futures):
    t.update(len(_))

# smi = dataset.__getitem__(1000)['smi']
# m=Chem.MolFromSmiles(smi)
# AllChem.EmbedMultipleConfs(m, numConfs=1, randomSeed=42)
# mol = m
# for i in range(1):
#     conf = mol.GetConformer(i)
#     print(f"Coordinates for Conformer {i + 1}:")
#     for j in range(mol.GetNumAtoms()):
#         pos = conf.GetAtomPosition(j)
#         print(f"Atom {j + 1}: x={pos.x:.3f}, y={pos.y:.3f}, z={pos.z:.3f}")

for i in range(dataset.__len__()):
    break
    if i >= 1000:
        break
    try:
        smi = dataset.__getitem__(i)['smi']
        m=Chem.MolFromSmiles(smi)
        AllChem.EmbedMultipleConfs(m, numConfs=1)
        mol2xyzrn(m, i)
    except Exception as e:
        print(f"{i} molecule failed!")
    
    
