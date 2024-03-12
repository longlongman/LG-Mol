import os
import lmdb
import tqdm
import pickle
import numpy as np
import pandas as pd
import json
with open('info.json') as f:
    config = json.load(f)

dataset_name = config['dataset_name']
split_name = config['split_name']
start_index = config['start_index'][split_name]
end_index = config['end_index'][split_name]

origin_path = f"/sharefs/longsiyu/projects/shape4classify/data_mol/molecular_property_prediction/{dataset_name}/{split_name}.lmdb"

charge_path = f'/sharefs/longsiyu/projects/shape4classify/data_mol/convert_data/{dataset_name + "_" + split_name}/charge.lmdb'
save_path = f'/sharefs/longsiyu/projects/shape4classify/data_mol/convert_data/{dataset_name + "_" + split_name}/{split_name}.lmdb'

db = lmdb.open(
    save_path,
    map_size=1024*(1024*1024*1024),
    create=True,
    subdir=False,
    readonly=False,
)

env_origin = lmdb.open(
    origin_path,
    subdir=False,
    readonly=True,
    lock=False,
    readahead=False,
    meminit=False,
    max_readers=256,
)

env_charge = lmdb.open(
    charge_path,
    subdir=False,
    readonly=True,
    lock=False,
    readahead=False,
    meminit=False,
    max_readers=256,
)

sum = 0
real = 0
for idx in tqdm.tqdm(range(start_index, end_index)):
    charge_pickled = env_charge.begin().get(f"{idx}".encode("ascii"))
    if charge_pickled is None:
        charge = np.zeros((800, 4), dtype=np.float32)
    else:
        charge = pickle.loads(charge_pickled)
        real+=1

    mol_pickled = env_origin.begin().get(f"{idx}".encode("ascii"))
    if mol_pickled is None:
        continue
    mol = pickle.loads(mol_pickled)
    mol['charge'] = charge

    curr_key = str(sum).encode("ascii")
    sum+=1
    serialized_mol = pickle.dumps(mol)
    with db.begin(write=True, buffers=True) as txn:
        txn.put(
            key=curr_key,
            value=serialized_mol
        )

db.close()
print(f"The number of samples: {real}")

# env_test = lmdb.open(
#     save_path,
#     subdir=False,
#     readonly=True,
#     lock=False,
#     readahead=False,
#     meminit=False,
#     max_readers=256,
# )

# charge_pickled = env_test.begin().get(f"{3}".encode("ascii"))
# charge = pickle.loads(charge_pickled)
# print(charge)