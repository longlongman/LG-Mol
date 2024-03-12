import os
import lmdb
import tqdm
import pickle
import numpy as np
import pandas as pd

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

split_name = dataset_name + "_" + split_name

charge_path = f'/sharefs/longsiyu/projects/shape4classify/data_mol/convert_data/{split_name}/charge'
lmdb_save_path = f'/sharefs/longsiyu/projects/shape4classify/data_mol/convert_data/{split_name}/charge.lmdb'



db = lmdb.open(
    lmdb_save_path,
    map_size=1024*(1024*1024*1024),
    create=True,
    subdir=False,
    readonly=False,
)

for i in tqdm.tqdm(range(start_index, end_index)):
    try:
        charge_file_name = f"{i}_out.csv"
        charge_file_path = os.path.join(charge_path, charge_file_name)
        if not os.path.exists(charge_file_path):
            continue
        charge = pd.read_csv(charge_file_path, header=None)
        charge = charge.to_numpy()
        charge = np.float32(charge)

        curr_key = charge_file_name.split('_')[0].encode("ascii")
        curr_value = pickle.dumps(charge)

        with db.begin(write=True, buffers=True) as txn:
            txn.put(
                key=curr_key,
                value=curr_value
            )
    except Exception as e:
        print(f"Index {i} Error: {e}")

db.close()
