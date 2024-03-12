from argparse import ArgumentParser
import pandas as pd
from subprocess import Popen, PIPE
import os

def mkdir(directory_path):
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            print(f"目录 '{directory_path}' 创建成功")
        except OSError as e:
            print(f"创建目录 '{dWirectory_path}' 失败：{e}")


parser = ArgumentParser()
parser.add_argument('-i')
args = parser.parse_args()

import json
with open('info.json') as f:
    config = json.load(f)

dataset_name = config['dataset_name']
split_name = config['split_name']
start_index = config['start_index'][split_name]
end_index = config['end_index'][split_name]


split_name = dataset_name + "_" + split_name

msms_bin = "/sharefs/longsiyu/projects/msms/msms.x86_64Linux2.2.6.1"
vert_dir = f'/sharefs/longsiyu/projects/shape4classify/data_mol/convert_data/{split_name}/vert'
xyzrn_dir = f'/sharefs/longsiyu/projects/shape4classify/data_mol/convert_data/{split_name}/xyzrn'

mkdir(vert_dir)


for i in range(start_index, end_index):
    try:
        xyzrn_path = f"{xyzrn_dir}/{i}.xyzrn"

        file_base = f"{vert_dir}/{i}"
        FNULL = open(os.devnull, 'w')
        args = [
            msms_bin, 
            "-density", "5.0", 
            "-hdensity", "5.0",
            "-probe", "1.5", 
            "-if",xyzrn_path,
            "-of",file_base, 
            "-af", file_base
        ]
        p2 = Popen(args, stdout=PIPE, stderr=PIPE)
        stdout, stderr = p2.communicate()
    except Exception as e:
        error_message = f"!Pqr{i}: An error occurred: {str(e)}\n"
        with open("error_log2.txt", "a") as f:
            f.write(error_message)
            
        print(error_message)