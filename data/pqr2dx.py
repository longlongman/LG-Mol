import os
import pandas as pd
from subprocess import Popen, PIPE
from argparse import ArgumentParser
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
apbs_bin = '/sharefs/longsiyu/projects/APBS-3.4.1.Linux/bin/apbs'
apbs_input_dir = f'/sharefs/longsiyu/projects/shape4classify/data_mol/convert_data/{split_name}/pqr'

for i in range(start_index, end_index):
    try:
        pqr_path = f"{apbs_input_dir}/{i}.pqr"
        apbs_path = f"{apbs_input_dir}/{i}.apbs_input"
        if os.path.exists(apbs_path):
            continue
        print(f"Processing {i}")
        from pdb2pqr.io import dump_apbs


        dump_apbs(
            pqr_path,
            apbs_path,
        )

        apbs_input = apbs_path
        args = [
            apbs_bin,
            apbs_input
        ]
        p = Popen(args, stdout=PIPE, stderr=PIPE, cwd=apbs_input_dir)
        stdout, stderr = p.communicate()
    except Exception as e:
        error_message = f"!Pqr{i}: An error occurred: {str(e)}\n"
        with open("error_log.txt", "a") as f:
            f.write(error_message)
            
        print(error_message)