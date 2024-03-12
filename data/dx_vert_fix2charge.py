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

multivalue_bin = '/sharefs/longsiyu/projects/shape2mol/multivalue'
vert_fix_dir = f'/sharefs/longsiyu/projects/shape4classify/data_mol/convert_data/{split_name}/vert_fix'
dx_dir = f'/sharefs/longsiyu/projects/shape4classify/data_mol/convert_data/{split_name}/pqr'
charge_save_dir = f'/sharefs/longsiyu/projects/shape4classify/data_mol/convert_data/{split_name}/charge'

mkdir(charge_save_dir)

for i in range(start_index, end_index):
    try:
        vert_fix_file = f"{i}.csv"
        vert_fix_path = f"{vert_fix_dir}/{i}.csv"
        dx_file = vert_fix_file[:-4]+'.pqr.dx'
        dx_path = os.path.join(dx_dir, dx_file)

        out_file = vert_fix_file[:-4] + '_out.csv'
        out_path = os.path.join(charge_save_dir, out_file)

        if not os.path.exists(dx_path):
            raise Exception('no corresponding .dx file')
        else:
            args = [
                multivalue_bin,
                vert_fix_path,
                dx_path,
                out_path
            ]
            p2 = Popen(args, stdout=PIPE, stderr=PIPE)
            stdout, stderr = p2.communicate()
    except Exception as e:
        error_message = f"!Pqr{i}: An error occurred: {str(e)}\n"
        with open("error_log3.txt", "a") as f:
            f.write(error_message)
            
        print(error_message)
