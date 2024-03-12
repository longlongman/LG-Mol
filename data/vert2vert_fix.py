import os
import pymesh
import numpy as np
import pandas as pd
from numpy.linalg import norm
from argparse import ArgumentParser

def read_msms(vertfile, facefile):
    vertfile = open(vertfile)
    meshdata = (vertfile.read().rstrip()).split("\n")
    vertfile.close()

    # Read number of vertices.
    count = {}
    header = meshdata[2].split()
    count["vertices"] = int(header[0])
    ## Data Structures
    vertices = np.zeros((count["vertices"], 3))
    normalv = np.zeros((count["vertices"], 3))
    atom_id = [""] * count["vertices"]
    res_id = [""] * count["vertices"]
    for i in range(3, len(meshdata)):
        fields = meshdata[i].split()
        vi = i - 3
        vertices[vi][0] = float(fields[0])
        vertices[vi][1] = float(fields[1])
        vertices[vi][2] = float(fields[2])
        normalv[vi][0] = float(fields[3])
        normalv[vi][1] = float(fields[4])
        normalv[vi][2] = float(fields[5])
        atom_id[vi] = fields[7]
        res_id[vi] = fields[9]
        count["vertices"] -= 1
    
    # Read faces.
    facefile = open(facefile)
    meshdata = (facefile.read().rstrip()).split("\n")
    facefile.close()

    # Read number of faces
    header = meshdata[2].split()
    count["faces"] = int(header[0])
    faces = np.zeros((count["faces"], 3), dtype=int)
    normalf = np.zeros((count["faces"], 3))

    for i in range(3, len(meshdata)):
        fi = i - 3
        fields = meshdata[i].split()
        faces[fi][0] = int(fields[0]) - 1
        faces[fi][1] = int(fields[1]) - 1
        faces[fi][2] = int(fields[2]) - 1
        count["faces"] -= 1

    assert count["vertices"] == 0
    assert count["faces"] == 0

    return vertices, faces, normalv, res_id

def fix_mesh(mesh, resolution, detail="normal"):
    bbox_min, bbox_max = mesh.bbox;
    diag_len = norm(bbox_max - bbox_min);
    if detail == "normal":
        target_len = diag_len * 5e-3;
    elif detail == "high":
        target_len = diag_len * 2.5e-3;
    elif detail == "low":
        target_len = diag_len * 1e-2;
    
    target_len = resolution
    #print("Target resolution: {} mm".format(target_len));
    # PGC 2017: Remove duplicated vertices first
    mesh, _ = pymesh.remove_duplicated_vertices(mesh, 0.001)

    count = 0;
    # print("Removing degenerated triangles")
    mesh, __ = pymesh.remove_degenerated_triangles(mesh, 100);
    mesh, __ = pymesh.split_long_edges(mesh, target_len);
    num_vertices = mesh.num_vertices;
    while True:
        mesh, __ = pymesh.collapse_short_edges(mesh, 1e-6);
        mesh, __ = pymesh.collapse_short_edges(mesh, target_len,
                preserve_feature=True);
        mesh, __ = pymesh.remove_obtuse_triangles(mesh, 150.0, 100);
        if mesh.num_vertices == num_vertices:
            break;

        num_vertices = mesh.num_vertices;
        #print("#v: {}".format(num_vertices));
        count += 1;
        if count > 10: break;

    mesh = pymesh.resolve_self_intersection(mesh);
    mesh, __ = pymesh.remove_duplicated_faces(mesh);
    mesh = pymesh.compute_outer_hull(mesh);
    mesh, __ = pymesh.remove_duplicated_faces(mesh);
    mesh, __ = pymesh.remove_obtuse_triangles(mesh, 179.0, 5);
    mesh, __ = pymesh.remove_isolated_vertices(mesh);
    mesh, _ = pymesh.remove_duplicated_vertices(mesh, 0.001)
    
    return mesh

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

vert_dir = f'/projects/shape4classify/data_mol/convert_data/{split_name}/vert'
vert_fix_dir = f'/projects/shape4classify/data_mol/convert_data/{split_name}/vert_fix'

mkdir(vert_fix_dir)

for i in range(start_index, end_index):
    try:
        vert_path = f"{vert_dir}/{i}.vert"
        vert = vert_path
        face = vert[:-5] + ".face"
        vertices, faces, normals, names = read_msms(vert, face)
        mesh = pymesh.form_mesh(vertices, faces)
        regular_mesh = fix_mesh(mesh, 0.3)

        vertices = regular_mesh.vertices
        vert_fix = os.path.join(vert_fix_dir, vert.split('/')[-1][:-5]+'.csv')
        vert_fix_file = open(vert_fix, "w")
        for vert in vertices:
            vert_fix_file.write("{},{},{}\n".format(vert[0], vert[1], vert[2]))
        vert_fix_file.close()
    except Exception as e:
        error_message = f"!Pqr{i}: An error occurred: {str(e)}\n"
        with open("error_log3.txt", "a") as f:
            f.write(error_message)
            
        print(error_message)

# docker run -idt --rm --shm-size=100g -v /sharefs/longsiyu/projects:/projects -v /sharefs/qiukeyue/data:/data --name unimol_longsiyu --gpus all pymesh/pymesh