import sys
source_file = sys.argv[1]  #'truss'  or 'shell'

s = source_file.lower()
if "truss" in s:
    s_type = "truss"
elif "shell" in s:
    s_type = "shell"
else:
    raise ValueError(
    f"Invalid source file name '{source_file}'. Expected a file name containing 'truss' or 'shell'."
)

gpu_id = '5'
target_dir = './Code/MorphologyLearning/data/cellculture/Samples_' + s_type + '/'
################################################

import os
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
from datetime import datetime
import numpy as np
import torch
import mcubes
import trimesh
from utils import *
from tqdm import tqdm
from pysdf import SDF
from scipy.ndimage import zoom
import h5py



num_samples_and_method = [(500000, 'uniformly'), (500000, 'near')]
n_dim = 64
os.makedirs(target_dir, exist_ok=True)
filepath = source_file

kk = 0
buffer_size = 3  
x = np.linspace(-0.5, 0.5, n_dim)
y = np.linspace(-0.5, 0.5, n_dim)
z = np.linspace(-0.5, 0.5, n_dim)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
xyz_cube = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
# load .h5
with h5py.File(filepath, 'r') as fp:
    filename = list(fp.keys())
    num_file = len(filename)
    print("Number of samples:", num_file)
    for i in tqdm(range(num_file)):
        t = i
        sdf_cube = -fp[filename[i]]['sdf']['matrix'][()]
        
        sdf_pad = np.pad(sdf_cube, pad_width=int(len(sdf_cube) / 4), mode='wrap')
        sdf_pad_hr = zoom(sdf_pad, zoom=(2.5, 2.5, 2.5), order=3)
        sdf_pad_hr = np.pad(sdf_pad_hr, pad_width=buffer_size, mode='constant',
                         constant_values=- 50)
        verts, faces = mcubes.marching_cubes(sdf_pad_hr, 0.0)
        target_face_count = 0.35 
        M = trimesh.Trimesh(verts, faces)
        M = M.simplify_quadric_decimation(1 - target_face_count)
        M.vertices = M.vertices - (len(sdf_pad_hr) - 1) / 2

        f = SDF(M.vertices, M.faces)
        scale = (len(sdf_pad_hr)-2*buffer_size-1) / (len(sdf_pad)-1) * (n_dim - 1)
        sdf_grid = f(xyz_cube * scale).reshape(n_dim, n_dim, n_dim) / scale
        verts_grid, faces_grid = mcubes.marching_cubes(sdf_grid, 0.0)
        verts_grid = verts_grid / (n_dim - 1) - 0.5
        mesh = obj2nvc(torch.tensor(verts_grid), torch.tensor(faces_grid, dtype=torch.int32))
        mesh_normals = face_normals(mesh)
        distrib = area_weighted_distribution(mesh, mesh_normals)
    
        xyz = sample_points(mesh, num_samples_and_method, mesh_normals, distrib)
        xyz = xyz.cpu().numpy()
        sdf = f(xyz * scale) / scale
        sdf = np.reshape(sdf, (-1, 1))
        xyz_sd = np.concatenate((xyz, sdf), axis=1)
        rand_idx = np.random.permutation(xyz_sd.shape[0])
        xyz_sd = xyz_sd[rand_idx]
    
        pos = []
        neg = []
        for j in range(xyz_sd.shape[0]):
            if xyz_sd[j, -1] >= 0:
                pos.append(xyz_sd[j, :])
            else:
                neg.append(xyz_sd[j, :])
        outfilename = target_dir + filename[i] + '.npz'
        np.savez(outfilename, pos=np.array(pos), neg=np.array(neg))

