import os
import argparse
import numpy as np

def read_colmap_depth(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    depth_map =  np.transpose(array, (1, 0, 2)).squeeze()
    min_depth, max_depth = np.percentile(
        depth_map, [1, 95])
    depth_map[depth_map < min_depth] = min_depth
    depth_map[depth_map > max_depth] = max_depth

    return depth_map

def read_colmap_normal(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    normal_map =  np.transpose(array, (1, 0, 2)).squeeze()

    return normal_map

def convert_depth_map_to_npy(depth_dir, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for fn in os.listdir(depth_dir):
        if not fn.endswith('.geometric.bin'):
            continue
        fpath = os.path.join(depth_dir, fn)
        depth_map = read_colmap_depth(fpath)
        save_path = os.path.join(save_dir, fn.replace('.bin', '.npy'))
        np.save(save_path, depth_map)

def convert_normal_map_to_npy(normal_dir, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for fn in os.listdir(normal_dir):
        if not fn.endswith('.geometric.bin'):
            continue
        fpath = os.path.join(normal_dir, fn)
        depth_map = read_colmap_normal(fpath)
        save_path = os.path.join(save_dir, fn.replace('.bin', '.npy'))
        np.save(save_path, depth_map)


parser = argparse.ArgumentParser(description='colmap_kitti')
parser.add_argument('--project_path', type=str, default='')
parser.add_argument('--save_dir', type=str, default='')

if __name__ == '__main__':
    args = parser.parse_args()
    project_path = args.project_path
    save_dir = args.save_dir
    depth_dir = os.path.join(project_path, 'dense/stereo/depth_maps')
    normal_dir = os.path.join(project_path, 'dense/stereo/normal_maps')
    depth_save_dir = os.path.join(save_dir, 'depth')
    normal_save_dir = os.path.join(save_dir, 'normal')
    convert_depth_map_to_npy(depth_dir, depth_save_dir)
    convert_normal_map_to_npy(normal_dir, normal_save_dir)