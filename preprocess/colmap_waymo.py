import sqlite3
import os
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R
import cv2

parser = argparse.ArgumentParser(description='colmap_waymo')
parser.add_argument('--project_path', type=str, default='')
parser.add_argument('--data_path', type=str, default='')

if __name__ == '__main__':
    args = parser.parse_args()
    data_path = args.data_path
    project_path = args.project_path

    c2ws = np.load(f'{data_path}/train_poses.npy')
    img_fns = sorted(os.listdir(f'{data_path}/train_imgs'))
    c2w_dict = {}
    for idx in range(len(img_fns)):
        img_name = img_fns[idx]
        c2w_dict[img_name] = c2ws[idx]

    sample_img = cv2.imread(f'{data_path}/train_imgs/{img_fns[0]}')
    img_h, img_w = sample_img.shape[:2]
    
    intrinsics = np.load(f'{data_path}/intrinsics.npy')

    db_fn = db_fn = f'{project_path}/database.db'
    conn = sqlite3.connect(db_fn)
    c = conn.cursor()
    c.execute('SELECT * FROM images')
    result = c.fetchall()

    out_fn = f'{project_path}/id_names.txt'
    with open(out_fn, 'w') as f:
        for i in result:
            f.write(str(i[0]) + ' ' + i[1] + '\n')
    f.close()

    path_idname = f'{project_path}/id_names.txt'
    
    f_id_name= open(path_idname, 'r')
    f_id_name_lines= f_id_name.readlines()

    images_save_dir = f'{project_path}/created/sparse/model'
    if not os.path.exists(images_save_dir):
        os.makedirs(images_save_dir)

    f_w= open(f'{project_path}/created/sparse/model/images.txt', 'w')
    id_names = []
    for l in f_id_name_lines:
        l = l.strip().split(' ')
        id_ = int(l[0])
        name = l[1]
        id_names.append([id_, name])

    for i in range(len(id_names)):
        id_ = id_names[i][0]
        name = id_names[i][1]
        
        transform = c2w_dict[name]
        transform = np.linalg.inv(transform)

        r = R.from_matrix(transform[:3,:3])
        rquat= r.as_quat()  # The returned value is in scalar-last (x, y, z, w) format.
        rquat[0], rquat[1], rquat[2], rquat[3] = rquat[3], rquat[0], rquat[1], rquat[2]
        out = np.concatenate((rquat, transform[:3, 3]), axis=0)

        id_ = id_names[i][0]
        name = id_names[i][1]
        img_idx = name.split('.')[0].split('_')[0]

        f_w.write(f'{id_} ')
        f_w.write(' '.join([str(a) for a in out.tolist()] ) )
        f_w.write(f' 1 {name}')
        f_w.write('\n\n')
    
    f_w.close()

    cameras_fn = os.path.join(images_save_dir, 'cameras.txt')
    with open(cameras_fn, 'w') as f:
        fx = intrinsics[0,0]
        fy = intrinsics[1,1]
        cx = intrinsics[0,2]
        cy = intrinsics[1,2]
        f.write(f'1 SIMPLE_PINHOLE {img_w} {img_h} {fx} {cx} {cy}')

    points3D_fn = os.path.join(images_save_dir, 'points3D.txt')
    os.system(f'touch {points3D_fn}')