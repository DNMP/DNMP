export TIME=$(date +"%Y-%m-%d_%H-%M-%S")
export DATASET="kitti360"
export SEQUENCE="seq_3"
export DRIVE="2013_05_28_drive_0009_sync"
export DATA_ROOT=""
export SAVE_DIR=""
export PRETRAINED_MESH_AE="pretrained/mesh_ae/mesh_ae.pth"

declare -A PRETRAIN_GEO_05
declare -A PRETRAIN_GEO_10

PRETRAIN_GEO_05['seq_1']=''
PRETRAIN_GEO_10['seq_1']=''

export CURR_PRETRAIN_GEO_05=${PRETRAIN_GEO_05[$SEQUENCE]}
export CURR_PRETRAIN_GEO_10=${PRETRAIN_GEO_10[$SEQUENCE]}
export PRETRAINED_RENDER=''
export PRETRAINED_MIPNERF=''

python test_render.py \
--dataset ${DATASET} \
--train_img_path ${DATA_ROOT}/${SEQUENCE}/train_imgs \
--test_img_path ${DATA_ROOT}/${SEQUENCE}/test_imgs \
--train_poses_path ${DATA_ROOT}/data_poses/${DRIVE}/poses.txt \
--test_poses_path ${DATA_ROOT}/data_poses/${DRIVE}/poses.txt \
--pts_file ${DATA_ROOT}/${SEQUENCE}/pcd.npz \
--calib_path ${DATA_ROOT}/calibration/perspective.txt \
--cam_to_pose_path ${DATA_ROOT}/calibration/calib_cam_to_pose.txt \
--num_rays 16384 \
--chunk_size 16384 \
--print_freq 100 \
--val_freq 5000 \
--N_freqs_xyz 10 \
--N_freqs_dir 4 \
--logscale False \
--scale_factor 1. \
--max_iter 100000 \
--num_faces 4 \
--coarse_num_faces 2 \
--pretrained_mesh_ae ${PRETRAINED_MESH_AE} \
--mesh_ae_hidden_size 8 \
--near_plane 0.5 \
--far_plane 100. \
--center_point_fn ${DATA_ROOT}/${SEQUENCE}/center_point.npy \
--scene_scale 10. \
--use_disp False \
--voxel_size_list 0.50 1.0 \
--pretrained_geo_list ${CURR_PRETRAIN_GEO_05} ${CURR_PRETRAIN_GEO_10} \
--use_bkgd True \
--render_multi_scale True \
--bin_size 72 \
--max_faces_per_bin_factor 5 \
--save_dir ${SAVE_DIR} \
--pretrained_render ${PRETRAINED_RENDER} \
--pretrained_mipnerf ${PRETRAINED_MIPNERF}