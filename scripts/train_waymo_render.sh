export TIME=$(date +"%Y-%m-%d_%H-%M-%S")
export DATASET="waymo"
export SEQUENCE=""
export DATA_ROOT=""
export VOXEL_SIZE=0.5
export LOG_DIR=""
export CKPT_DIR=""
export PRETRAINED_MESH_AE="pretrained/mesh_ae/mesh_ae.pth"

declare -A PRETRAIN_GEO_05
declare -A PRETRAIN_GEO_10

PRETRAIN_GEO_05['seq_1']=''
PRETRAIN_GEO_10['seq_1']=''

export CURR_PRETRAIN_GEO_05=${PRETRAIN_GEO_05[$SEQUENCE]}
export CURR_PRETRAIN_GEO_10=${PRETRAIN_GEO_10[$SEQUENCE]}

python train_render.py \
--dataset ${DATASET} \
--train_img_path ${DATA_ROOT}/${SEQUENCE}/train_imgs \
--test_img_path ${DATA_ROOT}/${SEQUENCE}/test_imgs \
--train_poses_path ${DATA_ROOT}/${SEQUENCE}/train_poses.npy \
--test_poses_path ${DATA_ROOT}/${SEQUENCE}/test_poses.npy \
--pts_file ${DATA_ROOT}/${SEQUENCE}/pcd.npz \
--calib_path ${DATA_ROOT}/${SEQUENCE}/intrinsics.npy \
--log_dir ${LOG_DIR}/render-${DATASET}-${SEQUENCE}-${TIME} \
--checkpoint_dir ${CKPT_DIR}/render-${DATASET}-${SEQUENCE}-${TIME} \
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
--render_multi_scale True 