# Please specify SEQUENCE, PROJECT_ROOT and ROOT_DIR
# SEQUENCE: pre-processed sequence id
# PROJECT_ROOT: root directory of the colmap project to save the results
# ROOT_DIR: root directory of the dataset (pre-processed)
SEQUENCE='seq_1'
PROJECT_ROOT=''
ROOT_DIR=''
PROJECT_PATH=${PROJECT_ROOT}/${SEQUENCE}

WORK_SPACE="$PWD"

if [ ! -d ${PROJECT_PATH} ]; then
    mkdir -p ${PROJECT_PATH}
fi
cd ${PROJECT_PATH}

colmap feature_extractor \
--ImageReader.camera_model SIMPLE_PINHOLE  \
--ImageReader.single_camera 1 \
--ImageReader.camera_params 2071.39328963,952.3805527,653.86698728 \
--database_path database.db \
--image_path ${ROOT_DIR}/${SEQUENCE}/train_imgs

python ${WORK_SPACE}/preprocess/colmap_waymo.py \
--project_path ${PROJECT_PATH} \
--data_path ${ROOT_DIR}/${SEQUENCE}

TRIANGULATED_DIR=${PROJECT_PATH}/triangulated/sparse/model
if [ ! -d ${TRIANGULATED_DIR} ]; then
    mkdir -p ${TRIANGULATED_DIR}
fi

colmap exhaustive_matcher \
--database_path database.db 
colmap point_triangulator \
--database_path database.db \
--image_path ${ROOT_DIR}/${SEQUENCE}/train_imgs \
--input_path created/sparse/model --output_path triangulated/sparse/model

colmap image_undistorter \
    --image_path ${ROOT_DIR}/${SEQUENCE}/train_imgs \
    --input_path triangulated/sparse/model \
    --output_path dense
colmap patch_match_stereo \
    --workspace_path dense
colmap stereo_fusion \
    --workspace_path dense \
    --output_path dense/fused.ply

python ${WORK_SPACE}/preprocess/colmap_bin2npy.py \
--project_path ${PROJECT_PATH} \
--save_dir ${ROOT_DIR}/${SEQUENCE}