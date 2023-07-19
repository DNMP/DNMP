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

python ${WORK_SPACE}/preprocess/metashape_kitti.py \
--project_path ${PROJECT_PATH} \
--data_path ${ROOT_DIR} \
--image_path ${ROOT_DIR}/${SEQUENCE}/train_imgs