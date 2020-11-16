#!/bin/bash

SESSIONS="1 2 3 4 8 10 13 16 17 20 22 25 27 28 31 36 42"
START_FRAME="0" 
CONFIG_FILE_NAME="jackal_visual_odom_stereo_training.yaml"


CREATE_IVSLAM_DATASET="true"
INFERENCE_MODE="false"
INTROSPECTION_FUNCTION_ENABLED="false"
LOAD_IMG_QUAL_HEATMAPS_FROM_FILE="false"
IVSLAM_PROPAGATE_KEYPT_QUAL="false"
RUN_SINGLE_THREADED="false"
ENABLE_VIEWER="true"
USE_GPU="false"
RECTIFY_IMGS="true"

OPTIMIZER_RUN_EXTRA_ITERATIONS="true"
OPTIMIZER_POSE_OPT_ITER_COUNT="4" # def: 4
TRACKING_BA_RATE="1" # def: 1
MAP_DRAWER_VISUALIZE_GT_POSE="true"

SAVE_VISUALIZATIONS="false"
IVSLAM_ENABLED="true"
LOAD_REL_POSE_UNCERTAINTY="false"

CONFIG_FILE_DIR="../Examples/Stereo"
SOURCE_DATASET_BASE_DIR=\
"../../Jackal_Visual_Odom/"

INTROSPECTION_MODEL_PATH=""

TARGET_RESULT_BASE_DIR="results/"

TARGET_DATASET_BASE_DIR="generated_training_data/"

PREDICTED_IMAGE_QUAL_BASE_DIR=""


mkdir -p $TARGET_RESULT_BASE_DIR
mkdir -p $generated_training_data



SEQUENCE_PATH=$SOURCE_DATASET_BASE_DIR/"sequences"
GROUND_TRUTH_PATH=$SOURCE_DATASET_BASE_DIR/"poses"
LIDAR_POSE_UNC_PATH=$SOURCE_DATASET_BASE_DIR/"lidar_poses"

for session in $SESSIONS; do
  printf -v SESSION_NUM_STR '%05d' "$session"
  
  echo "*********************************"
  echo "Running on $SESSION_NUM_STR"
  echo "*********************************"
 
  
  PREDICTED_IMAGE_QUAL_FOLDER=\
$PREDICTED_IMAGE_QUAL_BASE_DIR/$SESSION_NUM_STR/

  REL_POSE_UNC_PATH=$LIDAR_POSE_UNC_PATH/$SESSION_NUM_STR"_predicted_unc.txt"

  
  ../Examples/Stereo/stereo_kitti \
  --vocab_path="../Vocabulary/ORBvoc.txt" \
  --settings_path=$CONFIG_FILE_DIR/$CONFIG_FILE_NAME \
  --data_path=$SEQUENCE_PATH/$SESSION_NUM_STR \
  --ground_truth_path=$GROUND_TRUTH_PATH/$SESSION_NUM_STR".txt" \
  --img_qual_path=$PREDICTED_IMAGE_QUAL_FOLDER \
  --introspection_model_path=$INTROSPECTION_MODEL_PATH \
  --out_visualization_path=$TARGET_RESULT_BASE_DIR/$SESSION_NUM_STR/ \
  --out_dataset_path=$TARGET_DATASET_BASE_DIR/$SESSION_NUM_STR/ \
  --rel_pose_uncertainty_path=$REL_POSE_UNC_PATH \
  --start_frame=$START_FRAME \
  --introspection_func_enabled=$INTROSPECTION_FUNCTION_ENABLED \
  --load_img_qual_heatmaps=$LOAD_IMG_QUAL_HEATMAPS_FROM_FILE \
  --load_rel_pose_uncertainty=$LOAD_REL_POSE_UNCERTAINTY \
  --run_single_threaded=$RUN_SINGLE_THREADED \
  --create_ivslam_dataset=$CREATE_IVSLAM_DATASET \
  --ivslam_enabled=$IVSLAM_ENABLED \
  --inference_mode=$INFERENCE_MODE \
  --save_visualizations=$SAVE_VISUALIZATIONS \
  --enable_viewer=$ENABLE_VIEWER \
  --use_gpu=$USE_GPU \
  --rectify_images=$RECTIFY_IMGS \
  --optimizer_run_extra_iter=$OPTIMIZER_RUN_EXTRA_ITERATIONS \
  --optimizer_pose_opt_iter_count=$OPTIMIZER_POSE_OPT_ITER_COUNT \
  --tracking_ba_rate=$TRACKING_BA_RATE \
  --map_drawer_visualize_gt_pose=$MAP_DRAWER_VISUALIZE_GT_POSE \
  --ivslam_propagate_keyptqual=$IVSLAM_PROPAGATE_KEYPT_QUAL 

  
done



