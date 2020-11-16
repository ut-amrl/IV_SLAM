#!/bin/bash

# Usage: run_stereo_airsim_batch.bash <run_with_GDB>[optional]

DEBUG_MODE=$1
if [ -z "$1" ]
then
  DEBUG_MODE=0
fi


SESSIONS="1001"
CONFIG_FILE_NAME="airsim_stereo_01_inference.yaml"
START_FRAME="0"


CREATE_IVSLAM_DATASET="false"
INFERENCE_MODE="true"
INTROSPECTION_FUNCTION_ENABLED="true"
LOAD_IMG_QUAL_HEATMAPS_FROM_FILE="false"
IVSLAM_PROPAGATE_KEYPT_QUAL="false"
LOAG_GT_DEPTH_IMGS="false"
RUN_SINGLE_THREADED="false"
ENABLE_VIEWER="true"
USE_GPU="true"

OPTIMIZER_RUN_EXTRA_ITERATIONS="true" # def: true
OPTIMIZER_POSE_OPT_ITER_COUNT="4" # def: 4
TRACKING_BA_RATE="1" # def: 1
MAP_DRAWER_VISUALIZE_GT_POSE="true"

SAVE_VISUALIZATIONS="true"
IVSLAM_ENABLED="true"
IVSLAM_ANG_VAR_INV="2.0e+1"
IVSLAM_TRANS_VAR_INV="2.0e+1"
IVSLAM_KEYPT_QUAL_CHI2_PROB_THRESH="0.999" # 0.999


CONFIG_FILE_DIR="../Examples/Stereo"
SOURCE_DATASET_BASE_DIR=""

INTROSPECTION_MODEL_PATH=""

TARGET_RESULT_BASE_DIR="results/"

TARGET_DATASET_BASE_DIR="generated_training_data/"

PREDICTED_IMAGE_QUAL_BASE_DIR=""


mkdir -p $TARGET_RESULT_BASE_DIR
mkdir -p $generated_training_data


SEQUENCE_PATH=$SOURCE_DATASET_BASE_DIR
GROUND_TRUTH_PATH=$SOURCE_DATASET_BASE_DIR

for session in $SESSIONS; do
  printf -v SESSION_NUM_STR '%05d' "$session"
  
  echo "*********************************"
  echo "Running on $SESSION_NUM_STR"
  echo "*********************************"
 
 PREDICTED_IMAGE_QUAL_FOLDER=\
$PREDICTED_IMAGE_QUAL_BASE_DIR/$SESSION_NUM_STR/ 

 GROUND_TRUTH_FILE=$GROUND_TRUTH_PATH/$SESSION_NUM_STR/"left_cam_pose_TUM.txt"
 
 if [ ! $DEBUG_MODE -eq 0 ] 
 then
    gdb --args \
  ../Examples/Stereo/stereo_airsim \
  --vocab_path="../Vocabulary/ORBvoc.txt" \
  --settings_path=$CONFIG_FILE_DIR/$CONFIG_FILE_NAME \
  --data_path=$SEQUENCE_PATH/$SESSION_NUM_STR \
  --ground_truth_path=$GROUND_TRUTH_FILE \
  --img_qual_path=$PREDICTED_IMAGE_QUAL_FOLDER \
  --introspection_model_path=$INTROSPECTION_MODEL_PATH \
  --out_visualization_path=$TARGET_RESULT_BASE_DIR/$SESSION_NUM_STR/ \
  --out_dataset_path=$TARGET_DATASET_BASE_DIR/$SESSION_NUM_STR/ \
  --start_frame=$START_FRAME \
  --introspection_func_enabled=$INTROSPECTION_FUNCTION_ENABLED \
  --load_img_qual_heatmaps=$LOAD_IMG_QUAL_HEATMAPS_FROM_FILE \
  --run_single_threaded=$RUN_SINGLE_THREADED \
  --create_ivslam_dataset=$CREATE_IVSLAM_DATASET \
  --ivslam_enabled=$IVSLAM_ENABLED \
  --inference_mode=$INFERENCE_MODE \
  --save_visualizations=$SAVE_VISUALIZATIONS \
  --enable_viewer=$ENABLE_VIEWER \
  --use_gpu=$USE_GPU \
  --optimizer_run_extra_iter=$OPTIMIZER_RUN_EXTRA_ITERATIONS \
  --optimizer_pose_opt_iter_count=$OPTIMIZER_POSE_OPT_ITER_COUNT \
  --tracking_ba_rate=$TRACKING_BA_RATE \
  --map_drawer_visualize_gt_pose=$MAP_DRAWER_VISUALIZE_GT_POSE \
  --load_gt_depth_imgs=$LOAG_GT_DEPTH_IMGS \
  --ivslam_propagate_keyptqual=$IVSLAM_PROPAGATE_KEYPT_QUAL \
  --ivslam_ref_pose_ang_var_inv=$IVSLAM_ANG_VAR_INV \
  --ivslam_ref_pose_trans_var_inv=$IVSLAM_TRANS_VAR_INV \
  --ivslam_keypt_qual_chi2_prob_thresh=$IVSLAM_KEYPT_QUAL_CHI2_PROB_THRESH 
  
  
 else
  ../Examples/Stereo/stereo_airsim \
  --vocab_path="../Vocabulary/ORBvoc.txt" \
  --settings_path=$CONFIG_FILE_DIR/$CONFIG_FILE_NAME \
  --data_path=$SEQUENCE_PATH/$SESSION_NUM_STR \
  --ground_truth_path=$GROUND_TRUTH_FILE \
  --img_qual_path=$PREDICTED_IMAGE_QUAL_FOLDER \
  --introspection_model_path=$INTROSPECTION_MODEL_PATH \
  --out_visualization_path=$TARGET_RESULT_BASE_DIR/$SESSION_NUM_STR/ \
  --out_dataset_path=$TARGET_DATASET_BASE_DIR/$SESSION_NUM_STR/ \
  --start_frame=$START_FRAME \
  --introspection_func_enabled=$INTROSPECTION_FUNCTION_ENABLED \
  --load_img_qual_heatmaps=$LOAD_IMG_QUAL_HEATMAPS_FROM_FILE \
  --run_single_threaded=$RUN_SINGLE_THREADED \
  --create_ivslam_dataset=$CREATE_IVSLAM_DATASET \
  --ivslam_enabled=$IVSLAM_ENABLED \
  --inference_mode=$INFERENCE_MODE \
  --save_visualizations=$SAVE_VISUALIZATIONS \
  --enable_viewer=$ENABLE_VIEWER \
  --use_gpu=$USE_GPU \
  --optimizer_run_extra_iter=$OPTIMIZER_RUN_EXTRA_ITERATIONS \
  --optimizer_pose_opt_iter_count=$OPTIMIZER_POSE_OPT_ITER_COUNT \
  --tracking_ba_rate=$TRACKING_BA_RATE \
  --map_drawer_visualize_gt_pose=$MAP_DRAWER_VISUALIZE_GT_POSE \
  --load_gt_depth_imgs=$LOAG_GT_DEPTH_IMGS \
  --ivslam_propagate_keyptqual=$IVSLAM_PROPAGATE_KEYPT_QUAL \
  --ivslam_ref_pose_ang_var_inv=$IVSLAM_ANG_VAR_INV \
  --ivslam_ref_pose_trans_var_inv=$IVSLAM_TRANS_VAR_INV \
  --ivslam_keypt_qual_chi2_prob_thresh=$IVSLAM_KEYPT_QUAL_CHI2_PROB_THRESH 
  
 fi
 

  
done

