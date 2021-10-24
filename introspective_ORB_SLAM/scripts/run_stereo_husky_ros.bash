#!/bin/bash


SESSIONS="0"


# CONFIG_FILE_NAME="husky_visual_odom_stereo_inference.yaml"
# LEFT_CAM_TOPIC="/stereo/left/image_raw"
# RIGHT_CAM_TOPIC="/stereo/right/image_raw"
CONFIG_FILE_NAME="husky_visual_odom_stereo_down_scaled_inference.yaml"
LEFT_CAM_TOPIC="/left/image_proc_resize/image"
RIGHT_CAM_TOPIC="/right/image_proc_resize/image"

CREATE_IVSLAM_DATASET="false"
INFERENCE_MODE="true"
INTROSPECTION_FUNCTION_ENABLED="true"
LOAD_IMG_QUAL_HEATMAPS_FROM_FILE="false"
IVSLAM_PROPAGATE_KEYPT_QUAL="false"
RUN_SINGLE_THREADED="false"
ENABLE_VIEWER="true"
USE_GPU="true"
RECTIFY_IMGS="true"
UNDISTORT_IMGS="true"

OPTIMIZER_RUN_EXTRA_ITERATIONS="true"
OPTIMIZER_POSE_OPT_ITER_COUNT="4" # def: 4
TRACKING_BA_RATE="1" # def: 1
MAP_DRAWER_VISUALIZE_GT_POSE="false"

SAVE_VISUALIZATIONS="true"
IVSLAM_ENABLED="true"
LOAD_REL_POSE_UNCERTAINTY="false"

CONFIG_FILE_DIR="../Examples/Stereo"
SOURCE_DATASET_BASE_DIR=\
"../../Jackal_Visual_Odom/"

#****** AHG ******
# Applies to 1224X1024px images 
# INTROSPECTION_MODEL_PATH=\
# "/media/ssd2/Husky/IV_SLAM/MODEL/ahg_husky/exported_model_1224In_612Out/iv_ahg_husky_mobilenet_c1deepsup_light_1224In_612Out.pt"


# Applies to 512X512px images 
INTROSPECTION_MODEL_PATH=\
"/media/ssd2/Husky/IV_SLAM/MODEL/ahg_turn_test/exported_model_512In_612Out/iv_ahg_turn_test_mobilenet_c1deepsup_light_512In_612Out.pt"


#****** Speedway ******
# Applies to 1224X1024px images 
# INTROSPECTION_MODEL_PATH=\
# "/media/ssd2/Husky/IV_SLAM/MODEL/speedway_24th_cross/exported_model_1224In_612Out/iv_speedway_24th_cross_mobilenet_c1deepsup_light_1224In_612Out.pt"

# Applies to 512X512px images 
# INTROSPECTION_MODEL_PATH=\
# "/media/ssd2/Husky/IV_SLAM/MODEL/speedway_24th_cross/exported_model_512In_612Out/iv_speedway_24th_cross_mobilenet_c1deepsup_light_512In_612Out.pt"


TARGET_RESULT_BASE_DIR="results/"

TARGET_DATASET_BASE_DIR=""

PREDICTED_IMAGE_QUAL_BASE_DIR=""

mkdir -p $TARGET_RESULT_BASE_DIR

SEQUENCE_PATH=$SOURCE_DATASET_BASE_DIR/"sequences"
GROUND_TRUTH_PATH=$SOURCE_DATASET_BASE_DIR/"poses"
LIDAR_POSE_UNC_PATH=$SOURCE_DATASET_BASE_DIR/"lidar_poses"

START_FRAME="0"


PREDICTED_IMAGE_QUAL_FOLDER=\
$PREDICTED_IMAGE_QUAL_BASE_DIR/$SESSION_NUM_STR/

REL_POSE_UNC_PATH=$LIDAR_POSE_UNC_PATH/$SESSION_NUM_STR"_predicted_unc.txt"


../Examples/ROS/ORB_SLAM2/build/Stereo \
--vocab_path="../Vocabulary/ORBvoc.txt" \
--settings_path=$CONFIG_FILE_DIR/$CONFIG_FILE_NAME \
--data_path=$SEQUENCE_PATH/$SESSION_NUM_STR \
--minloglevel="0" \
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
--enable_viewer=$ENABLE_VIEWER \
--use_gpu=$USE_GPU \
--rectify_images=$RECTIFY_IMGS \
--undistort_images=$UNDISTORT_IMGS \
--optimizer_run_extra_iter=$OPTIMIZER_RUN_EXTRA_ITERATIONS \
--optimizer_pose_opt_iter_count=$OPTIMIZER_POSE_OPT_ITER_COUNT \
--tracking_ba_rate=$TRACKING_BA_RATE \
--map_drawer_visualize_gt_pose=$MAP_DRAWER_VISUALIZE_GT_POSE \
--ivslam_propagate_keyptqual=$IVSLAM_PROPAGATE_KEYPT_QUAL \
--left_cam_topic=$LEFT_CAM_TOPIC \
--right_cam_topic=$RIGHT_CAM_TOPIC 





