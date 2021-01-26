#!/usr/bin/env python3

# Author Sadegh Rabiee

import os, argparse
from os.path import isfile, join
import shutil
import sys
from tqdm import tqdm
import subprocess
import numpy as np
from numpy import linalg as LA

# This script goes through a list of given sessions in the dataset and
# runs evaluations for each. Aggragated results are saved for each session.

# Ground truth path: Jackal_Visual_Odom Dataset
GT_PATH = "/home/administrator/DATA/KITTI_FORMAT"

# Ground truth path: Airsim Dataset
# GT_PATH = "/media/ssd2/datasets/AirSim_IVSLAM/cityenv_wb/"

# Ground truth path: KITTI Dataset
# GT_PATH = "/media/ssd2/public_datasets/KITTI/dataset/sequences"

# Ground truth path: EuRoC Dataset
# GT_PATH = "/media/ssd2/public_datasets/EuroC/"


EVAL_PATH_DEF = "/home/administrator/DATA/MODEL/ahg_husky/evaluate_model"

# **********************
# ***** Config Files ***

# Monocular *******
# RESULTS_FOLDER_BASE="eval_results"
# CONFIG_FILES=['rpe_withSim3Align_config_rotOnly.json',
#               'rpe_withSim3Align_config_transOnly.json']
# POSE_RELATIONS=['rot_', 'trans_']  # rot_, trans_, pose_, 
#                                   # rot_ape_, trans_ape_ , pose_ape_
# EVALUATION_METHOD = 'evo_rpe' # 'evo_rpe', 'evo_ape'

# Stereo *******
RESULTS_FOLDER_BASE="eval_results"
CONFIG_FILES=['config/rpe_withSE3Align_config_rotOnly.json',
              'config/rpe_withSE3Align_config_transOnly.json']
POSE_RELATIONS=['rot_', 'trans_']  # rot_, trans_, pose_, 
                                  # rot_ape_, trans_ape_ , pose_ape_
EVALUATION_METHOD = 'evo_rpe' # 'evo_rpe', 'evo_ape'
DELTA = 2.0 # meters distance of consecutive evaluation points 
             # 2.0m for jackal
             # 20.0m for AirSim
             # 1.0m for EuRoC

MODE="ORB_SLAM"
EVAL_TRAJ_REL_PATH=MODE+"/trajectory" # includes estimated pose of keyframes
# EVAL_TRAJ_REL_PATH="trajectory_sync" # includes synced subtrajectories for 
                              # multiple methods
# EVAL_TRAJ_REL_PATH="trajectory_all_frames" # includes estimated pose of all frames

REFERENCE_TRAJ_REL_PATH="tum/tum.txt" # 
# SESSION_NAME_FORMAT="{0:02d}"
SESSION_NAME_FORMAT="{0:05d}"

# REFERENCE_TRAJ_REL_PATH="mav0/state_groundtruth_estimate0/data_LCamFrame.tum" # euroc
# SESSION_NAME_FORMAT="alphabetical"

# ahg_husky
session_idxs = [1,2,3]


# AirSim
# session_idxs = [1007, 1011, 1012, 1020, 1022, 1030, 1032,
#                  1034, 1036,
#                  2007, 2011, 2012, 2018, 2022, 2032,
#                  2034, 2036,
#                  3007, 3011, 3012, 3018, 3020, 3022, 3032,
#                  3034, 3036,
#                  4007, 4011, 4012, 4018, 4020, 4030, 4032,
#                  4034, 4036]

# Euroc
# session_idxs = ['MH1', 'MH2', 'MH3', 'MH4', 'MH5', 
#                 'V1_1', 'V1_2', 'V1_3', 
#                 'V2_1', 'V2_2', 'V2_3']



def calc_trajectory_length(traj_file_path):
  traj_data = np.loadtxt(open(traj_file_path, "rb"),
                         delimiter=" ", skiprows=0)
  length = 0
  if traj_data.shape[0] > 1 and traj_data.size > 8:
    for i in range(1 , traj_data.shape[0]):
      displacement = traj_data[i, 1:4] - traj_data[i-1, 1:4]
      length += LA.norm(displacement)
  return length

# Given the timestamps of the subdirectory, calculates the length of the
# corresponding subsection of the ground truth trajectory
def get_gt_length(query_traj_path, gt_traj_path):
  traj_query = np.loadtxt(open(query_traj_path, "rb"),
                         delimiter=" ", skiprows=0)
  traj_gt = np.loadtxt(open(gt_traj_path, "rb"),
                         delimiter=" ", skiprows=0)
  min_time_diff_thresh = 0.1

  # Catch empty trajectories
  if not (traj_query.shape[0] > 1 and traj_query.size > 8):
    return 0.0

  st_time = traj_query[0, 0]
  end_time = traj_query[-1, 0]

  st_idx_gt = np.argmin(np.abs(traj_gt[:,0] - st_time))
  end_idx_gt = np.argmin(np.abs(traj_gt[:, 0] - end_time))

  # Do not do the time stamp check for EuRoC since the gt
  # time stamps start earlier than that of the image frames
  if not SESSION_NAME_FORMAT=="alphabetical":
    if (np.abs(traj_gt[st_idx_gt,0] - st_time) > min_time_diff_thresh or
        np.abs(traj_gt[end_idx_gt,0] - end_time) > min_time_diff_thresh):
      print("Matching time stamps not found!")
      exit()

  length = 0.0
  for i in range(st_idx_gt + 1, end_idx_gt):
    displacement = traj_gt[i, 1:4] - traj_gt[i - 1, 1:4]
    length += LA.norm(displacement)

  return length

def main():
  parser = argparse.ArgumentParser(description='This script goes through '
              'a list of given sessions in the dataset and runs '
              'evaluations for each. Aggragated results are saved for '
              'each session.')
  parser.add_argument("--data_path",
              default=EVAL_PATH_DEF,
              help="path to the base directory of SLAM results to be evaluated",
              type=str)
  args = parser.parse_args()
  EVAL_PATH = args.data_path

  print("EVAL_PATH: " + EVAL_PATH)
  for k in range(len(POSE_RELATIONS)):
    RESULT_FILES_PREFIX = POSE_RELATIONS[k]
    CONFIG_FILE = CONFIG_FILES[k]
    RESULTS_FOLDER = MODE +"/" + RESULTS_FOLDER_BASE + "_" + RESULT_FILES_PREFIX
    print("Evaluating trajectories with the pose relation metric: ", 
          RESULT_FILES_PREFIX)

    for session_id in tqdm(session_idxs):
      if SESSION_NAME_FORMAT=="alphabetical":
        session_id_str = session_id
      else:
        # session_id_str = "{0:05d}".format(session_id)
        session_id_str = SESSION_NAME_FORMAT.format(session_id)

      reference_traj = GT_PATH + '/' + session_id_str + '/'+ REFERENCE_TRAJ_REL_PATH

      test_traj_dir = EVAL_PATH + '/' + session_id_str + '/' + EVAL_TRAJ_REL_PATH

      sub_traj_files = [f for f in os.listdir(test_traj_dir) if isfile(join(test_traj_dir, f))]

      eval_result_path = os.path.join(EVAL_PATH, session_id_str, RESULTS_FOLDER)

      # Prepare the directory for saving results. Remove old directories
      if not os.path.exists(eval_result_path):
        os.makedirs(eval_result_path)
      else:
        shutil.rmtree(eval_result_path)
        os.makedirs(eval_result_path)

      sub_traj_ids = []
      for sub_traj in sub_traj_files:
        sub_traj_path = test_traj_dir + '/' + sub_traj
        sub_traj_id = sub_traj[-7:-4]

        # Check if the sub trajectory is long enough
        # traj_length = calc_trajectory_length(sub_traj_path)
        traj_length = get_gt_length(sub_traj_path, reference_traj)

        if traj_length < 1.3:
          print('Trajectory ' , session_id , ":" ,sub_traj_id
                , ' was too short. Skipping ...')
          continue

        result_file_path = os.path.join(eval_result_path,
                                        RESULT_FILES_PREFIX+ sub_traj_id+'.zip')
        result_fig_path = os.path.join(eval_result_path,
                                        RESULT_FILES_PREFIX+ sub_traj_id + '.pdf')

        cmd = [EVALUATION_METHOD, 'tum']
        cmd += [reference_traj]
        cmd += [sub_traj_path]
        cmd += ['--delta=' + str(DELTA)]
        cmd += ['--config=' + CONFIG_FILE]
        cmd += ['--save_results', result_file_path]
        cmd += ['--save_plot', result_fig_path]

        try:
          subprocess.run(cmd, check=True,
                                stdout=subprocess.PIPE, universal_newlines=True)
        except subprocess.CalledProcessError as e:
          print(e.output)
          print("Skipping ", session_id_str, ":", sub_traj)
          continue
          
        sub_traj_ids += [sub_traj_id]


      result_table_path = os.path.join(eval_result_path,
                                      RESULT_FILES_PREFIX+ 'res' + '.csv')
      result_fig_path = os.path.join(eval_result_path,
                                      RESULT_FILES_PREFIX +'res' + '.pdf')
      res_zip_files = [os.path.join(eval_result_path,
                                    RESULT_FILES_PREFIX + idx + '.zip') for
                      idx in sub_traj_ids]

      cmd2 = ['evo_res']
      cmd2 += res_zip_files
      cmd2 += ['--save_table', result_table_path]
      cmd2 += ['--save_plot', result_fig_path]
      try:
        subprocess.run(cmd2, check=True,
                              stdout=subprocess.PIPE, universal_newlines=True)
      except subprocess.CalledProcessError as e:
        print(e.output)

if __name__=="__main__":
  main()