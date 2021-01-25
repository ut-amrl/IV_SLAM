import os, argparse
from os.path import isfile, join
import shutil
import sys
import math
import json
from tqdm import tqdm
import csv
import subprocess
import numpy as np
from numpy import linalg as LA

# After running evaluate_batch.py on SLAM results, run this batch to aggregate
# the results that are already saved for each session separately. This allows you
# to compare different algorithms over all the dataset


# Ground truth path: Jackal_Visual_Odom Dataset
GT_PATH = "/media/ssd2/datasets/Jackal_Visual_Odom/sequences/"

# Ground truth path: Airsim Dataset
# GT_PATH = "/media/ssd2/datasets/AirSim_IVSLAM/cityenv_wb/"

# Ground truth path: KITTI Dataset
# GT_PATH = "/media/ssd2/public_datasets/KITTI/dataset/sequences"

# Ground truth path: EuRoC Dataset
# GT_PATH = "/media/ssd2/public_datasets/EuroC/"



EVAL_PATH_DEF = ("")

EVAL_TRAJ_REL_PATH="trajectory" # includes estimated pose of keyframes
# EVAL_TRAJ_REL_PATH="trajectory_sync" # includes synced subtrajectories for 
                              # multiple methods
# EVAL_TRAJ_REL_PATH="trajectory_all_frames" # includes estimated pose of all frames

INPUT_RESULTS_FOLDER_BASE="eval_results"
RESULT_FILES_PREFIXES=['rot_', 'trans_']

REFERENCE_TRAJ_REL_PATH="/left_cam_pose_TUM.txt" # kitti
# SESSION_NAME_FORMAT="{0:02d}"
SESSION_NAME_FORMAT="{0:05d}"

# REFERENCE_TRAJ_REL_PATH="mav0/state_groundtruth_estimate0/data.tum" # euroc
# SESSION_NAME_FORMAT="alphabetical"


# Jackal Visual Odom
session_idxs = [5, 6, 7, 11, 15, 18, 19, 23, 24, 29, 30, 33, 37, 40, 43]


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
  if traj_data.shape[0] > 0:
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
  parser = argparse.ArgumentParser(description='After running '
             'evaluate_batch.py on SLAM results, run this script to aggregate '
             'the results that are already saved for each session separately.' 
             ' This allows you to compare different algorithms over all the' 
             ' dataset')
  parser.add_argument("--data_path",
              default=EVAL_PATH_DEF,
              help="path to the base directory of SLAM results to be evaluated",
              type=str)
  args = parser.parse_args()
  EVAL_PATH = args.data_path

  print("EVAL_PATH: " + EVAL_PATH)
  for RESULT_FILES_PREFIX in RESULT_FILES_PREFIXES:
    INPUT_RESULTS_FOLDER = RESULT_FILES_PREFIX + INPUT_RESULTS_FOLDER_BASE
    print('Post processing the result category: ', RESULT_FILES_PREFIX)

    trajectories = dict()
    for session_id in tqdm(session_idxs):
      if SESSION_NAME_FORMAT=="alphabetical":
        session_id_str = session_id
      else:
        # session_id_str = "{0:05d}".format(session_id)
        session_id_str = SESSION_NAME_FORMAT.format(session_id)

      preprocess_result_file = os.path.join(EVAL_PATH, session_id_str,
                              INPUT_RESULTS_FOLDER,
                              RESULT_FILES_PREFIX +'res.csv')

      traj_dir = os.path.join(EVAL_PATH, session_id_str, EVAL_TRAJ_REL_PATH)

      reference_traj = GT_PATH + '/' + session_id_str + '/' + REFERENCE_TRAJ_REL_PATH
      full_traj_length = calc_trajectory_length(reference_traj)

      # Holds the names of sub trajectories that have been pruned in terms of
      # a minimum 1.0m length during the original evaluation
      sub_trajectories = []
      sub_traj_names = []
      sub_traj_ids = []
      rmse_idx = 1
      with open(preprocess_result_file) as csvfile:
        prep_result = csv.reader(csvfile, delimiter=',')
        count = 0
        completion = 0.0
        for row in prep_result:
          if count == 0:
            rmse_idx = row.index('rmse')

            count += 1
            continue

          sub_traj_names += [row[0]]
          sub_traj_ids += [row[0][-3:]]
          rmse = float(row[rmse_idx])

          sub_traj_file_path = os.path.join(traj_dir, sub_traj_names[-1]+'.txt')
          # sub_traj_length = calc_trajectory_length(sub_traj_file_path)
          sub_traj_length = get_gt_length(sub_traj_file_path, reference_traj)

          # print(100 * sub_traj_length/ full_traj_length, "\% ")
          completion += 100 * sub_traj_length/ full_traj_length

          sub_trajectory = {'rmse': rmse,
                            'length': sub_traj_length}
          sub_trajectories += [sub_trajectory]
          count += 1

        print("Completed: ", completion)
        trajectory = dict()
        # Calculate the mean error for the whole trajectory. It is calculated
        # as a weighted mean of the error of subtrajectories (weighted by length)
        rmse_total_squared = np.array([0.0])
        traversed_length = 0.0
        for i in range(len(sub_trajectories)):
          sub_trajectory = sub_trajectories[i]
          rmse = sub_trajectory['rmse']
          length = sub_trajectory['length']
          rmse_total_squared = rmse* rmse * length  + rmse_total_squared

          traversed_length += length
          trajectory[sub_traj_ids[i]] = sub_trajectory

        trajectory['rmse'] = math.sqrt(rmse_total_squared / traversed_length)
        trajectory['failure_count'] = len(sub_trajectories)
        trajectory['traversed_length'] = traversed_length
        trajectory['total_length'] = full_traj_length
        trajectories[session_id] = trajectory

    # Save the results for current SLAM algorithm to file
    result_file_path = os.path.join(EVAL_PATH,
                                    RESULT_FILES_PREFIX + 'eval_results.json')
    with open(result_file_path, 'w') as outfile:
      json.dump(trajectories, outfile)

if __name__=="__main__":
  main()