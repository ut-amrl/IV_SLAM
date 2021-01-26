#!/usr/bin/env python3

# Author Sadegh Rabiee

import os, argparse
from os.path import isfile, join
import shutil
import sys
import math
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
import subprocess
import numpy as np
from numpy import linalg as LA
# from drawing_tools import *


# After running evaluate_postproc.py on SLAM results, run this batch to
# generate the visualizations


EVAL_PATH_DEF = "/home/administrator/DATA/MODEL/ahg_husky/evaluate_model"


INPUT_RESULT_FILE_NAME = 'eval_results.json'
RESULT_FILES_PREFIXES=['rot_', 'trans_'] # rot_, trans_, pose_


# Sort the trajectories in the bar plot in descending order in terms of the
# failure count of method1.
SORT_TRAJ=True
SHOW_PLOTS=False
USE_TEX_FORMAT=False
MANUAL_Y_AXIS_LIMITS=False
USE_BROKEN_PLOTS=False

# If set to true, different error plots as well as the bar plot of failure 
# counts will be plotted seperately
SAVE_SEPARATE_GRAPHS=True 

# If set to true, data sessions that correspond to the same trajectories
# but in different environmental conditions will be unified together and their
# mean error will be reported
UNIFY_TRAJECTORIES=False

# Number of y ticks to be shown on the failure count figure
FAILURE_COUNT_Y_TICKS_NUM = 3 # 3 for jackal, 6 for airsim

modes=["ORB_SLAM", "IV_SLAM"]
session_idxs = [1,2,3]

def main():
  global METHOD_NAMES
  parser = argparse.ArgumentParser(description='After running '
                 'evaluate_postproc.py on SLAM results, run this script to '
                 'generate the visualizations')
  parser.add_argument("--eval_path",
              default=EVAL_PATH_DEF,
              help="path to the base directory of the first set of SLAM results to be evaluated",
              type=str)
  args = parser.parse_args()
  EVAL_PATH = args.eval_path
  OUTPUT_VIS_DIR = EVAL_PATH

  # Set matplotlib params
  # beautify_plots()
  if USE_TEX_FORMAT:
    plt.rc('text', usetex=True)
    plt.rcParams['text.latex.preamble'] = [r'\boldmath']
  #colors = generate_colors(3)
  #colors[1] = colors[1]
  
  # Overwrite figure size
  plt.rc('figure', figsize=[7, 4]) # For double column paper format
  # matplotlib.rc('figure', figsize=[7, 5]) # For single column paper format

  for RESULT_FILES_PREFIX in RESULT_FILES_PREFIXES:
    results = []

    for mode in modes:
      res_file = os.path.join(EVAL_PATH, mode + "_" + RESULT_FILES_PREFIX + INPUT_RESULT_FILE_NAME)
      with open(res_file) as json_file:
        data = json.load(json_file)
        results += [data]

    trajectories = results[0].keys()
    trajectory_names = list(trajectories)

    failure_counts = np.zeros((len(trajectories),2), dtype=int)
    error_vals =  np.zeros((len(trajectories),2), dtype=float)
    traversed_traj_lengths = np.zeros((len(trajectories), 2), dtype=float)
    full_traj_lengths = np.zeros((len(trajectories), 2), dtype=float)

    i = 0
    for traj in trajectories:
      for j in range(len(results)):
        failure_counts[i, j] = results[j][traj]['failure_count']
        error_vals[i,j] = results[j][traj]['rmse']
        traversed_traj_lengths[i,j] = results[j][traj]['traversed_length']
        full_traj_lengths[i,j] = results[j][traj]['total_length']
      i += 1

    failure_counts = failure_counts - 1

    # *********************
    # Unifying trajectoreis 
    if UNIFY_TRAJECTORIES:
      failure_counts_dicts = [dict(), dict()]
      error_vals_dicts = [dict(), dict()]
      traversed_traj_lengths_dicts = [dict(), dict()]
      full_traj_lengths_dicts = [dict(), dict()]

      i = 0
      for traj in trajectories:
        path_idx = int(traj) % 1000 
        # print('Trajectory: ', traj, ' -> ', path_idx)
        for j in range(len(results)):
          if path_idx in error_vals_dicts[j]:
            error_vals_dicts[j][path_idx].append(error_vals[i, j]) 
            failure_counts_dicts[j][path_idx].append(failure_counts[i, j]) 
            traversed_traj_lengths_dicts[j][path_idx].append(traversed_traj_lengths[i,j]) 
            full_traj_lengths_dicts[j][path_idx].append(full_traj_lengths[i,j]) 
          else:
            error_vals_dicts[j][path_idx] = [error_vals[i, j]]
            failure_counts_dicts[j][path_idx] = [failure_counts[i, j]]
            traversed_traj_lengths_dicts[j][path_idx] = [traversed_traj_lengths[i,j]]
            full_traj_lengths_dicts[j][path_idx] = [full_traj_lengths[i,j]]
        i += 1

      print(failure_counts_dicts)

      trajectories = []
      for unified_traj in error_vals_dicts[0]:
        trajectories.append(str(unified_traj))
      trajectory_names = trajectories

      failure_counts = np.zeros((len(trajectories),2), dtype=int)
      error_vals =  np.zeros((len(trajectories),2), dtype=float)
      traversed_traj_lengths = np.zeros((len(trajectories), 2), dtype=float)
      full_traj_lengths = np.zeros((len(trajectories), 2), dtype=float)

      # print('Unified Trajectories: ', trajectories)

      i = 0
      for traj in trajectories:
        traj_num = int(traj)
        for j in range(len(results)):
          failure_counts[i, j] = sum(failure_counts_dicts[j][traj_num])
          traversed_traj_lengths[i,j] = sum(traversed_traj_lengths_dicts[j][traj_num])
          full_traj_lengths[i,j] = sum(full_traj_lengths_dicts[j][traj_num])

          errors_np = np.array(error_vals_dicts[j][traj_num], dtype=float)
          error_vals[i,j] = np.sqrt(np.sum(np.square(errors_np)) 
                                    / errors_np.size)
        i += 1


    # *********************

    # Sort trajectories and all their corresponding statistics in descending order
    # of the failure counts of the EVAL_Method1
    if SORT_TRAJ:
      # sorted_idx = np.argsort(failure_counts[:,0], axis=0)
      sorted_idx = np.argsort(error_vals[:,0], axis=0)
      sorted_idx = sorted_idx[::-1]
      failure_counts = failure_counts[sorted_idx, :]
      error_vals = error_vals[sorted_idx, :]
      traversed_traj_lengths = traversed_traj_lengths[sorted_idx, :]
      full_traj_lengths = full_traj_lengths[sorted_idx, :]

      trajectory_names = [trajectory_names[i] for i in sorted_idx]

    X = np.arange(len(trajectories))
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.bar(X + 0.00, np.transpose(failure_counts[:, 0]), width=0.25)
    ax.bar(X + 0.25, np.transpose(failure_counts[:, 1]), width=0.25)

    if USE_TEX_FORMAT:
      trajectory_names = bold_text_list(trajectory_names)
      METHOD_NAMES = bold_text_list(METHOD_NAMES)

    ax.set_ylabel('Failure Count')
    # ax.set_title('Failure Count Comparison')
    plt.xticks(X, trajectory_names, rotation=45)
    ax.legend(labels=modes)
    plt.xlabel('Trajectories')
    plt.ylabel("Failure Count")



    X = np.arange(len(trajectories))
    ax = fig.add_subplot(212)
    ax.set_ylabel('RPE (RMSE)')
    ax.bar(X + 0.00, np.transpose(error_vals[:, 0]), width=0.25)
    ax.bar(X + 0.25, np.transpose(error_vals[:, 1]), width=0.25)

    error_unit = ""
    if RESULT_FILES_PREFIX == 'rot_':
      error_unit = '(deg)'
    elif RESULT_FILES_PREFIX == 'trans_':
      error_unit = '(m)'
    elif RESULT_FILES_PREFIX == 'pose_':
      error_unit = '(unit-less)'
    # ax.set_title('Failure Count Comparison')
    plt.xticks(X, trajectory_names, rotation=45)
    ax.legend(labels=modes)
    plt.xlabel('Trajectories')
    plt.ylabel("RPE " + error_unit)


    if SAVE_SEPARATE_GRAPHS:
      
      y_label = "RPE " + error_unit
      if error_unit == '(m)':
        y_label = 'Trans. Err. (m)'
      elif error_unit == '(deg)':
        y_label = 'Rot. Err. (deg)'
      fig3 = plt.figure()
      ax = fig3.add_subplot(111)
      ax.bar(X + 0.00, np.transpose(error_vals[:, 0]), width=0.25)
      ax.bar(X + 0.25, np.transpose(error_vals[:, 1]), width=0.25)
      plt.xticks(X, trajectory_names, rotation=45)
      ax.legend(labels=modes)
      plt.xlabel('Trajectories')
      plt.ylabel(y_label)
      plt.tight_layout()

      # The broken plot is hand crafted specifically for figure 5b, i.e. 
      # translation error on the Jackal data
      if USE_BROKEN_PLOTS and error_unit == '(m)':
        fig3, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw=   
                                {'height_ratios': [1, 2]})
        fig3.subplots_adjust(hspace=0.01)  # adjust space between axes

        # plot the same data on both axes
        axes = [ax1, ax2]
        for ax in axes:
          ax.bar(X + 0.00, np.transpose(error_vals[:, 0]), color=colors[0], width=0.25)
          ax.bar(X + 0.25, np.transpose(error_vals[:, 1]), color=colors[1], width=0.25)
          plt.xticks(X, trajectory_names, rotation=45)

        # zoom-in / limit the view to different portions of the data
        ax1.set_ylim(.9, 1.05)  # outliers only
        ax2.set_ylim(0, .25)  # most of the data

        # hide the spines between ax and ax2
        ax1.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax1.xaxis.set_visible(False)
        ax2.xaxis.tick_bottom()

        ax1.legend(labels=METHOD_NAMES)
        # plt.ylabel(y_label)

        d = .5  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                      linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
        ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

        # Overlay with single subplot
        # plt.tight_layout()
        # fig3.add_subplot(111, frame_on=False)
        # plt.tick_params(labelcolor="none", bottom=False, left=False)
        # plt.xlabel(bold_text('Trajectories'))
        # plt.ylabel(y_label)
        
        ax2.set_ylabel('.', color=(0, 0, 0, 0))
        fig3.text(0.04, 0.6, y_label, va='center', rotation='vertical')

        ax2.set_xlabel(bold_text('Trajectories'))
        plt.tight_layout()


      # Sort failure counts in descending order based on the values for 
      # eval1 data
      sorted_idx = np.argsort(failure_counts[:,0], axis=0)
      sorted_idx = sorted_idx[::-1]
      failure_counts_sorted = np.copy(failure_counts)
      failure_counts_sorted = failure_counts_sorted[sorted_idx, :]
      trajectory_names_sorted = [trajectory_names[i] for i in sorted_idx]

      fig4 = plt.figure()
      ax = fig4.add_subplot(111)
      ax.bar(X + 0.00, np.transpose(failure_counts_sorted[:, 0]), width=0.25)
      ax.bar(X + 0.25, np.transpose(failure_counts_sorted[:, 1]), width=0.25)
      plt.xticks(X, trajectory_names_sorted, rotation=45)
      ymin = np.min(failure_counts_sorted)
      ymax = math.ceil(np.max(failure_counts_sorted))+1
      if MANUAL_Y_AXIS_LIMITS:
        yint = range(0, ymax, math.floor((ymax - ymin) / 
                                          FAILURE_COUNT_Y_TICKS_NUM) )
        plt.yticks(yint)
      ax.legend(labels=modes)
      plt.xlabel('Trajectories')
      plt.ylabel("Failure Count")
      plt.tight_layout()

    # *********************************************
    # Plot the trajectory lengths
    fig2 = plt.figure()
    ax = fig2.add_subplot(111)
    ax.bar(X + 0.00, np.transpose(100.0 * traversed_traj_lengths[:, 0] / full_traj_lengths[:,0]),
                                  width=0.25)
    ax.bar(X + 0.25, np.transpose(100.0 * traversed_traj_lengths[:, 1] / full_traj_lengths[:,1]),
                                  width=0.25)

    ax.set_ylabel('Completion Percentage')
    plt.xticks(X, trajectory_names, rotation=45)
    ax.legend(labels=modes)
    plt.xlabel('Trajectories')
    plt.ylabel("Completion Percentage")



    # print(traversed_traj_lengths[1:3, 0])
    # print(full_traj_lengths[1:3,0])
    # print(traversed_traj_lengths[1:3, 0]/  full_traj_lengths[1:3,0])


    # # ************************************************
    # # Plot a filtered set of trajectories
    #
    # # Plot the info for only those trajectories where the ratio of completion
    # # for both algorithms is above a threshold
    # MIN_COMPLETION = 0.85
    # completion_ratios = traversed_traj_lengths / full_traj_lengths
    # pruned_trajs = completion_ratios > MIN_COMPLETION
    # traj_logical_idx = np.logical_and(pruned_trajs[:,0], pruned_trajs[:,1])
    # traj_idx = np.nonzero(traj_logical_idx)
    #
    # print("test")
    # print(traj_idx)
    # print(len(traj_idx))
    # print(traj_logical_idx.shape)
    # print(failure_counts[traj_idx, 0])
    #
    # X = np.arange(len(traj_idx))
    # fig = plt.figure()
    # ax = fig.add_subplot(211)
    # ax.bar(X + 0.00, np.transpose(failure_counts[traj_idx, 0]), color='b', width=0.25)
    # ax.bar(X + 0.25, np.transpose(failure_counts[traj_idx, 1]), color='g', width=0.25)
    #
    # ax.set_ylabel('Failure Count')
    # # ax.set_title('Failure Count Comparison')
    # ax.set_xticks(X, (trajectories[traj_idx]))
    # ax.legend(labels=['ORB-SLAM', 'IV-SLAM'])
    # plt.xlabel('Trajectories')
    # plt.ylabel("Failure Count")
    #
    #
    #
    # X = np.arange(len(traj_idx))
    # ax = fig.add_subplot(212)
    # ax.set_ylabel('RPE (RMSE)')
    # ax.bar(X + 0.00, np.transpose(error_vals[traj_idx, 0]), color='b', width=0.25)
    # ax.bar(X + 0.25, np.transpose(error_vals[traj_idx, 1]), color='g', width=0.25)
    #
    #
    # # ax.set_title('Failure Count Comparison')
    # ax.set_xticks(X, (trajectories))
    # # ax.set_yticks(np.arange(0, 20, 5))
    # ax.legend(labels=['ORB-SLAM', 'IV-SLAM'])
    # plt.xlabel('Trajectories')
    # plt.ylabel("RPE (RMSE)")

    # # ************************************************
    # # Print out mean performance statistics

    # Save figures to file
    vis_file_path1 = os.path.join(OUTPUT_VIS_DIR, 
                                  RESULT_FILES_PREFIX+'performance_comp.pdf')
    vis_file_path2 = os.path.join(OUTPUT_VIS_DIR, 
                                  RESULT_FILES_PREFIX+'completion_comp.pdf')
    fig.savefig(vis_file_path1,  dpi=150)
    fig2.savefig(vis_file_path2,  dpi=150)

    if SAVE_SEPARATE_GRAPHS:
      vis_file_path3 = os.path.join(OUTPUT_VIS_DIR, 
                            RESULT_FILES_PREFIX+'performance_comp_single.pdf')
      fig3.savefig(vis_file_path3,  dpi=150)

      vis_file_path4 = os.path.join(OUTPUT_VIS_DIR,'failure_count_comp.pdf')
      fig4.savefig(vis_file_path4,  dpi=150)


    # Mean number of failures per meter traversed (for each trajectory)
    failure_per_meter = failure_counts[:,:] / traversed_traj_lengths[:, :]
    mean_failure_per_meter = np.mean(failure_per_meter, axis=0);
    std_failure_per_meter = np.std(failure_per_meter, axis=0);

    # Mean RPE
    mean_err = np.mean(error_vals, axis=0);
    std_err = np.std(error_vals, axis=0);

    print("************************")
    print(EVAL_PATH.rsplit('/', 1)[-1])
    print(modes[0], ":")
    print("mean failure# per meter travelled: ", mean_failure_per_meter[0])
    print("std failure# per meter travelled: ", std_failure_per_meter[0])
    print("mean RPE ", error_unit, ": ", mean_err[0])
    print("std RPE: ", std_err[0])

    print("************************")
    print(EVAL_PATH.rsplit('/', 1)[-1])
    print(modes[1], ":")
    print("mean failure# per meter travelled: ", mean_failure_per_meter[1])
    print("std failure# per meter travelled: ", std_failure_per_meter[1])
    print("mean RPE ", error_unit, ": ", mean_err[1])
    print("std RPE: ", std_err[1])
  
  if SHOW_PLOTS:
    plt.show()

if __name__=="__main__":
  main()