# This script evaluates the already saved outputs of the image
# evaluation network. It loads the outputs of the network as well
# the extracted descriptors extracted on each image in the original
# dataset as well as the ground truth epipolar error corresponding
# to each descriptor

import sys, os
sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

import torch
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pickle
import cv2
import time
import json
import copy
import argparse
from tqdm import tqdm
from scipy import ndimage
from skimage import io, transform
from torchvision import transforms, utils
from PIL import Image
import matplotlib 
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from math import floor
from collections import OrderedDict
from data_loader.load_images_pp import ImageQualityDatasetPP
from networks.ENet.ENet import ENet

def generate_colors(color_num):
    start = 0.0
    stop = 1.0
    cm_subsection = np.linspace(start, stop, color_num) 

    colors = [ cm.viridis(x) for x in cm_subsection ]
    return colors

# Helper function to sort the descriptros based on the predicted image quality
# scores and calculate the mean error for different percentages of the data
# being retained. The purpose is to validate the correctness of the perception
# model in sorting descriptors based on their quality
def prepare_retained_data_plot(epipolar_err, 
                               predicted_desc_cost,
                               desc_response,
                               quantization_num):
  # IF set, random ordering is used as the baseline instead of sorting based on
  # the keypoint response values
  USE_RANDOM_ORDERING_BASELINE = True
  RANDOM_SAMPLING_ITER = 100 # 1000

  # Sort in ascending order
  sorted_indices_pred = np.argsort(predicted_desc_cost)
  sorted_indices_desc_response = np.argsort(-desc_response)
  sorted_indices_gt = np.argsort(epipolar_err)

  if USE_RANDOM_ORDERING_BASELINE:
    random_order = np.zeros((RANDOM_SAMPLING_ITER, epipolar_err.size),
                             dtype=int)
    
    for j in range(RANDOM_SAMPLING_ITER):
      random_order_instance = np.array(range(epipolar_err.size), dtype=int)
      np.random.shuffle(random_order_instance)
      random_order[j, :] = random_order_instance
  
  # Calculate the mean error over 1- the ground truth data, 2-the predicted
  # order of descriptors for <quantization_num>, and 3-the descriptors 
  # ordered based on their response score(as a baseline)
  # for different levels of retained data. 
  data_num = np.size(epipolar_err)
  segment_len = floor(data_num / quantization_num)
  retained_data_ratio_list = [0]
  introspection_model_mean_err_list  = [0]
  baseline_mean_err_list  = [0]
  ideal_mean_err_list  = [0]
  print('Preparing the plot data.')
  start_idx = 0
  for i in tqdm(range(quantization_num)):
    end_idx = int((i + 1) * segment_len - 1)
    
    if (i == (quantization_num - 1)):
      end_idx = data_num - 1
      
    retained_data_ratio = float(end_idx + 1) / float(data_num)
    ideal_mean_err_new_slice = np.mean(
                        epipolar_err[
                          sorted_indices_gt[start_idx:end_idx].tolist()])
    ideal_mean_err = (ideal_mean_err_new_slice * (end_idx - start_idx)
                    + ideal_mean_err_list[-1] * (start_idx + 1)) / end_idx

    introspection_model_mean_err_new_slice = np.mean(
                        epipolar_err[
                          sorted_indices_pred[start_idx:end_idx].tolist()])
    introspection_model_mean_err = (introspection_model_mean_err_new_slice *  
                                     (end_idx - start_idx)
                                  + introspection_model_mean_err_list[-1] * 
                                    (start_idx + 1)) / end_idx

    # baseline_mean_err = np.mean(
    #            epipolar_err[sorted_indices_desc_response[0:end_idx].tolist()])

    if USE_RANDOM_ORDERING_BASELINE:
      baseline_mean_err_new_slice = np.mean(
            epipolar_err[random_order[:, start_idx:end_idx]])
    else:
      baseline_mean_err_new_slice = np.mean(
               epipolar_err[
                 sorted_indices_desc_response[start_idx:end_idx].tolist()])

    baseline_mean_err = (baseline_mean_err_new_slice * (end_idx - start_idx)
                    + baseline_mean_err_list[-1] * (start_idx + 1)) / end_idx

    
    retained_data_ratio_list = retained_data_ratio_list + [retained_data_ratio]
    ideal_mean_err_list = ideal_mean_err_list + [ideal_mean_err]
    introspection_model_mean_err_list = (introspection_model_mean_err_list
                                      + [introspection_model_mean_err])
    baseline_mean_err_list = (baseline_mean_err_list
                                      + [baseline_mean_err])

    start_idx = end_idx
    
  return {'retained_data_ratio': retained_data_ratio_list,
          'ideal_mean_err': ideal_mean_err_list,
          'introspection_model_mean_err': introspection_model_mean_err_list,
          'baseline_mean_err': baseline_mean_err_list}


def main():
  root_dir ='PATH/TO/GENERATED_TRAINING_DATA'
  raw_img_dir = 'PATH/TO/RAW_IMGS'
  network_output_dir = ('PATH/TO/NETWORK/INFERENCE/OUTPUT')


  BATCH_SIZE = 1
  NUM_WORKERS = 12

   
  session_list_test = [1007, 1011, 1012, 1020, 1022, 1030, 1032,
                      1034, 1036,
                      2007, 2011, 2012, 2018, 2022, 2032,
                      2034, 2036,
                      3007, 3011, 3012, 3018, 3020, 3022, 3032,
                      3034, 3036,
                      4007, 4011, 4012, 4018, 4020, 4030, 4032,
                      4034, 4036]


  data_transform_input = transforms.Compose([
            transforms.ToTensor(),
        ])
      
  data_transform_target = transforms.Compose([
            transforms.ToTensor()
        ])
  
  dataset = ImageQualityDatasetPP(DATASET_DIR,
                                      RAW_IMG_DIR,
                                      session_list_test,
                                      NETWORK_OUTPUT_DIR,
                                      transform_input=data_transform_input,
                                      transform_target=data_transform_target,
                                      loaded_image_color=True,
                                      session_prefix_length=5, 
                                      raw_img_folder="img_left")
  
  data_loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=BATCH_SIZE,
                                            num_workers=NUM_WORKERS)
  
  eppipolar_err_total = np.array([], dtype=float)
  desc_response_total = np.array([], dtype=float)
  predicted_desc_cost_total = np.array([], dtype=float)
  predicted_desc_score_total = np.array([], dtype=float)
  
  CHECKPOINT_NUM = 500
  # Go through the data image by image
  for i, data in enumerate(data_loader, 0):
    desc_x = data["corr_desc_x"]
    desc_y = data["corr_desc_y"]
    desc_response = data["corr_desc_response"]
    eppipolar_err = data["corr_desc_epipolar_err"]
    
    # The higher the value, the higher the predicted potential for errors
    # happening in that area of the image
    predicted_img_qual = data["network_output"]
    predicted_img_qual = predicted_img_qual[0, :, :]
    
    # Smooth the predicted image_quality
    #kernel = np.ones((5,5), dtype=float) / 25.0
    #predicted_img_qual = ndimage.filters.convolve(predicted_img_qual, kernel)
   
    
    # The predicted costs for each extracted descriptor. Values are queried
    # from the the predicted image quality
    desc_x_pixel = np.rint(desc_x.numpy()).astype(int).tolist()
    desc_y_pixel = np.rint(desc_y.numpy()).astype(int).tolist()
    predicted_desc_cost = predicted_img_qual[0, desc_y_pixel, desc_x_pixel]
    
    # Convert the cost to score(the higher the better)
    #predicted_desc_score = 1 / (1 + predicted_desc_cost)
    #predicted_desc_score = 2 * predicted_desc_score - 1
    predicted_desc_score = np.power(10000, -predicted_desc_cost)
 
    eppipolar_err_total = np.append(eppipolar_err_total, eppipolar_err)
    desc_response_total = np.append(desc_response_total, desc_response)
    predicted_desc_cost_total = np.append(predicted_desc_cost_total,
                                          predicted_desc_cost)
    predicted_desc_score_total = np.append(predicted_desc_score_total,
                                          predicted_desc_score)
   
 
    if ((i % CHECKPOINT_NUM) == CHECKPOINT_NUM - 1):
      print(i + 1, '/ ', data_loader.__len__(), ' images processed.')
  
  # Calculate the weighted mean error and compare it for these two scenarios:
  # 1- All descriptros are weighted equally
  # 2- Descriptors are weighted according to the predicted image quality scores
  equally_weighted_mean_err = np.mean(eppipolar_err_total)
  weighted_mean_err = (np.sum(predicted_desc_score_total *
                                 eppipolar_err_total)
                       / np.sum(predicted_desc_score_total))
  
  
  print('equally_weighted_mean_err: ', equally_weighted_mean_err)
  print('weighted_mean_err: ', weighted_mean_err)
  
  # TODO: remove this temporary debugging
  # eppipolar_err_total[eppipolar_err_total > 10.0] = 10.0

  plt_data = prepare_retained_data_plot(eppipolar_err_total,
                                    predicted_desc_cost_total,
                                    desc_response_total,
                                    quantization_num = 100)
  

  line_colors = generate_colors(3)
  style_list = ['-', '--', ':']
  labels = ['ideal', 'introspection model', 'baseline']
  y_data = [plt_data['ideal_mean_err'], 
            plt_data['introspection_model_mean_err'],
            plt_data['baseline_mean_err']]
  x_data = plt_data['retained_data_ratio']
  fig = plt.figure()
  for i in range(len(labels)):
    plt.plot(x_data, 
             y_data[i],
            label=labels[i],
            color=line_colors[i],
            linewidth=6,
            linestyle=style_list[i])
  plt.xlabel("Retained Features Ratio")
  plt.ylabel("Mean Err")
  plt.legend()
  plt.tight_layout()

  output_dir = 'feature_qual_pred_results'
  output_file_name = 'feature_qual_pred_reproj__all_AirsimTest0_compWithRandom'
  vis_file_path = os.path.join(output_dir, output_file_name + '.pdf')
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  fig.savefig(vis_file_path,  dpi=150)

  data_file_path = os.path.join(output_dir, output_file_name)
  try:
    with open(data_file_path, 'wb') as file:
      pickle.dump(plt_data, file)
      file.close()
  except:
    print('Could not save plot data. ', data_file_path, 'could not be opened!')

  # plt.show()
  
  
if __name__=="__main__":
  main()
