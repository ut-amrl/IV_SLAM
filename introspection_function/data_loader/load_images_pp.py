import os
import torch
import json
import csv
import re
from skimage import io, transform
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# The main purpose of this dataset is postprocessing and evaluation
# of the outputs of the image evaluation networks. It loads 
# the outpus of the image evaluation network as well as the 
# full images and the corresponding ground truth image 
# quality score for the task of SLAM/VO. It also loads the 
# corresponding extracted image descriptors for each input image


# This dataset loads full images and the corresponding image 
# quality score for the task of SLAM/VO.
class ImageQualityDatasetPP(Dataset):
  def __init__(self, root_dir, raw_img_dir, session_list,
          network_output_dir, # dir of output of image evaluation network
          loaded_image_color=False, 
          transform_input=None,
          transform_target=None,
          session_prefix_length=2,
          raw_img_folder="image_0"):
    DESCRIPTOR_FILE = "descriptors.csv"
    IMG_NAMES_FILE = "img_names.json"
    KEYPOINTS_FILE = "keypoints.json"

    self.root_dir = root_dir
    self.raw_img_dir = raw_img_dir
    self.network_output_dir = network_output_dir
    self.loaded_image_color=loaded_image_color
    self.transform_input = transform_input
    self.transform_target = transform_target
    self.session_name_format = ""
    self.raw_img_folder = raw_img_folder # "image_0", "img_left"
    # List of corresponding descriptor indices for each image (nested list)
    self.corr_desc_idx = []
    
    #self.descriptors = []
    self.img_names = []
    # List of corresponding session number for all image names
    self.session_num = np.array([], dtype=int)
    # Array of location of extracted descriptros
    self.corr_desc_x = np.array([], dtype=float)
    self.corr_desc_y = np.array([], dtype=float)
    self.corr_desc_response = np.array([], dtype=float)
    self.corr_desc_epipolar_err = np.array([], dtype=float)

    # The length of the session name prefix
    if (session_prefix_length == 2):
      self.session_name_format = "{0:02d}"
    elif (session_prefix_length == 5):
      self.session_name_format = "{0:05d}"
    else:
      logging.error("Session prefix length of %d is not supported.",
                    session_prefix_length)
      exit()

    for session_num in session_list:
      session_folder = self.session_name_format.format(session_num)
      session_path = root_dir + '/' + session_folder + '/'
      
      with open(session_path + KEYPOINTS_FILE, 'r') as file:
        keypoints_obj = json.load(file)

      with open(session_path + IMG_NAMES_FILE, 'r') as file:
        img_names_file_obj = json.load(file)
        
      curr_corr_desc_x = np.array(keypoints_obj["x_coord"])
      curr_corr_desc_y = np.array(keypoints_obj["y_coord"])
      curr_corr_desc_response = np.array(keypoints_obj["response"])
      curr_corr_desc_epipolar_err = np.array(keypoints_obj["epipolar_err"])
      self.corr_desc_x = np.append(self.corr_desc_x, curr_corr_desc_x)
      self.corr_desc_y = np.append(self.corr_desc_y, curr_corr_desc_y)
      self.corr_desc_response = np.append(self.corr_desc_response,
                                          curr_corr_desc_response)
      self.corr_desc_epipolar_err = np.append(self.corr_desc_epipolar_err, 
                                              curr_corr_desc_epipolar_err)
      
      self.img_names = (self.img_names + 
                        img_names_file_obj["img_name"])
      session_num_list = (session_num * 
                  np.ones(len(img_names_file_obj["img_name"]))).astype(int)
      self.session_num = np.concatenate((self.session_num, session_num_list))
      
      for desc_idx_list in img_names_file_obj["corresponding_keypt_id"]:
        self.corr_desc_idx = (self.corr_desc_idx +
                              [desc_idx_list["keypt_id"]])

        
     
  def __len__(self):
    return len(self.img_names)

  # Returns the raw image as well as the corresponding heatmap image that 
  # works as a metric of which parts of the input image are bad for VO/SLAM
  def __getitem__(self, idx):
    session_folder = self.session_name_format.format(self.session_num[idx])
    curr_img_path = self.raw_img_dir + '/' + session_folder + \
                  '/' +self.raw_img_folder + '/' + self.img_names[idx].rstrip()
    curr_img_score_path = self.root_dir + '/' + session_folder + \
                    '/bad_region_heatmap/' + self.img_names[idx].rstrip()
                  
    curr_network_out_path = self.network_output_dir + '/' + session_folder + \
                          '/' + self.img_names[idx].rstrip()[:-4] + '.jpg'

    img = io.imread(curr_img_path)

    # Handle color and mono images accordingly
    raw_img_channels = 1
    if self.loaded_image_color:
      raw_img_channels = 3
      img = img[:, : , 0:raw_img_channels]

    img = img.reshape((img.shape[0], img.shape[1], raw_img_channels))
    
    
    score_img = io.imread(curr_img_score_path)
    score_img = score_img.reshape((score_img.shape[0], score_img.shape[1], 1))
    
    # Output of the image evaluation network. A numpy array the same size as
    # the original image size.
    network_output = io.imread(curr_network_out_path)
    network_output = network_output[:, : , 0]
    network_output = network_output.reshape((img.shape[0], img.shape[1], 1))
    # network_output = np.load(curr_network_out_path)
    
    # Crop the input image to the size of the target image (score image)
    score_img_width = score_img.shape[1]
    score_img_height = score_img.shape[0]
    img = img[0:score_img_height, 0:score_img_width, :]
  
    
    # Transform the images to PIL image
    img = transforms.ToPILImage()(img)
    score_img = transforms.ToPILImage()(score_img)
    # network_output = transforms.ToPILImage()(network_output)
    
    if self.transform_input:
        img = self.transform_input(img)
    if self.transform_target:
        score_img = self.transform_target(score_img)
        network_output = self.transform_target(network_output)

    corr_desc_response = self.corr_desc_response[self.corr_desc_idx[idx]]
    corr_desc_epipolar_err = self.corr_desc_epipolar_err[
                                                      self.corr_desc_idx[idx]]
    
    sample = {'img': img,
              'score_img': score_img,
              'img_name': self.img_names[idx],
              'session': self.session_num[idx],
              'corr_desc_x': self.corr_desc_x[self.corr_desc_idx[idx]],
              'corr_desc_y': self.corr_desc_y[self.corr_desc_idx[idx]],
              'corr_desc_response': corr_desc_response,
              'corr_desc_epipolar_err': corr_desc_epipolar_err,
              'network_output': network_output}

    return sample
  
  def get_dataset_info(self):
    return {'size': len(self.img_names)}


# Helper function for drawing annotations on images
def annotate_image(input_img, keypts_x, keypts_y, err_list, pred_cost_img, 
                   response_list):
  MAX_ERR_THRESH = 10
  MAX_RESPONSE_THRESH = 256 # 256
  MAX_PRED_COST = 256 # 256 for pil img; 1.0 for tensor

  err_circle_rad = 10
  response_circle_rad = 5
  pred_cost_circle_rad = 15

  img = input_img.copy()
  img_draw = ImageDraw.Draw(img)

  # Pred cost image as tensor
  # pred_cost_img = pred_cost_img[0, :, :]
  
  # img = pred_cost_img.copy()
  # img = transforms.ToPILImage()(img).convert('RGB')
  # img_draw = ImageDraw.Draw(img)


  for i in range(keypts_x.size):
    x = keypts_x[i]
    y = keypts_y[i]
    
    
    pred_cost = pred_cost_img[int(np.rint(y)), int(np.rint(x)), 0] # def
    # pred_cost = pred_cost_img[(int(np.rint(y)), int(np.rint(x)))]

    # Assuming pred_cost_img is a tensor
    # pred_cost = pred_cost_img[int(np.rint(y)), int(np.rint(x))]

    err_color = cm.viridis(min(1.0, err_list[i] / MAX_ERR_THRESH))
    response_color = cm.viridis(min(1.0, 
                                    response_list[i] / MAX_RESPONSE_THRESH))
    pred_cost_color = cm.viridis(min(1.0, pred_cost / MAX_PRED_COST))


    err_color_int = tuple(int(err_color[i] * 255) 
                           for i in range(len(err_color)))
    response_color_int = tuple(int(response_color[i] * 255) 
                           for i in range(len(response_color)))
    pred_cost_color_int = tuple(int(pred_cost_color[i] * 255) 
                           for i in range(len(pred_cost_color)))

    bbox_pred = [(x - pred_cost_circle_rad, 
                  y - pred_cost_circle_rad),
                  (x + pred_cost_circle_rad,
                   y + pred_cost_circle_rad)]
    
    bbox_gt = [(x - err_circle_rad, 
                y - err_circle_rad),
                (x + err_circle_rad,
                 y + err_circle_rad)]

    bbox_response = [(x - response_circle_rad, 
                      y - response_circle_rad),
                      (x + response_circle_rad,
                      y + response_circle_rad)]

    img_draw.ellipse(bbox_pred, fill=pred_cost_color_int)
    img_draw.ellipse(bbox_gt, fill=err_color_int)
    img_draw.ellipse(bbox_response, fill=response_color_int)

  # out_img = Image.blend(img, input_img, 0.1)
  # out_img = Image.blend(img, pred_cost_img, 0.1)
  return img

if __name__ == "__main__":
  root_dir ='PATH/TO/GENERATED_TRAINING_DATA'
  raw_img_dir = 'PATH/TO/RAW_IMGS'
  network_output_dir = ('PATH/TO/NETWORK/INFERENCE/OUTPUT')

  data_transform_target = transforms.Compose([
            transforms.ToTensor()
        ])

  sessions = [1005]

  dataset = ImageQualityDatasetPP(root_dir, raw_img_dir, sessions,
                                  network_output_dir,
                                  loaded_image_color=True,
                                  session_prefix_length=5, 
                                  raw_img_folder="img_left")
  
  sample = dataset[0]
  print(sample['corr_desc_epipolar_err'].shape)

  ann_img =  annotate_image(input_img = sample['img'], 
                            keypts_x = sample['corr_desc_x'], 
                            keypts_y = sample['corr_desc_y'], 
                            err_list = sample['corr_desc_epipolar_err'], 
                            pred_cost_img = sample['network_output'], 
                            response_list = sample['corr_desc_response'])
  ann_img.show()

 
  
  
  


