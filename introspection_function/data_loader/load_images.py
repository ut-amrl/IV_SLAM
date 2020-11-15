import os
from os.path import isfile, join
import torch
import json
import csv
import re
import logging
from skimage import io, transform
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


# This dataset loads full images and the corresponding image 
# quality scores for the task of SLAM/VO.
class ImageQualityDataset(Dataset):
  def __init__(self, root_dir, raw_img_dir, session_list,
               loaded_image_color=False, 
               output_image_color=False,
               session_prefix_length=2,
               raw_img_folder="image_0",
               no_meta_data_available=False,
               load_only_with_labels=False,
               transform_input=None,
               transform_target=None,
               load_masks=False,
               regression_mode=True,
               binarize_target=False):
    DESCRIPTOR_FILE = "descriptors.csv"
    IMG_NAMES_FILE = "img_names.json"
    KEYPOINTS_FILE = "keypoints.json"
    
    self.binarize_target = binarize_target
    # The threshold that is used to binarize images with pixel values
    # between 0 and 255.
    self.binarize_thresh = 180.0
    self.root_dir = root_dir
    self.raw_img_dir = raw_img_dir
    self.loaded_image_color=loaded_image_color
    self.output_image_color=output_image_color
    self.transform_input = transform_input
    self.transform_target = transform_target
    self.session_name_format = ""
    self.raw_img_folder = raw_img_folder # "image_0", "img_left"
    # If set to true, it means that image file name is NOT available. 
    # The data loader should look at the raw_img_dir, and load all 
    # available images if training labels are not explicitly requested.
    # If load_only_with_labels==True, then the list of images should
    # be created from the folder of training label files.
    self.no_meta_data_available=no_meta_data_available

    # If set to true loads only datapoints for which a training label
    # is available. The list of images to be loaded is by default loaded
    # from the meta data files but if no_meta_data_available then the list
    # is created by actually reading through available label files.
    self.load_only_with_labels=load_only_with_labels
    self.load_masks=load_masks
    self.regression_mode=regression_mode
    
    if no_meta_data_available and (not load_only_with_labels):
      self.load_labels=False
    else:
      self.load_labels=True

    # The length of the session name prefix
    if (session_prefix_length == 2):
      self.session_name_format = "{0:02d}"
    elif (session_prefix_length == 5):
      self.session_name_format = "{0:05d}"
    else:
      logging.error("Session prefix length of %d is not supported.",
                    session_prefix_length)
      exit()
      
    if ((not loaded_image_color) and output_image_color):
      logging.error("Cannot convert from mono to color.")
      exit()
    
    self.img_names = []
    # List of corresponding session number for all image names
    self.session_num = np.array([], dtype=int)

    for session_num in session_list:
      session_folder = self.session_name_format.format(session_num)
      session_path = root_dir + '/' + session_folder + '/'

      if self.no_meta_data_available:
        if self.load_labels:
          img_dir = (self.root_dir + '/' + session_folder + \
                    '/bad_region_heatmap/')
        else:
          img_dir = (self.raw_img_dir + '/' + session_folder +
                   '/' + self.raw_img_folder)
        session_img_names = self.get_img_names(img_dir)
        self.img_names = self.img_names + session_img_names
      else:
        with open(session_path + IMG_NAMES_FILE, 'r') as file:
          img_names_file_obj = json.load(file)
        session_img_names = img_names_file_obj["img_name"]
        self.img_names = self.img_names + session_img_names

      session_num_list = (session_num * 
                  np.ones(len(session_img_names))).astype(int)
      self.session_num = np.concatenate((self.session_num, session_num_list))

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
    curr_img_mask_path = self.root_dir + '/' + session_folder + \
                    '/bad_region_heatmap_mask/' + self.img_names[idx].rstrip()

    # *****************
    # Load target image if available
    labels=torch.empty(0)
    if self.load_labels:
      score_img = io.imread(curr_img_score_path)
      score_img = score_img.reshape((score_img.shape[0], score_img.shape[1], 1))
      score_img_width = score_img.shape[1]
      score_img_height = score_img.shape[0]
      if self.binarize_target:
        bin_msk = score_img > self.binarize_thresh
        score_img[bin_msk] = 255
        score_img[np.logical_not(bin_msk)] = 0

      # Transform the images to PIL image
      score_img = transforms.ToPILImage()(score_img)

      if self.transform_target:
        score_img = self.transform_target(score_img)

      # Create binary class labels if not in regression mode
      if not self.regression_mode:
        if not torch.is_tensor(score_img):
          score_img = transforms.functional.to_tensor(score_img)
        labels = torch.zeros_like(score_img, dtype=torch.int64)
        bin_msk = score_img > (self.binarize_thresh / 255.0)
        labels[bin_msk] = 1
        

    # Load the target image masks
    if self.load_labels and self.load_masks:
      mask_img = io.imread(curr_img_mask_path)
      mask_img = mask_img.reshape((mask_img.shape[0], mask_img.shape[1], 1))
      mask_img_width = mask_img.shape[1]
      mask_img_height = mask_img.shape[0]

      if ((mask_img_width != score_img_width) or 
          (mask_img_height != score_img_height)):
        logging.error("Mask image and heatmap image sizes are not equal")
        exit()

      # Transform the images to PIL image
      mask_img = transforms.ToPILImage()(mask_img)

      if self.transform_target:
        mask_img = self.transform_target(mask_img)

      # In classification/segmentation mode, the mask is applied to the labels.
      # masked pixels will get a label class of -1
      if not self.regression_mode:
        if not torch.is_tensor(mask_img):
          mask_img = transforms.functional.to_tensor(mask_img)
        labels[mask_img < (self.binarize_thresh / 255.0)] = -1

      if labels.nelement() != 0:
        labels = torch.reshape(labels, (score_img.size()[1], 
                                        score_img.size()[2])) 

    # *****************
    # Load input image
    img = io.imread(curr_img_path)

    # Handle color and mono images accordingly
    raw_img_channels = 1
    if self.loaded_image_color:
      raw_img_channels = 3
      img = img[:, : , 0:raw_img_channels]

    img = img.reshape((img.shape[0], img.shape[1], raw_img_channels))

    # Crop the input image to the size of the target image (score image)
    # if the target images are available
    if self.load_labels:
      img = img[0:score_img_height, 0:score_img_width, :]

    # Transform the images to PIL image
    img = transforms.ToPILImage()(img)

    # Convert the input image to mono if required by the arguments
    if self.loaded_image_color != self.output_image_color:
      if not self.output_image_color:
        img = img.convert(mode="L")

    if self.transform_input:
      img = self.transform_input(img)

    if not self.load_labels:
      sample = {'img': img,
                'img_name': self.img_names[idx],
                'session': self.session_num[idx]}
    elif self.load_masks:
      sample = {'img': img,
                'score_img': score_img,
                'labels': labels,
                'mask_img' : mask_img,
                'img_name': self.img_names[idx],
                'session': self.session_num[idx]}
    else:
      sample = {'img': img,
                'score_img': score_img,
                'labels': labels, 
                'img_name': self.img_names[idx],
                'session': self.session_num[idx]}


    return sample
  
  def get_dataset_info(self):
    return {'size': len(self.img_names)}

  # Returns a list of name of all images in the input directory
  def get_img_names(self, dir):
    if os.path.isdir(dir):
      img_names = [f for f in os.listdir(dir) if isfile(join(dir, f))]
    else:
      logging.error("Directory does not exist: ", dir)
      exit()
    return img_names

if __name__ == "__main__":
  root_dir ='PATH/TO/GENERATED_TRAINING_DATA'
  raw_img_dir = 'PATH/TO/RAW_IMGS'
  sessions = [0]
  dataset = ImageQualityDataset(root_dir,
                                raw_img_dir,
                                sessions,
                                loaded_image_color=True,
                                output_image_color=False,
                                session_prefix_length=5,
                                raw_img_folder="image_0",
                                no_meta_data_available=True,
                                load_only_with_labels=False,
                                transform_input=None,
                                transform_target=None,
                                load_masks=True,
                                regression_mode=False)
  print("dataset size: ", dataset.__len__())                                
  sample = dataset[0]
  sample['img'].show()


  
  
  

