import sys, os
sys.path.append(os.path.join(os.path.dirname(sys.path[0])))


import torch
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
import torch.optim as optim
import torch.nn as nn
import numpy as np
import time
import json
import copy
import argparse, os
from skimage import io, transform
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from math import floor
import re
import functools
from distutils.version import LooseVersion
from collections import OrderedDict
from data_loader.load_images import ImageQualityDataset

# Third party libs
from config import cfg
from networks.models import ModelBuilder, SegmentationModule
from lib.nn import convert_model, DataParallelWithCallback, patch_replication_callback
from lib.utils.utils import parse_devices
from lib.utils.utils import MaskedMSELoss

# This script is used for building and training segmentation architectures
# by putting together various encoder and decoder modules

def group_weight(module):
  group_decay = []
  group_no_decay = []
  for m in module.modules():
    if isinstance(m, nn.Linear):
      group_decay.append(m.weight)
      if m.bias is not None:
        group_no_decay.append(m.bias)
    elif isinstance(m, nn.modules.conv._ConvNd):
      group_decay.append(m.weight)
      if m.bias is not None:
        group_no_decay.append(m.bias)
    elif isinstance(m, nn.modules.batchnorm._BatchNorm):
      if m.weight is not None:
        group_no_decay.append(m.weight)
      if m.bias is not None:
        group_no_decay.append(m.bias)

  assert len(list(module.parameters())) == len(group_decay) + len(
    group_no_decay)
  groups = [dict(params=group_decay),
            dict(params=group_no_decay, weight_decay=.0)]
  return groups

def create_optimizers(modules, cfg):
  (net_encoder, net_decoder, crit) = modules
  optimizer_encoder = torch.optim.SGD(
    group_weight(net_encoder),
    lr=cfg.TRAIN.lr_encoder,
    momentum=cfg.TRAIN.beta1,
    weight_decay=cfg.TRAIN.weight_decay)
  optimizer_decoder = torch.optim.SGD(
    group_weight(net_decoder),
    lr=cfg.TRAIN.lr_decoder,
    momentum=cfg.TRAIN.beta1,
    weight_decay=cfg.TRAIN.weight_decay)
  return (optimizer_encoder, optimizer_decoder)

# Helper function for saving training results
def save_results(target_dir,
                 model_name,
                 last_model_state,
                 best_model_state,
                 train_history,
                 snapshot_idx):
  final_res_dir = target_dir + '/' + model_name + '/'
  snapshot_res_dir = target_dir + '/' + model_name + '/' + 'snapshots/'
  save_dir = final_res_dir
  if not os.path.exists(final_res_dir):
    os.makedirs(final_res_dir)
  if not os.path.exists(snapshot_res_dir):
    os.makedirs(snapshot_res_dir)

  (last_model_enc , last_model_dec) = last_model_state
  (best_model_enc, best_model_dec) = best_model_state
  # last_model_enc = getattr(last_model,'encoder')
  # last_model_dec = getattr(last_model,'decoder')
  # best_model_enc = getattr(best_model,'encoder')
  # best_model_dec = getattr(best_model,'decoder')

  suffix = ''
  if (snapshot_idx >= 0):
    suffix = '_' + str('%04d' % (snapshot_idx + 1))
    save_dir = snapshot_res_dir
  torch.save(last_model_enc,
              save_dir + model_name + '_encoder_last_model' + suffix + '.pth')
  torch.save(last_model_dec,
              save_dir + model_name + '_decoder_last_model' + suffix + '.pth')
  torch.save(best_model_enc,
              save_dir + model_name + '_encoder_best_model' + suffix + '.pth')
  torch.save(best_model_dec,
              save_dir + model_name + '_decoder_best_model' + suffix + '.pth')
  hist_file_path = (save_dir + model_name + '_training_history' +
                    suffix + '.json')
  with open(hist_file_path, 'w') as fp:
    json.dump(train_history, fp, indent=2)

def calculate_gpu_usage(gpus):
  total_usage_mb = 0.0
  for i in range(len(gpus)):
    total_usage_mb += float(torch.cuda.max_memory_allocated(gpus[i]) +
                        torch.cuda.max_memory_cached(gpus[i])) / 1000000.0
  return total_usage_mb


def main(cfg, args, gpus):
  USE_MULTI_GPU = True if len(gpus) > 1 else False
  NUM_WORKERS = cfg.TRAIN.workers
  if cfg.TRAIN.use_gpu:
    BATCH_SIZE = cfg.TRAIN.batch_size_per_gpu * len(gpus)
  else:
    BATCH_SIZE = cfg.TRAIN.batch_size


  train_set_dict = {
    "train_airsim_city_0":[
                           1005, 1006, 1009, 1010, 1013, 1014, 1015, 1016, 1021,
                           1023, 1025, 1027, 1028, 1029, 1031, 1035,
                           2005, 2006, 2009, 2010, 2013, 2014, 2015, 2016, 2021,
                           2023, 2025, 2027, 2028, 2029, 2031, 2033, 2035,
                           3005, 3006, 3009, 3010, 3013, 3014, 3015, 3016, 3021,
                           3023, 3025, 3027, 3028, 3029, 3031, 3033, 3035,
                           4005, 4006, 4009, 4010, 4013, 4014, 4015, 4021,
                           4023, 4025, 4027, 4028, 4029, 4031, 4033, 4035],
    "train_jackal_0": [
          1, 2, 3, 4, 8, 10, 13, 16, 17, 20, 22, 25, 27, 28, 31, 36, 42],
    "train_jackal_t": [37],
    "train_stereo_2020_12_21_run1_t": [1]
    }

  valid_set_dict = {
    "valid_airsim_city_0": [1008, 1016, 1017, 1024],
    "valid_jackal_0": [14],
    "valid_jackal_t": [37],
    "valid_stereo_2020_12_21_run1_t": [1],
  }


  session_list_train = [0, 1 , 2, 3, 4, 5]
  session_list_val = [6, 8]
  session_list_test = [7, 9, 10]


  if (cfg.DATASET.train_set is not None) and (cfg.DATASET.validation_set is not None):
    session_list_train = train_set_dict[cfg.DATASET.train_set]
    session_list_val = valid_set_dict[cfg.DATASET.validation_set]

  if cfg.TRAIN.use_gpu and torch.cuda.is_available():
    device = torch.device("cuda:"+str(gpus[0]))
    used_gpu_count = 1
    total_mem =(
          float(torch.cuda.get_device_properties(device).total_memory)
          / 1000000.0)
    gpu_name = torch.cuda.get_device_name(device)
    print("Using ", gpu_name, " with ", total_mem, " MB of memory.")
  else:
    device = torch.device("cpu")
    used_gpu_count = 0


  print("Output Model: ", cfg.MODEL.name)
  print("Train data: ", session_list_train)
  print("Validation data: ", session_list_val)
  print("Network base model is ", cfg.MODEL.arch_encoder  + '+' +
         cfg.MODEL.arch_decoder)
  print("Batch size: ", BATCH_SIZE)
  print("Workers num: ", NUM_WORKERS)
  print("device: ", device)
  phases = ['train', 'val']
  #data_set_portion_to_sample = {'train': 0.8, 'val': 0.2}
  data_set_portion_to_sample = {'train': 1.0, 'val': 1.0}

  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

  input_img_width = int(cfg.DATASET.img_width)
  input_img_height = int(cfg.DATASET.img_height)
  target_img_width = int(cfg.DATASET.img_width
                      / cfg.DATASET.target_downsampling_rate)
  target_img_height = int(cfg.DATASET.img_height
                      / cfg.DATASET.target_downsampling_rate)

  # Transform loaded images. If not using color images, it will copy the single
  # channel 3 times to keep the size of an RGB image.
  if cfg.DATASET.use_color_images:
    if cfg.DATASET.normalize_input :
      data_transform_input = transforms.Compose([
                transforms.Resize((input_img_height, input_img_width)),
                transforms.ToTensor(),
                normalize
            ])
    else:
      data_transform_input = transforms.Compose([
              transforms.Resize((input_img_height, input_img_width)),
              transforms.ToTensor(),
          ])
  else:
    if cfg.DATASET.normalize_input :
      data_transform_input = transforms.Compose([
                transforms.Resize((input_img_height, input_img_width)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
                normalize
            ])
    else:
      data_transform_input = transforms.Compose([
              transforms.Resize((input_img_height, input_img_width)),
              transforms.ToTensor(),
              transforms.Lambda(lambda x: torch.cat([x, x, x], 0))
          ])

  data_transform_target = transforms.Compose([
            transforms.Resize((target_img_height, target_img_width)),
            transforms.ToTensor()
        ])

  load_mask = cfg.TRAIN.use_masked_loss or cfg.MODEL.predict_conf_mask
  train_dataset = ImageQualityDataset(cfg.DATASET.root,
                              cfg.DATASET.raw_img_root,
                              session_list_train,
                              loaded_image_color=cfg.DATASET.is_dataset_color,
                              output_image_color=cfg.DATASET.use_color_images,
                              session_prefix_length=cfg.DATASET.session_prefix_len,
                              raw_img_folder=cfg.DATASET.raw_img_folder,
                              no_meta_data_available=True,
                              load_only_with_labels=True,
                              transform_input=data_transform_input,
                              transform_target=data_transform_target,
                              load_masks=load_mask,
                              regression_mode=cfg.MODEL.is_regression_mode,
                              binarize_target=cfg.DATASET.binarize_target)
  val_dataset = ImageQualityDataset(cfg.DATASET.root,
                              cfg.DATASET.raw_img_root,
                              session_list_val,
                              loaded_image_color=cfg.DATASET.is_dataset_color,
                              output_image_color=cfg.DATASET.use_color_images,
                              session_prefix_length=cfg.DATASET.session_prefix_len,
                              raw_img_folder=cfg.DATASET.raw_img_folder,
                              no_meta_data_available=True,
                              load_only_with_labels=True,
                              transform_input=data_transform_input,
                              transform_target=data_transform_target,
                              load_masks=load_mask,
                              regression_mode=cfg.MODEL.is_regression_mode,
                              binarize_target=cfg.DATASET.binarize_target)
  datasets = {phases[0]: train_dataset, phases[1]: val_dataset}

  data_loaders = {x: torch.utils.data.DataLoader(datasets[x],
                                                batch_size=BATCH_SIZE,
                                                num_workers=NUM_WORKERS)
                  for x in phases}

  # Build the network from selected modules
  net_encoder = ModelBuilder.build_encoder(
    arch=cfg.MODEL.arch_encoder.lower(),
    fc_dim=cfg.MODEL.fc_dim,
    weights=cfg.MODEL.weights_encoder)
  net_decoder = ModelBuilder.build_decoder(
    arch=cfg.MODEL.arch_decoder.lower(),
    fc_dim=cfg.MODEL.fc_dim,
    num_class=cfg.DATASET.num_class,
    weights=cfg.MODEL.weights_decoder,
    regression_mode=cfg.MODEL.is_regression_mode,
    inference_mode=False)

  if cfg.MODEL.is_regression_mode:
      if cfg.TRAIN.use_masked_loss:
        criterion = MaskedMSELoss()
        print("Regression Mode with Masked Loss")
      else:
        criterion = nn.MSELoss(reduction='mean')
        print("Regression Mode")
  else:
    criterion = nn.NLLLoss(ignore_index=-1)
    print("Segmentation Mode")


  use_mask = cfg.TRAIN.use_masked_loss and cfg.MODEL.is_regression_mode
  if cfg.MODEL.arch_decoder.endswith('deepsup'):
    net = SegmentationModule(
      net_encoder, net_decoder, criterion, cfg.TRAIN.deep_sup_scale,
      use_mask = use_mask)
  else:
    net = SegmentationModule(
      net_encoder, net_decoder, criterion, use_mask = use_mask)



  if cfg.TRAIN.use_gpu and USE_MULTI_GPU:
    if torch.cuda.device_count() >= len(gpus):
      available_gpu_count = torch.cuda.device_count()
      print("Using ", len(gpus), " GPUs out of available ", available_gpu_count)
      print("Used GPUs: ", gpus)
      net = nn.DataParallel(net, device_ids=gpus)
      # For synchronized batch normalization:
      patch_replication_callback(net)
    else:
      print("Requested GPUs not available: ", gpus)
      exit()

  net = net.to(device)


  # Set up optimizers
  modules = (net_encoder, net_decoder, criterion)
  optimizers = create_optimizers(modules, cfg)


  best_loss = 1000000000.0
  best_model_enc = copy.deepcopy(net_encoder.state_dict())
  best_model_dec = copy.deepcopy(net_decoder.state_dict())

  training_history = {x: {'loss': [], 'acc': []}
                      for x in phases}

  print("Starting Training...")
  start_time = time.time()

  # Runs training and validation
  for epoch in range(cfg.TRAIN.num_epoch):
    for cur_phase in phases:
      if cur_phase == 'train':
          net.train()  # Set model to training mode
      else:
          net.eval()   # Set model to evaluate mode

      epoch_loss = 0.0
      running_loss = 0.0
      # Iterate over data
      i = 0
      for i, data in enumerate(data_loaders[cur_phase], 0):
        # get the inputs
        input = data['img']
        feed_dict = dict()
        feed_dict['input'] = input.to(device)
        if not cfg.MODEL.is_regression_mode:
          target = data['labels']
          feed_dict['target'] = target.to(device)
        else:
          target = data['score_img']
          feed_dict['target'] = target.to(device)
          if cfg.TRAIN.use_masked_loss:
            mask = data['mask_img']
            feed_dict['mask'] = mask.to(device)

        # zero the parameter gradients
        net.zero_grad()

        # track history if only in train
        with torch.set_grad_enabled(cur_phase == 'train'):
          # forward + backward + optimize
          loss, acc = net(feed_dict)

          # For multi gpu case, loss will be a vector the same length
          # as the number of gpus. This is a side effect of loss function
          # being inside the module.
          loss = loss.mean()

          # backward and optimize only if in training phase
          if cur_phase == 'train':
            loss.backward()
            for optimizer in optimizers:
              optimizer.step()

        # print statistics
        loss = loss.item()
        running_loss += loss
        epoch_loss += loss


        if cur_phase == 'train':
          if i % 100 == 99:    # print every 100 mini-batches
            print('[%d, %5d] Loss: %.6f' %
                (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

            if used_gpu_count:
              print("Total GPU usage (MB): ",
                calculate_gpu_usage(gpus), " / ",
                used_gpu_count * total_mem)

      epoch_loss = epoch_loss / i
      print('%s: Loss: %.6f' %
            (cur_phase, epoch_loss))

      # Keep the best model so far
      if cur_phase == 'val' and epoch_loss < best_loss:
        best_loss = epoch_loss
        best_model_enc = copy.deepcopy(net_encoder.state_dict())
        best_model_dec = copy.deepcopy(net_decoder.state_dict())

      training_history[cur_phase]['loss'].append(epoch_loss)

    print('Epoch #%d finished. *******************' % (epoch + 1))
    if (epoch + 1) % cfg.TRAIN.snapshot_interval == 0:
      last_model_state = (net_encoder.state_dict(), net_decoder.state_dict())
      best_model_state = (best_model_enc, best_model_dec)
      save_results(cfg.DIR, cfg.MODEL.name,last_model_state ,
                    best_model_state, training_history, snapshot_idx = epoch)
      print('Snapshot saved.')

  print('Finished Training')
  time_elapsed = time.time() - start_time
  print('Completed in {:.0f}h {:.0f}m {:.0f}s'.format(
      time_elapsed//3600, (time_elapsed % 3600)//60, time_elapsed % 60))

  last_model_state = (net_encoder.state_dict(), net_decoder.state_dict())
  best_model_state = (best_model_enc, best_model_dec)
  save_results(cfg.DIR, cfg.MODEL.name, last_model_state,
                best_model_state, training_history, snapshot_idx = -1)



if __name__=="__main__":
  assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
    'PyTorch>=0.4.0 is required'

  parser = argparse.ArgumentParser(description='Training Modular '
                      'Segmentation Network Architectures for IVSLAM.')
  parser.add_argument(
                      "--cfg",
                      default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
                      metavar="FILE",
                      help="path to config file",
                      type=str)
  parser.add_argument(
                      "--gpus",
                      default="0",
                      help="gpus to use, e.g. 0-3 or 0,1,2,3")
  parser.add_argument(
                      "opts",
                      help="Modify config options using the command-line",
                      default=None,
                      nargs=argparse.REMAINDER)

  args = parser.parse_args()
  cfg.merge_from_file(args.cfg)
  cfg.merge_from_list(args.opts)

  print("Reading config file " + args.cfg)

  # Start from checkpoint
  if cfg.TRAIN.start_epoch > 0:
    suffix = '_' + str('%04d' % cfg.TRAIN.start_epoch)
    cfg.MODEL.weights_encoder = (
      cfg.DIR + '/' + cfg.MODEL.name +
      '/snapshots/{}_encoder_last_model{}.pth'.format(cfg.MODEL.name, suffix))
    cfg.MODEL.weights_decoder = (
      cfg.DIR + '/' + cfg.MODEL.name +
      '/snapshots/{}_decoder_last_model{}.pth'.format(cfg.MODEL.name, suffix))

    print("Loading encoder from " + cfg.MODEL.weights_encoder)
    print("Loading decoder from " + cfg.MODEL.weights_decoder)
    assert os.path.exists(cfg.MODEL.weights_encoder) and \
           os.path.exists(
             cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

  # Parse gpu ids
  gpus = parse_devices(args.gpus)
  gpus = [x.replace('gpu', '') for x in gpus]
  gpus = [int(x) for x in gpus]
  num_gpus = len(gpus)
  # cfg.TRAIN.batch_size = num_gpus * cfg.TRAIN.batch_size_per_gpu

  cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder
  cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder

  main(cfg, args, gpus)

