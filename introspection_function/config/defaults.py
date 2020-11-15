from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
# The path to save the trained model under as well as load the checkpoints
# The checkpoints are expected under DIR/MODEL.name/snapshots
# The final models will be saved under DIR/MODEL.name/
_C.DIR = ""

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()

_C.DATASET.num_class = 1

# Path to processed training data
_C.DATASET.root = ""
# Path to the base dataset directory that has the raw images.
_C.DATASET.raw_img_root = ""
# The name of folder under raw_img_root/session#/ that holds the input images.
_C.DATASET.raw_img_folder = ""
# Training set name
_C.DATASET.train_set = ""
# Validation set name
_C.DATASET.validation_set = ""
# Test set name
_C.DATASET.test_set = ""
# True if you want to feed the network with color
_C.DATASET.use_color_images = True
# True if the dataset has color images.
_C.DATASET.is_dataset_color = True
# Length of session names in digits (%0Nd)
_C.DATASET.session_prefix_len = 2
# Normalize images
_C.DATASET.normalize_input = True


# The size to load the input and target images at (before downsampling)
_C.DATASET.img_width = 960
_C.DATASET.img_height = 600
# downsampling rate of the target image
_C.DATASET.target_downsampling_rate = 8

# binarizes the target heatmap images upon loading. It is overwritten in
# segmentation mode, where the target images should always be binarized
_C.DATASET.binarize_target = False


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# True if you want to train in regression mode.
_C.MODEL.is_regression_mode = True
# name of the model
_C.MODEL.name = "model_name"
# architecture of net_encoder
_C.MODEL.arch_encoder = "resnet50dilated"
# architecture of net_decoder
_C.MODEL.arch_decoder = "ppm_deepsup"
# weights to finetune net_encoder
_C.MODEL.weights_encoder = ""
# weights to finetune net_decoder
_C.MODEL.weights_decoder = ""
# number of feature channels between encoder and decoder
_C.MODEL.fc_dim = 2048
# predict confidence mask. If set to true, the model will predict a confidence
# mask as well as the predicted heatmap for image features.
_C.MODEL.predict_conf_mask = False

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.use_gpu = True
_C.TRAIN.batch_size_per_gpu = 2
# If use_gpu is True , batch_size will be overwritten by
# batch_size_per_gpu * num_gpus
_C.TRAIN.batch_size = 5
# number of data loading workers
_C.TRAIN.workers = 2
# epochs to train for
_C.TRAIN.num_epoch = 200
# epoch to start training. useful if continue from a checkpoint
_C.TRAIN.start_epoch = 0
# Save a checkpoint every snapshot_interval epochs
_C.TRAIN.snapshot_interval = 5


_C.TRAIN.optim = "SGD"
_C.TRAIN.lr_encoder = 0.02
_C.TRAIN.lr_decoder = 0.02
# power in poly to drop LR
_C.TRAIN.lr_pow = 0.9
# momentum for sgd, beta1 for adam
_C.TRAIN.beta1 = 0.9
# weights regularizer
_C.TRAIN.weight_decay = 1e-4
# the weighting of deep supervision loss
_C.TRAIN.deep_sup_scale = 0.4
# fix bn params, only under finetuning
_C.TRAIN.fix_bn = False
# use masked loss function
_C.TRAIN.use_masked_loss = False





# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------
_C.TEST = CN()
# If ground truth is available it will be added to the visualization
_C.TEST.ground_truth_available = False
# Set to true if you want to visualize output images as well as their
# visualization next to ground truth target or the input
_C.TEST.save_raw_output = False
# The size of the raw output image (network output will be resized to that)
_C.TEST.output_img_width = 960
_C.TEST.output_img_height = 600
# The path to model to test on
_C.TEST.test_model_encoder = "epoch_20.pth"
_C.TEST.test_model_decoder = "epoch_20.pth"
_C.TEST.use_gpu = True
_C.TEST.batch_size_per_gpu = 2
# If use_gpu is True , batch_size will be overwritten by
# batch_size_per_gpu * num_gpus
_C.TEST.batch_size = 10
# number of data loading workers
_C.TEST.workers = 16
# folder to output visualization results
_C.TEST.result = "./"
