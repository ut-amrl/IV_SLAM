# IV_SLAM
Introspective Vision for Simultaneous Localization and Mapping. 
This is an implementation of IV-SLAM for [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2). 

## Dependencies

1. [Pangolin](https://github.com/stevenlovegrove/Pangolin) 
2. [OpenCV](http://opencv.org) Version 2.4.3 and above.
3. [glog](https://github.com/google/glog)
4. [gflags](https://github.com/gflags/gflags)
5. Eigen3
6. [JsonCpp](https://github.com/open-source-parsers/jsoncpp)

Download and install Pangolin from [here](https://github.com/stevenlovegrove/Pangolin). 
You can install the rest of the dependencies on ubuntu using:
```
sudo apt-get install libgoogle-glog-dev libgflags-dev libjsoncpp-dev libeigen3-dev
```

## Build
```
cd introspective_ORB_SLAM
./build.sh
```

## Environment Setup for Training
We use [Pytorch](https://pytorch.org/) for training the introspection function. Setting up a virtual environment using [Conda](https://docs.conda.io/en/latest/) is suggested. You can install the minimal installer for Conda from [here](https://docs.conda.io/en/latest/miniconda.html). Then, create an environment using:
```
conda create -n ivslam-env python=3.7
conda activate ivslam-env

conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
conda install -c conda-forge scikit-image pillow==6.2.1 yacs tqdm 
conda install -c anaconda scikit-learn
conda install -c menpo opencv
```

## Running IV-SLAM

### Download Sample Data and Pre-trained Models
Download a pre-trained model using:
```
./download_pretrained_models.bash
```

Download a short robot deployment session using:
```
./download_sample_data.bash
```

### Run IV-SLAM in Inference Mode
Run IV-SLAM using a pre-trained model on the downloaded data:
```
cd introspective_ORB_SLAM/scripts
./run_stereo_jackal_batch_inference.bash
```
GPU will be used if available, by default. The program has been tested with cuDNN v7.6.5 and CUDA 10.2. 

### Run IV-SLAM for Training Data Generation
When run in training mode, IV-SLAM evaluates extracted image features and generates 
the labelled data required for training the introspection function. Run IV-SLAM in training mode using the following script: 
```
cd introspective_ORB_SLAM/scripts
./run_stereo_jackal_batch_training.bash
```

## Training
Once labelled training data is generated, the introspection function, implemented as a fully convolutional network, can be trained using the following command:
```
cd introspection_function/training/run_scripts
./exec_train_modular_jackal.bash
```
The path to the training data and the model architecture are provided in a config file that is passed to the training process in the above script. 


In order to use the trained PyTorch model during inference, you should first export it to Torch Script using the following script:
```
cd introspection_function/training/run_scripts
./exec_export_model_light.bash
```
Provide the path to the exported model in the [execution script](), in order for it to be loaded and used at run-time. 


## Dataset Format
IV-SLAM currently operates on input data that is formatted the same as the [KITTI](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) dataset.

## Citation
If you find this work useful in your research, please consider citing:
```
@inproceedings{rabiee2020ivslam,
    title={IV-SLAM: Introspective Vision for Simultaneous Localization and Mapping},
    author={Sadegh Rabiee and Joydeep Biswas},
    booktitle={Conference on Robot Learning (CoRL)},
    year={2020},
}
```