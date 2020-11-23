#!/bin/bash

sudo apt-get install libgoogle-glog-dev libgflags-dev libjsoncpp-dev libeigen3-dev nvidia-cuda-toolkit

wget https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.7.0.zip

unzip libtorch-cxx11-abi-shared-with-deps-1.7.0.zip

mv libtorch introspective_ORB_SLAM/Thirdparty/libtorch