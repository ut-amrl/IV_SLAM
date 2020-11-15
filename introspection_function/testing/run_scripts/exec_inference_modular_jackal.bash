#!/bin/bash

python \
../inference_modular.py \
 --cfg "/home/srabiee/My_Repos/introspective_SLAM/neural_nets/image_evaluation/config/jackal/"\
"jackal_mobilenetv2dialated-c1_deepsup_reg.yaml" \
 --gpus "0"