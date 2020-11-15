#!/bin/bash

python \
../train_modular.py \
 --cfg "../../config/jackal/"\
"jackal_mobilenetv2dialated-c1_deepsup_reg.yaml" \
 --gpus "0"
