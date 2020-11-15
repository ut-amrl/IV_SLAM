#!/bin/bash

 python \
../export_model_light.py \
 --cfg "../../config/jackal/jackal_mobilenetv2dialated-c1_deepsup_reg.yaml" \
 --output_model "iv_jackal_mobilenet_c1deepsup_light.pt"