#!/bin/bash

fileIds=( 1c8EiGj1xMAskA4sjn-_X88VJdgykFAFF )
fileNames=( iv_jackal_mobilenet_c1deepsup_light.pt )

for i in ${!fileIds[@]}; do
  fileId=${fileIds[$i]}
  fileName=${fileNames[$i]}

  echo "Downloading model $fileName" 

  curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${fileId}" > /dev/null
  code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
  curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${fileId}" -o ${fileName} 

  mkdir -p introspection_function/pretrained/
  mv $fileName introspection_function/pretrained/$fileName

done