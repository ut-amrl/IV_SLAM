#!/bin/bash

fileId=1ozUlkYiVstek8Q9y68lsJgaH_dRgANgD
fileName=Jackal_Visual_Odom.zip

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${fileId}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${fileId}" -o ${fileName} 

unzip $fileName
