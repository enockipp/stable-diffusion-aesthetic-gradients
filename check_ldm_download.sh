#!/bin/bash

model_path=$1
model_md5=$(md5sum ${model_path})

echo "LDM model md5=$model_md5"

if [ "$model_md5" = "c01059060130b8242849d86e97212c84" ];then
    echo "[OK]LDM model download successfully!"
else
    echo "[ERROR]Failed to download LDM model. Please try again!"
fi
