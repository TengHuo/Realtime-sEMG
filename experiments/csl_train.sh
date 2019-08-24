#! /bin/bash

## Train models for classifying first 8 gestures in CSL dataset

# LSTM model ########################################################################
python -m emg.train lstm \
                  --name CSL-Test \
                  --sub_folder LSTM \
                  --dataset csl \
                  --gesture_num 8 \
                  --epoch 120 \
                  --train_batch_size 256 \
                  --valid_batch_size 1024 \
                  --lr 0.0001 \
                  --lr_step 20

# C3D model ########################################################################
python -m emg.train c3d_csl \
                  --name CSL-Test \
                  --sub_folder C3D \
                  --dataset csl \
                  --gesture_num 8 \
                  --epoch 120 \
                  --train_batch_size 256 \
                  --valid_batch_size 1024 \
                  --lr 0.0001 \
                  --lr_step 20

# MLP model ########################################################################
# python -m emg.train mlp \
#                  --name CSL-Test \
#                  --sub_folder MLP \
#                  --dataset csl \
#                  --gesture_num 8 \
#                  --epoch 120 \
#                  --train_batch_size 512 \
#                  --valid_batch_size 2048 \
#                  --lr 0.001 \
#                  --lr_step 30
