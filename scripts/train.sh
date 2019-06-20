#! /bin/bash

## Train models for classifying 8-20 gestures in CapgMyo dataset

# LSTM model ########################################################################
#python -m emg.train lstm \
#                  --suffix default \
#                  --sub_folder gesture-8 \
#                  --gesture_num 8 \
#                  --epoch 200 \
#                  --train_batch_size 256 \
#                  --valid_batch_size 1024 \
#                  --lr 0.001 \
#                  --lr_step 70

# C3D model ########################################################################
#python -m emg.train c3d \
#                  --suffix default \
#                  --sub_folder gesture-8 \
#                  --gesture_num 8 \
#                  --epoch 200 \
#                  --train_batch_size 256 \
#                  --valid_batch_size 1024 \
#                  --lr 0.001 \
#                  --lr_step 70

# CNN model ########################################################################
#python -m emg.train cnn \
#                  --suffix default \
#                  --sub_folder gesture-8 \
#                  --gesture_num 8 \
#                  --epoch 120 \
#                  --train_batch_size 256 \
#                  --valid_batch_size 1024 \
#                  --lr 0.01 \
#                  --lr_step 40

# MLP model ########################################################################
#python -m emg.train mlp \
#                  --suffix default \
#                  --sub_folder gesture-8 \
#                  --gesture_num 8 \
#                  --epoch 200 \
#                  --train_batch_size 256 \
#                  --valid_batch_size 1024 \
#                  --lr 0.001 \
#                  --lr_step 70
