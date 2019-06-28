#! /bin/bash

## Train models for classifying 8-20 gestures in CapgMyo dataset

# LSTM model ########################################################################
#python -m emg.train lstm \
#                  --suffix earlystop \
#                  --sub_folder stop-5 \
#                  --gesture_num 8 \
#                  --epoch 200 \
#                  --train_batch_size 256 \
#                  --valid_batch_size 1024 \
#                  --lr 0.001 \
#                  --lr_step 70 \
#                  --stop_patience 5

# C3D model ########################################################################
#python -m emg.train c3d \
#                  --suffix earlystop \
#                  --sub_folder k-ford \
#                  --gesture_num 8 \
#                  --epoch 180 \
#                  --train_batch_size 256 \
#                  --valid_batch_size 1024 \
#                  --lr 0.001 \
#                  --lr_step 40

# CNN model ########################################################################
#python -m emg.train cnn \
#                  --suffix baseline \
#                  --sub_folder gesture-8 \
#                  --gesture_num 8 \
#                  --epoch 180 \
#                  --train_batch_size 256 \
#                  --valid_batch_size 1024 \
#                  --lr 0.001 \
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
