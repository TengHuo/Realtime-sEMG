#! /bin/bash

## Train models for classifying 8-20 gestures in CSL dataset

# LSTM model ########################################################################
python -m emg.train lstm \
                  --name csl-test \
                  --sub_folder test-lstm2 \
                  --dataset csl \
                  --gesture_num 8 \
                  --epoch 10 \
                  --train_batch_size 256 \
                  --valid_batch_size 1024 \
                  --lr 0.001 \
                  --lr_step 30

# C3D model ########################################################################
#python -m emg.train c3d \
#                  --suffix 12Gesture_Compare \
#                  --sub_folder C3D \
#                  --gesture_num 12 \
#                  --epoch 120 \
#                  --train_batch_size 256 \
#                  --valid_batch_size 1024 \
#                  --lr 0.001 \
#                  --lr_step 30

# CNN model ########################################################################
#python -m emg.train cnn \
#                  --name 12Gesture_Compare \
#                  --sub_folder ConvNet \
#                  --dataset csl \
#                  --gesture_num 12 \
#                  --epoch 120 \
#                  --train_batch_size 256 \
#                  --valid_batch_size 1024 \
#                  --lr 0.001 \
#                  --lr_step 30
#
## MLP model ########################################################################
#python -m emg.train mlp \
#                  --name csl-test \
#                  --sub_folder test-mlp \
#                  --dataset csl \
#                  --gesture_num 8 \
#                  --epoch 2 \
#                  --train_batch_size 512 \
#                  --valid_batch_size 2048 \
#                  --lr 0.001 \
#                  --lr_step 30
