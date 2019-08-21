#! /bin/bash

# Train models for classifying gestures in CapgMyo dataset

# LSTM model ########################################################################
 python -m emg.train lstm \
                   --name 12Gesture_Compare \
                   --sub_folder LSTM \
                   --dataset capg \
                   --gesture_num 12 \
                   --epoch 120 \
                   --train_batch_size 256 \
                   --valid_batch_size 1024 \
                   --lr 0.001 \
                   --lr_step 30

 C3D model ########################################################################
 python -m emg.train c3d \
                   --name 12Gesture_Compare \
                   --sub_folder C3D \
                   --dataset capg \
                   --gesture_num 12 \
                   --epoch 120 \
                   --train_batch_size 256 \
                   --valid_batch_size 1024 \
                   --lr 0.001 \
                   --lr_step 30

 # ConvNet model ########################################################################
 python -m emg.train cnn \
                   --name 12Gesture_Compare \
                   --sub_folder ConvNet \
                   --dataset capg \
                   --gesture_num 12 \
                   --epoch 120 \
                   --train_batch_size 256 \
                   --valid_batch_size 1024 \
                   --lr 0.001 \
                   --lr_step 30
 #
 ## MLP model ########################################################################
python -m emg.train mlp \
                  --name 12Gesture_Compare \
                  --sub_folder MLP \
                  --dataset capg \
                  --gesture_num 8 \
                  --epoch 1 \
                  --train_batch_size 512 \
                  --valid_batch_size 2048 \
                  --lr 0.001 \
                  --lr_step 30
