#! /bin/bash

# Train models for classifying 8-20 gestures in CapgMyo dataset
for((i=8; i <= 8; i++));
do
    python -m emg.app --model cnn \
                      --gesture_num $i \
                      --lr 0.001 \
                      --epoch 30 \
                      --train_batch_size 128 \
                      --val_batch_size 1024 \
                      --stop_patience 7 \
                      --log_interval 100 \
                      --load_model False
done
