#! /bin/bash

# Train models for classifying 8-20 gestures in CapgMyo dataset
for((i=8; i <= 8; i++));
do
    python -m emg.app --model net \
                      --gesture_num $i \
                      --lr 0.01 \
                      --epoch 30 \
                      --train_batch_size 128 \
                      --val_batch_size 1024 \
                      --stop_patience 7 \
                      --load_model False
done
