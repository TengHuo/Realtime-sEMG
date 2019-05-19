#! /bin/bash

# Train models for classifying 8-20 gestures in CapgMyo dataset
for((i=8; i <= 20; i++));
do
    python -m emg.app --model seq2seq \
                      --gesture_num $i \
                      --lr 0.01 \
                      --epoch 60 \
                      --train_batch_size 256 \
                      --val_batch_size 1024 \
                      --stop_patience 5
done

