#! /bin/bash

# Train models for classifying 8-20 gestures in CapgMyo dataset
for((i=8; i <= 20; i++));
do
    python -m emg.app --model cnn \
                      --gesture_num $i \
                      --lr 0.001 \
                      --epoch 60 \
                      --train_batch_size 256 \
                      --val_batch_size 1024 \
                      --stop_patience 7 \
                      --log_interval 100 \
                      --load_model False
done


# lstm model with bn
# python -m emg.app --model lstm \
#                   --gesture_num 8 \
#                   --lr 0.001 \
#                   --epoch 60 \
#                   --train_batch_size 256 \
#                   --val_batch_size 1024 \
#                   --stop_patience 7 \
#                   --log_interval 100 \
#                   --load_model False

# seq2seq model test
# python -m emg.app --model seq2seq \
#                   --gesture_num 8 \
#                   --lr 0.001 \
#                   --epoch 40 \
#                   --train_batch_size 128 \
#                   --val_batch_size 1024 \
#                   --stop_patience 7 \
#                   --log_interval 100 \
#                   --load_model False
