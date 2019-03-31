#!/bin/bash

# Train models for classifying 8-20 gestures in CapgMyo dataset
# TODO: 分别训练MLP，CNN，LSTM，ConvLSTM
# TODO: 写个循环训练从8-20
# scripts/runsrep python -m sigr.app exp --log log --snapshot model \
#   --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
#   --root .cache/srep-dba-universal-one-fold-intra-subject \
#   --num-semg-row 16 --num-semg-col 8 \
#   --batch-size 1000 --decay-all --dataset dba \
#   --num-filter 64 \
#   crossval --crossval-type universal-one-fold-intra-subject --fold 0
# for i in $(seq 0 17 | shuf); do
#   scripts/runsrep python -m sigr.app exp --log log --snapshot model \
#     --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
#     --root .cache/srep-dba-one-fold-intra-subject-$i \
#     --num-semg-row 16 --num-semg-col 8 \
#     --batch-size 1000 --decay-all --dataset dba \
#     --num-filter 64 \
#     --params .cache/srep-dba-universal-one-fold-intra-subject/model-0028.params \
#     crossval --crossval-type one-fold-intra-subject --fold $i
# done