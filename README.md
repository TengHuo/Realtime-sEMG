# MSc Project Real Time sEMG

## Summary

Surface electromyography (sEMG) is a non-invasive measurements of the electrical activity in human muscles, which has a wide variety of uses in the clinical and rehabilitation domains. Gesture recognition is also an important application scenario for sEMG data. It plays an important role in prosthetic control and human-computer interaction. In this project, several different gesture recognition models based on deep learning approach were evaluated. The state-of-the-art deep learning method based on sEMG data mainly uses two-dimension convolutional neural networks (CNNs). Inspired by this, this project applied a three-dimensional CNN on sEMG dataset for gesture recognition problem. By utilizing both temporal and spatial changes, it can extract implicit information from segmented sEMG image sequence. Moreover, sEMG is a signal sequence data that is naturally suitable for recurrent neural networks (RNNs). This project also tried a novel RNNs model, stacked Long short-term memory (LSTM), applied to sEMG data. To validate the capabilities of the two models on sEMG, and also compare with the existing work, there are two sEMG dataset used in this project, CapgMyo and CSL-HDEMG. The accuracy of 8 gesture classification of the three-dimensional CNN and stacked LSTM model are 90.16% and 86.12% on CapgMyo dataset. In contrast, the accuracy of the existing two-dimensional CNN model reproduced in this project is 78.10%. And the gesture classification accuracy of three-dimensional CNN also reached 70.17% on CSL-HDEMG dataset.

## Getting Started

### Requirements

* Ubuntu 19.04
* A CUDA compatible GPU
* [CUDA](https://developer.nvidia.com/cuda-downloads) 10.2 
* [Anaconda](https://www.anaconda.com/)

### Config Environment

```bash
$ conda env update --file environment.yml
$ conda activate emg
$ mkdir data # the folder for training data
$ mkdir checkpoints # the folder for training log
$ mkdir tensorboard # the folder for tensorboard log
```

### Quick Usage

Please download the data set first. 

CapgMyo data is [HERE](https://drive.google.com/open?id=1aEy2_5O4j7J7ls26EWdj95PGWBwr2hzK). Put the data into the ```data/capg``` folder.

CSL-HDEMG data can download from [HERE](https://www.uni-bremen.de/en/csl/research/motion-recognition.html). Put the data into the ```data/csl``` folder.

#### Train model with CapgMyo data set

```bash
$ sudo chmod +x ./experiments/capg_train.sh
$ ./experiments/capg_train.sh
```

#### Train model with CSL-HDEMG data set

```bash
$ sudo chmod +x ./experiments/csl_train.sh
$ ./experiments/capg_train.sh
```

## Directory

```bash
.
├── README.md
├── checkpoints
├── data
│   ├── capg
│   ├── capg-processed
│   ├── csl
│   └── csl-processed
├── emg
├── experiments
│   ├── capg_train.sh
│   ├── config.sh
│   ├── csl_train.sh
│   └── run
├── environment.yml
└── tensorboard
```

