# MSc Project Real Time sEMG

## Summary

Measurements of the electrical activity in human muscles (electromyography, aka EMG) has a wide variety of uses in the clinical and rehabilitation domains. Machine learning using EMG data is an active research field and is often applied to making predictions on real-time EMG recordings to control prosthetic or orthotic devices. For example, EMG signals collected from the upper arm can be used to control the motion of a prosthetic hand, which is attached to the residual limb. This project will look at different deep learning methods, to investigate and evaluate which state of the art deep learning architectures are most suitable for real-time EMG prediction. The project will use existing open datasets of EMG signals collected during hand prosthetic control.

## Getting Started

### Requirements

* [Docker](http://docker.io/)
* Optional: A CUDA compatible GPU
* Optional: [Nvidia Docker](https://github.com/NVIDIA/nvidia-docker)

### Quick Usage

**TODO: add usage commands, how to config**

```bash
conda install --yes --file requirements.txt

# CPU training
$ docker run -it -v $(pwd):/code -p 7777:8888 teng/emg:cpu

# GPU training
$ nvidia-docker run -it -v $(pwd):/code -p 7777:8888 teng/emg:gpu
```

### Building from Source

**TODO: add build commands**

## Directory

**TODO: add explanation**

```bash
.
├── README.md
├── code
├── data
│   ├── capg
│   │   ├── dba
│   │   ├── dbb
│   │   └── dbc
│   ├── csl
│   │   ├── subject1
│   │   ├── subject2
│   │   ├── subject3
│   │   ├── subject4
│   │   └── subject5
│   ├── nina
│   │   └── DB1
│   └── sample
├── doc
│   ├── README.md
│   ├── data_report
│   ├── proposal
│   └── thesis
└── docker
    ├── cpu
    └── gpu
```

