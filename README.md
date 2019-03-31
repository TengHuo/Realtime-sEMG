# MSc Project Real Time sEMG

## Summary

Measurements of the electrical activity in human muscles (electromyography, aka EMG) has a wide variety of uses in the clinical and rehabilitation domains. Machine learning using EMG data is an active research field and is often applied to making predictions on real-time EMG recordings to control prosthetic or orthotic devices. For example, EMG signals collected from the upper arm can be used to control the motion of a prosthetic hand, which is attached to the residual limb. This project will look at different deep learning methods, to investigate and evaluate which state of the art deep learning architectures are most suitable for real-time EMG prediction. The project will use existing open datasets of EMG signals collected during hand prosthetic control.

## TODO

1. 把模型训练的代码封装到py文件里，notebook调用
2. 封装生成val数据的代码到训练py中
3. 封装保存参数和模型的代码
4. 添加保存history的代码

## Project Progress

### data research and preprocessing code
- [x] illustrate the original data
- [x] submit data report
- [x] visualize EMG data：将EMG图片合成视频，但是无法看出有明显的规律（比如说某个sensor比较敏感）

### existing gesture classification model training 
- [ ] keras YouTube tutorial
- [ ] tensorboard tutorial
- [x] 训练一个MLP
- [x] 实现一遍论文模型，训练一个CNN
- [x] 按照gesture classification的模型训练一个CNN
- [ ] 训练一个LSTM
- [ ] 训练一个CovnLSTM

### gesture classification by transformer model, 
- [ ] 训练一个用于分类的transformer
- [ ] 找其他的gesture classification with deep learning method的论文，研究模型和数据处理方式，改进训练好的模型
- [ ] review the project proposal and submit MSc preliminary project report
- [ ] complete 1st MSc project progress report

### triplet pairs generator code and model training
- [ ] 实现一个triplet data generator，封装数据load的代码
- [ ] 用CNN训练一个siamese network，如何训练？参考blog搞个demo
- [ ] complete 2nd MSc project progress report

### model improvement
**TODO: add content**

### the essay and the demonstration software system
- [ ] 搞一组Myo传感器
- [ ] 高密度传感器，首先识别出来那些部位的传感器正对着肌肉，然后选一部分传入模型（特征选择），从视频看来可能行不太通，没法找出一个明显的区域

## Schedule

* Deadline: 15th March, data research and preprocessing code,
* Deadline: 22th March, existing gesture classification models training ,
* Deadline: 29th March, triplet pairs generator code,
* Deadline: 12th April, gesture classification by transformer model, 
* Deadline: 26th April, existing models training with triplet pairs,
* Deadline: 17th May, the transformer model training with triplet pairs,
* Deadline: 28th June, model improvement,
* Deadline: 26th July, the essay and the demonstration software system.

## Getting Started

### Requirements

* [Docker](http://docker.io/)
* Optional: A CUDA compatible GPU
* Optional: [Nvidia Docker](https://github.com/NVIDIA/nvidia-docker)

### Quick Usage

**TODO: add usage commands, how to config**

```bash
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

