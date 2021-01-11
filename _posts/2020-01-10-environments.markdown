---
layout:     post
title:      "Dali工具箱🔧1——torch GPU版本环境配置"
subtitle:   " \"从CUDA，cuDNN到gcc版本，傻瓜式配置各torch版本\""
date:       2021-01-10 20:46:00
author:     "fuhao7i"
header-img: "img/in-post/tools.jpg"
catalog: true
tags:
    - Dali工具箱🔧
---

> 最近在用mmdetection工具箱完成一个目标检测模块，但是这个工具箱版本更新实在太快，而且各个版本，pytorch gpu的版本还不相同。搭建GPU环境实在繁琐，而且稍不注意程序就会报各种各样的错误。既然可以有各种工具箱用于目标检测，语义分割等任务，那为什么不可以有一个工具箱来帮助大家管理和配置python环境呢？这也是我创建Dali工具箱的初衷。

# 1. Anaconda安装

Anaconda是一个管理Python环境特别好的工具箱。它可以很方便的让你在电脑上管理多个Python环境，具体可以参考Anaconda官网或者其他博客。

# 2. 安装CUDA

> 要想利用GPU进行运算，除了在电脑上安装GPU的驱动外，我们还需要安装对应版本的CUDA(Compute Unified Device Architecture)，这是显卡厂商NVIDIA推出的运算平台。CUDA™是一种由NVIDIA推出的通用并行计算架构，该架构使GPU能够解决复杂的计算问题。

我们要安装GPU版本的torch就需要使用CUDA。

## 2.1 查看电脑上现有的CUDA版本

```Bash
cat /usr/local/cuda/version.txt 
```

## 2.2 [确认GPU所支持的CUDA版本](https://developer.nvidia.com/zh-cn/cuda-gpus)

## 2.3 卸载CUDA

在目录:

```Bash
/usr/local/cuda-7.0/bin
```

有CUDA自带的卸载工具uninstall_cuda_toolkit_7.0.pl，使用命令：

```Bash
sudo ./uninstall_cuda_toolkit_7.0.pl
```

## 2.4 安装我们所需的CUDA版本

### 2.4.1 确认我们目前的环境

- 1.显卡驱动已安装
- 2.nouveau已经禁用
- 3.验证系统是否安装了gcc

GCC 编译器是 Linux 系统下最常用的 C/C++ 编译器，大部分 Linux 发行版中都会默认安装。GCC 编译器通常以gcc命令的形式在终端（Shell）中使用。有时候我们也需要注意gcc的版本问题。

```Bash
gcc --version
```

### 2.4.2 [下载CUDA文件并安装](https://developer.nvidia.com/zh-cn/cuda-downloads)

通过命令行输入`nvidia-smi`查看自己的显卡驱动版本以及支持的最大CUDA版本.

```Bash
Mon Jan 11 01:54:41 2021       
+-----------------------------------------------------------------------------+  
| NVIDIA-SMI 460.27.04    Driver Version: 418.67       CUDA Version: 10.1     |  
|-------------------------------+----------------------+----------------------+  
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |  
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |  
|                               |                      |               MIG M. |  
|===============================+======================+======================|  
|   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |  
| N/A   40C    P8     9W /  70W |      0MiB / 15079MiB |      0%      Default |  
|                               |                      |                 ERR! |  
+-------------------------------+----------------------+----------------------+  
                                                                                  
+-----------------------------------------------------------------------------+  
| Processes:                                                                  |  
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |  
|        ID   ID                                                   Usage      |  
|=============================================================================|  
|  No running processes found                                                 |  
+-----------------------------------------------------------------------------+  
```

安装的教程可以参考官网或其他博客。

### 2.4.3 配置好CUDA系统环境

将cuda的bin和lib写入系统环境
打开`～/.bashrc`文件，在末尾追加两句：
```Bash
export CUDA_HOME=/usr/local/cuda-9.2
export LD_LIBRARY_PATH=/usr/local/cuda-9.2/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-9.2/bin:$PATH
```
并重新激活环境`source ~/.bashrc`。

查看是否安装成功：
```Bash
nvcc -V
```

> Linux 安装CUDA9.0 可以使用如下代码，会自动配置好系统环境。

```Bash
!wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64-deb
!dpkg -i cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64-deb
!apt-key add /var/cuda-repo-9-0-local/7fa2af80.pub
!apt-get update
!apt-get install cuda=9.0.176-1
```

# 3. 安装cuDAA

> NVIDIA cuDNN是用于深度神经网络的GPU加速库。它强调性能、易用性和低内存开销。NVIDIA cuDNN可以集成到更高级别的机器学习框架中，如谷歌的Tensorflow、加州大学伯克利分校的流行caffe软件。简单的插入式设计可以让开发人员专注于设计和实现神经网络模型，而不是简单调整性能，同时还可以在GPU上实现高性能现代并行计算。

## 3.1 [查看和你CUDA版本对应的cuDNN版本，并下载](https://developer.nvidia.com/rdp/cudnn-archive)

## 3.2 查看电脑上现有的cuDNN版本

```Bash
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
```

## 3.3 下载相应的cuDNN文件，并‘插入式’安装到CUDA文件夹下

就是将下载的cuDNN文件复制到相应的CUDA文件夹下。

```Bash
tar -xzvf cudnn-10.2-linux-x64-v7.6.5.32.tgz

sudo cp -P cuda/include/cudnn.h /usr/local/cuda/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

sudo ldconfig
```

# 4. CUDA和cuDNN的关系

**CUDA是底层架构，类似于一个工具箱。cuDNN是基于CUDA的深度学习GPU加速库，有了它才能在GPU上完成深度学习的计算，它就相当于是实现具体功能的一个具体的工具，比如说扳手🔧。但是我们安装CUDA的时候，并没有赠送这个扳手，我们需要把cuDNN下载下来，放入工具箱中(即插入式设计，`cuDNN不会对CUDA产生任何影响`，因为是把cuDNN的文件复制到CUDA文件夹里，并没有相同文件覆盖的问题)。所以安装cuDNN也就是放文件和删除文件的问题。**

# 5. [torch gpu 安装](https://pytorch.org/get-started/locally/)

安装对应CUDA版本的torch gpu版本。

**torchversion介绍**

torchvision包是服务于pytorch深度学习框架的,用来生成图片,视频数据集,和一些流行的模型类和预训练模型. 主要包含四个模块：
`(注意torchvision和torch的版本对应关系，一般 [官网](https://pytorch.org/get-started/locally/) 安装命令会附带相应的torchvision安装)`

```Bash
1. torchvision.datasets
2. torchvision.models
3. torchvision.transforms
4. torchvision.utils
```

查看torch, torchvision版本信息：

```python
import torch
print(torch.__version__)

import torchvision
print(torchvision.__version__)

print(torch.cuda.is_available()) # torch能否正常使用GPU
print(torch.version.cuda) # torch使用的CUDA版本
```