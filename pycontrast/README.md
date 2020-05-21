## PyContrast

### Introduction
Recently, contrastive learning approaches have significantly advanced the SoTA of 
unsupervised (visual) representation learning. This repo contains pytorch 
implementation of a set of (improved) SoTA methods using the same training and 
evaluation pipeline. 

It supports multi-node distributed training (e.g., 32 GPUs across 4 machines) and 
the mixed precision feature.

### Contents
**(1) For now, it covers the following methods as well as their combinations 
(the order follows the forward arrow of time):**

- (InstDis) Unsupervised Feature Learning via Non-parameteric Instance Discrimination
  [[pdf]](https://arxiv.org/pdf/1805.01978.pdf) 
  - Zhirong Wu, Yuanjun Xiong and X Yu Stella and Dahua Lin.
  
- (CMC) Contrastive Multiview Coding.
  [[pdf]](https://arxiv.org/abs/1906.05849) 
  [[project page]](https://hobbitlong.github.io/CMC/)
  - Yonglong Tian and Dilip Krishnan and Phillip Isola.

- (MoCo) Momentum Contrast for Unsupervised Visual Representation Learning
  [[pdf]](https://arxiv.org/pdf/1911.05722.pdf)
  - Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, Ross Girshick.

- (PIRL) Self-Supervised Learning of Pretext-Invariant Representations
  [[pdf]](https://arxiv.org/abs/1912.01991)
  - Ishan Misra, Laurens van der Maaten.

- (MoCo v2) Improved Baselines with Momentum Contrastive Learning
  [[pdf]](https://arxiv.org/pdf/2003.04297.pdf)
  - Xinlei Chen, Haoqi Fan, Ross Girshick, Kaiming He

- (InfoMin) What Makes for Good Views for Contrastive Learning?
  [[pdf]](https://arxiv.org/pdf/2005.10243.pdf)
  [[project page]](https://hobbitlong.github.io/InfoMin/)
  - Yonglong Tian, Chen Sun, Ben Poole, Dilip Krishnan, Cordelia Schmid, Phillip Isola 

**(2) The following figure illustrates the similarity and dissimilarity between these methods, 
in terms of training pipeline. Question mark `?` means unreported methods which are also supported.**
<p align="center">
  <img src="figures/models.png" width="600">
</p>


### Results on ImageNet linear readout benchmark
(1) Results with ResNet-50:
|          |Arch | # Parameters | Epochs | Accuracy(%) |
|----------|:----:|:---:|:---:|:---:|
|  InstDis         | ResNet-50 | 24M   | 200 |  59.5  |
|  CMC (no RA)     | ResNet-50*| 12M   | 200 |  58.6  |
|  MoCo            | ResNet-50 | 24M   | 200 |  60.8  | 
|  PIRL            | ResNet-50 | 24M   | 200 |  61.7  |
|  MoCo v2         | ResNet-50 | 24M   | 200 |  67.5  |
|  InfoMin         | ResNet-50 | 24M   | 100 |  67.4  |
|  InfoMin         | ResNet-50 | 24M   | 200 |  70.1  |
|  InfoMin         | ResNet-50 | 24M   | 800 |  73.0  |

(2) InfoMin with other architectures:
|          |Arch | # Parameters | Epochs | Accuracy(%) |
|----------|:----:|:---:|:---:|:---:|
|  InfoMin         | ResNet-101 | 43M   | 300 |  73.4  |
|  InfoMin         | ResNet-152 | 58M   | 200 |  73.4  |
|  InfoMin         | ResNeXt-100 | 87   | 200 |  74.5  | 
|  InfoMin         | ResNeXt-150 | 120  | 200 |  75.2  |

### Install Environments
Please see [INSTALL.md](docs/INSTALL.md).

### Running
For training and testing different models, please see [RUN.md](docs/RUN.md).

### Model Zoo
For pre-trained models and results, please check [MODEL_ZOO.md](docs/MODEL_ZOO.md)

### Detection

 
