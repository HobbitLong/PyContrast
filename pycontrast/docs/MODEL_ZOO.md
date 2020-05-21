### Model Zoo
#### (1) ResNet-50 with different methods
The pretrained weights for ResNet-50 can be found via the dropbox
link [here](https://www.dropbox.com/sh/87d24jqsl6ra7t2/AABdDXZZBTnMQCBrg1yrQKVCa?dl=0). 
Both the weights of backbone and projection heads are provided.

|   Name    |Arch | # Parameters | Epochs |
|----------|:----:|:---:|:---:|
|  InsDis.pth            | ResNet-50 | 24M   | 200 | 
|  CMC.pth               | ResNet-50 | 12M   | 200 | 
|  MoCo.pth              | ResNet-50 | 24M   | 200 | 
|  PIRL.pth              | ResNet-50 | 24M   | 200 | 
|  MoCov2.pth            | ResNet-50 | 24M   | 200 | 
|  InfoMin_200.pth       | ResNet-50 | 24M   | 200 | 
|  InfoMin_800.pth       | ResNet-50 | 24M   | 800 | 
|  InfoMin_800_run2.pth* | ResNet-50 | 24M   | 800 | 

*: compared with `InfoMin_800.pth`, model `InfoMin_800_run2.pth` has slightly stronger 
transfer performance on COCO, and slightly weaker linear accuracy on ImageNet.

#### (2) ResNet and  ResNeXt with InfoMin
The pretrained weights for InfoMin with other architectures can be found via the dropbox
link [here](https://www.dropbox.com/sh/fit1szeqf4kiwob/AADIgPT4EItaMOtGCunBfWXCa?dl=0). 
Both the weights of backbone and projection heads are provided.

|   Name   |Arch | (bsz, gpus) | Epochs |
|----------|:----:|:---:|:---:|
|  InfoMin_resnet101_e300.pth            | ResNet-101          | (256, 8)   | 300 | 
|  InfoMin_resnet152_e200.pth            | ResNet-152          | (256, 8)   | 200 | 
|  InfoMin_resnext101v1_e200.pth         | ResNeXt-101 (32x4d) | (512, 32)  | 200 |
|  InfoMin_resnext101v2_e200.pth*        | ResNeXt-101 (32x8d) | (512, 32)  | 200 | 
|  InfoMin_resnext101v3_e200.pth         | ResNeXt-101 (64x4d) | (512, 32)  | 200 | 
|  InfoMin_resnext152v1_e200.pth         | ResNeXt-152 (32x4d) | (512, 32)  | 200 | 
|  InfoMin_resnext152v2_e200.pth*        | ResNeXt-152 (32x8d) | (512, 32)  | 200 | 

*: used to compare with ImageNet pretrained models on COCO.

#### (3) ResNeSt with InfoMin (TODO if resources suffice)
