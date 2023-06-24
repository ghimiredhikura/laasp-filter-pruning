# LAASP: Loss-Aware Automatic Selection of Filter Pruning Criteria for Deep Neural Network Acceleration 

> [Ghimire, Deepak, Kilho Lee, and Seong-heum Kim. "Loss-aware automatic selection of structured pruning criteria for deep neural network acceleration." Image and Vision Computing (2023): 104745.](https://www.sciencedirect.com/science/article/pii/S0262885623001191)

![alt text](images/LAASP_flyer.png)

## Table of Contents

- [Requirements](#requirements)
- [Models](#models)
- [VGGNet on CIFAR-10](#vggnet-on-cifar-10)
  - [Training-Pruning](#training-pruning)
  - [Evaluation](#evaluation)
- [ResNet on CIFAR-10](#resnet-on-cifar-10)
  - [Training-Pruning](#training-pruning-1)
  - [Evaluation](#evaluation-1)
- [ResNet on ImageNet](#resnet-on-imagenet)
  - [Prepare ImageNet dataset](#prepare-imagenet-dataset)
  - [Training-Pruning](#training-pruning-2)
  - [Evaluation](#evaluation-2)
- [Reference](#reference)

## Requirements
- Python 3.9.7
- PyTorch 1.10.2
- TorchVision 0.11.2
- matplotlib 3.5.1
- scipy 1.8.0

## Models


## VGGNet on CIFAR-10

- TODO

### Training-Pruning

- TODO

### Evaluation

- TODO

## ResNet on CIFAR-10

- TODO

### Training-Pruning

- TODO

### Evaluation

- TODO

## ResNet on ImageNet

### Prepare ImageNet dataset

- Download the images from http://image-net.org/download-images

- Extract the training data:

  ```ruby
  mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
  tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
  find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
  cd ..
  ```

- Extract the validation data and move images to subfolders:

  ```ruby
  mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
  wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
  ```

### Training-Pruning

- TODO

### Evaluation

- TODO

### Reference

```
@article{ghimire2023loss,
  title={Loss-aware automatic selection of structured pruning criteria for deep neural network acceleration},
  author={Ghimire, Deepak and Lee, Kilho and Kim, Seong-heum},
  journal={Image and Vision Computing},
  pages={104745},
  year={2023},
  publisher={Elsevier}
}
```