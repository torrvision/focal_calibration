# Focal Calibration

This repository contains the code for [*Calibrating Deep Neural Networks using Focal Loss*](https://arxiv.org/abs/2002.09437), which has been accepted for publication in NeurIPS 2020.

If the code or the paper has been useful in your research, please add a citation to our work:

```
@article{mukhoti2020calibrating,
  title={Calibrating Deep Neural Networks using Focal Loss},
  author={Mukhoti, Jishnu and Kulharia, Viveka and Sanyal, Amartya and Golodetz, Stuart and Torr, Philip HS and Dokania, Puneet K},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}
```

## Dependencies

The code is based on PyTorch and requires a few further dependencies, listed in [environment.yml](environment.yml). It should work with newer versions as well.


### Datasets

Most datasets will be downloaded directly on running the code. However, [Tiny ImageNet](https://tiny-imagenet.herokuapp.com/) needs to be downloaded separately for the data loader to work.

## Training

In order to train a model, please use the [train.py](train.py) script. The default configuration (i.e., just running ```python train.py```) will train a ResNet50 model on the cross-entropy loss function. The following are the important parameters of the training:
```
--dataset: dataset to train on [cifar10/cifar100/tiny_imagenet]
--dataset-root: path of the Tiny ImageNet dataset (not necessary for CIFAR-10/100)
--loss: loss function of choice (cross_entropy/focal_loss/focal_loss_adaptive/mmce/mmce_weighted/brier_score)
--gamma: gamma for focal loss
--lamda: lambda value for MMCE
--gamma-schedule: whether to use a scheduled gamma during training
--save-path: path for saving models
--model: model to train (resnet50/resnet110/wide_resnet/densenet121)
```

As an example, in order to train a ResNet-50 model on CIFAR-10 using focal loss with ```\gamma = 3```, we can write the following script:
```
python train.py --dataset cifar10 --model resnet50 --loss focal_loss --gamma 3.0
``` 

