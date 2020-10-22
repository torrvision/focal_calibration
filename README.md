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


### Pretrained models

The pretrained models for Focal Loss (Sample-Dependent gamma 5, 3) and Focal Loss (gamma 3) for all the datasets can be [downloaded from here](http://www.robots.ox.ac.uk/~viveka/focal_calibration/).

## Training a model

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

As an example, in order to train a ResNet-50 model on CIFAR-10 using focal loss with ```gamma = 3```, we can write the following script:
```
python train.py --dataset cifar10 --model resnet50 --loss focal_loss --gamma 3.0
``` 

## Evaluating a model

In order to evaluate a trained model, either use the [evaluate_single_model.ipynb](Experiments/evaluate_single_model.ipynb) notebook or you can also use the [evaluate.py](evaluate.py) script. The script provides values of calibration error scores ECE, AdaECE and Classwise-ECE along with model accuracy. In order to use the script the following parameters need to be provided:
```
--dataset: dataset to evaluate on [cifar10/cifar100/tiny_imagenet]
--dataset-root: path of the Tiny ImageNet dataset (not necessary for CIFAR-10/100)
--model: model to train (resnet50/resnet110/wide_resnet/densenet121)
--save-path: path showing the trained model's location
--saved_model_name: name of the saved model file
--cverror: error to cross-validate on for temperature scaling (ece/nll)
```

As an example, to evaluate a ResNet-50 model trained on CIFAR-10, run the script as:
```
python evaluate.py --dataset cifar10 --model resnet50 --save-path /path/to/saved/model/ --saved_model_name resnet50_cross_entropy_350.model --cverror ece
```

The jupyter notebook is simpler to use and quite self explanatory. The following is the result of evaluating the ResNet-50 we trained using the cross-entropy objective.

![ResNet50_Result](resnet50_results.png)


## OOD Notebook

To plot the ROC curve and compute the AUROC for a model trained on CIFAR-10 (in-distribution dataset) and tested on SVHN (out-of-distribution dataset), please use the [evaluate_single_model_ood.ipynb](Experiments/evaluate_single_model_ood.ipynb) notebook. The following is the ROC plot obtained from the ResNet-50 model which we trained on CIFAR-10 using the cross-entropy objective function.

<center>
<img src="roc.png" width="500" class="center">
</center>

## Questions

If you have any questions or doubts, please feel free to reach out to us at the email addresses provided in the paper or you can directly open an issue in this repository.

