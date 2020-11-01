########## RESNET50 ##################

##CE
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar10 \
--model resnet50 \
--loss cross_entropy \
--save-path MODEL_DIRECTORY/

##Brier Loss
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar10 \
--model resnet50 \
--loss brier_score \
--save-path MODEL_DIRECTORY/

##MMCE
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar10 \
--model resnet50 \
--loss mmce_weighted --lamda 2.0 \
--save-path MODEL_DIRECTORY/

##Focal loss with fixed gamma 1 (FL-1)
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar10 \
--model resnet50 \
--loss focal_loss --gamma 1.0 \
--save-path MODEL_DIRECTORY/

##Focal loss with fixed gamma 2 (FL-2)
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar10 \
--model resnet50 \
--loss focal_loss --gamma 2.0 \
--save-path MODEL_DIRECTORY/

##Focal loss with fixed gamma 3 (FL-3)
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar10 \
--model resnet50 \
--loss focal_loss --gamma 3.0 \
--save-path MODEL_DIRECTORY/

##Focal loss with Scheduled gamma 5,3,1 (FLSc-531)
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar10 \
--model resnet50 \
--loss focal_loss --gamma-schedule 1 --gamma 5.0 --gamma2 3.0 --gamma3 1.0 \
--save-path MODEL_DIRECTORY/

##Focal loss with Scheduled gamma 5,3,2 (FLSc-532)
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar10 \
--model resnet50 \
--loss focal_loss --gamma-schedule 1 --gamma 5.0 --gamma2 3.0 --gamma3 2.0 \
--save-path MODEL_DIRECTORY/

##Focal loss with sample dependent gamma 5,2 (FLSD-52)
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar10 \
--model resnet50 \
--loss focal_loss_adaptive --gamma 2.0 \
--save-path MODEL_DIRECTORY/

##Focal loss with sample dependent gamma 5,3 (FLSD-53)
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar10 \
--model resnet50 \
--loss focal_loss_adaptive --gamma 3.0 \
--save-path MODEL_DIRECTORY/

########## RESNET110 ##################

##CE
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar10 \
--model resnet110 \
--loss cross_entropy \
--save-path MODEL_DIRECTORY/

##Brier Loss
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar10 \
--model resnet110 \
--loss brier_score \
--save-path MODEL_DIRECTORY/

##MMCE
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar10 \
--model resnet110 \
--loss mmce_weighted --lamda 2.0 \
--save-path MODEL_DIRECTORY/

##Focal loss with fixed gamma 1 (FL-1)
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar10 \
--model resnet110 \
--loss focal_loss --gamma 1.0 \
--save-path MODEL_DIRECTORY/

##Focal loss with fixed gamma 2 (FL-2)
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar10 \
--model resnet110 \
--loss focal_loss --gamma 2.0 \
--save-path MODEL_DIRECTORY/

##Focal loss with fixed gamma 3 (FL-3)
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar10 \
--model resnet110 \
--loss focal_loss --gamma 3.0 \
--save-path MODEL_DIRECTORY/

##Focal loss with Scheduled gamma 5,3,1 (FLSc-531)
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar10 \
--model resnet110 \
--loss focal_loss --gamma-schedule 1 --gamma 5.0 --gamma2 3.0 --gamma3 1.0 \
--save-path MODEL_DIRECTORY/

##Focal loss with Scheduled gamma 5,3,2 (FLSc-532)
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar10 \
--model resnet110 \
--loss focal_loss --gamma-schedule 1 --gamma 5.0 --gamma2 3.0 --gamma3 2.0 \
--save-path MODEL_DIRECTORY/

##Focal loss with sample dependent gamma 5,2 (FLSD-52)
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar10 \
--model resnet110 \
--loss focal_loss_adaptive --gamma 2.0 \
--save-path MODEL_DIRECTORY/

##Focal loss with sample dependent gamma 5,3 (FLSD-53)
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar10 \
--model resnet110 \
--loss focal_loss_adaptive --gamma 3.0 \
--save-path MODEL_DIRECTORY/

########## WIDE-RESNET ##################

##CE
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar10 \
--model wide_resnet \
--loss cross_entropy \
--save-path MODEL_DIRECTORY/

##Brier Loss
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar10 \
--model wide_resnet \
--loss brier_score -e 550 \
--save-path MODEL_DIRECTORY/

##MMCE
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar10 \
--model wide_resnet \
--loss mmce_weighted --lamda 2.0 \
--save-path MODEL_DIRECTORY/

##Focal loss with fixed gamma 1 (FL-1)
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar10 \
--model wide_resnet \
--loss focal_loss --gamma 1.0 \
--save-path MODEL_DIRECTORY/

##Focal loss with fixed gamma 2 (FL-2)
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar10 \
--model wide_resnet \
--loss focal_loss --gamma 2.0 \
--save-path MODEL_DIRECTORY/

##Focal loss with fixed gamma 3 (FL-3)
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar10 \
--model wide_resnet \
--loss focal_loss --gamma 3.0 \
--save-path MODEL_DIRECTORY/

##Focal loss with Scheduled gamma 5,3,1 (FLSc-531)
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar10 \
--model wide_resnet \
--loss focal_loss --gamma-schedule 1 --gamma 5.0 --gamma2 3.0 --gamma3 1.0 \
--save-path MODEL_DIRECTORY/

##Focal loss with Scheduled gamma 5,3,2 (FLSc-532)
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar10 \
--model wide_resnet \
--loss focal_loss --gamma-schedule 1 --gamma 5.0 --gamma2 3.0 --gamma3 2.0 \
--save-path MODEL_DIRECTORY/

##Focal loss with sample dependent gamma 5,2 (FLSD-52)
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar10 \
--model wide_resnet \
--loss focal_loss_adaptive --gamma 2.0 \
--save-path MODEL_DIRECTORY/

##Focal loss with sample dependent gamma 5,3 (FLSD-53)
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar10 \
--model wide_resnet \
--loss focal_loss_adaptive --gamma 3.0 \
--save-path MODEL_DIRECTORY/

########## DENSENET121 ##################

##CE
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar10 \
--model densenet121 \
--loss cross_entropy \
--save-path MODEL_DIRECTORY/

##Brier Loss
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar10 \
--model densenet121 \
--loss brier_score \
--save-path MODEL_DIRECTORY/

##MMCE
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar10 \
--model densenet121 \
--loss mmce_weighted --lamda 2.0 \
--save-path MODEL_DIRECTORY/

##Focal loss with fixed gamma 1 (FL-1)
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar10 \
--model densenet121 \
--loss focal_loss --gamma 1.0 \
--save-path MODEL_DIRECTORY/

##Focal loss with fixed gamma 2 (FL-2)
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar10 \
--model densenet121 \
--loss focal_loss --gamma 2.0 \
--save-path MODEL_DIRECTORY/

##Focal loss with fixed gamma 3 (FL-3)
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar10 \
--model densenet121 \
--loss focal_loss --gamma 3.0 \
--save-path MODEL_DIRECTORY/

##Focal loss with Scheduled gamma 5,3,1 (FLSc-531)
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar10 \
--model densenet121 \
--loss focal_loss --gamma-schedule 1 --gamma 5.0 --gamma2 3.0 --gamma3 1.0 \
--save-path MODEL_DIRECTORY/

##Focal loss with Scheduled gamma 5,3,2 (FLSc-532)
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar10 \
--model densenet121 \
--loss focal_loss --gamma-schedule 1 --gamma 5.0 --gamma2 3.0 --gamma3 2.0 \
--save-path MODEL_DIRECTORY/

##Focal loss with sample dependent gamma 5,2 (FLSD-52)
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar10 \
--model densenet121 \
--loss focal_loss_adaptive --gamma 2.0 \
--save-path MODEL_DIRECTORY/

##Focal loss with sample dependent gamma 5,3 (FLSD-53)
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar10 \
--model densenet121 \
--loss focal_loss_adaptive --gamma 3.0 \
--save-path MODEL_DIRECTORY/
