########## RESNET50 ##################

##CE
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset tiny_imagenet \
--first-milestone 40 --second-milestone 60 -e 100 -b 64 -tb 64 \
--model resnet50 \
--dataset-root TINY_IMAGENET_DIRECTORY \
--loss cross_entropy \
--save-path MODEL_DIRECTORY/

##Brier Loss
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset tiny_imagenet \
--first-milestone 40 --second-milestone 60 -e 100 -b 64 -tb 64 \
--model resnet50 \
--dataset-root TINY_IMAGENET_DIRECTORY \
--loss brier_score --lr 0.035 \
--save-path MODEL_DIRECTORY/

##MMCE
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset tiny_imagenet \
--first-milestone 40 --second-milestone 60 -e 100 -b 64 -tb 64 \
--model resnet50 \
--dataset-root TINY_IMAGENET_DIRECTORY \
--loss mmce_weighted --lamda 2.0 \
--save-path MODEL_DIRECTORY/

##Focal loss with fixed gamma 1 (FL-1)
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset tiny_imagenet \
--first-milestone 40 --second-milestone 60 -e 100 -b 64 -tb 64 \
--model resnet50 \
--dataset-root TINY_IMAGENET_DIRECTORY \
--loss focal_loss --gamma 1.0 \
--save-path MODEL_DIRECTORY/

##Focal loss with fixed gamma 2 (FL-2)
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset tiny_imagenet \
--first-milestone 40 --second-milestone 60 -e 100 -b 64 -tb 64 \
--model resnet50 \
--dataset-root TINY_IMAGENET_DIRECTORY \
--loss focal_loss --gamma 2.0 \
--save-path MODEL_DIRECTORY/

##Focal loss with fixed gamma 3 (FL-3)
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset tiny_imagenet \
--first-milestone 40 --second-milestone 60 -e 100 -b 64 -tb 64 \
--model resnet50 \
--dataset-root TINY_IMAGENET_DIRECTORY \
--loss focal_loss --gamma 3.0 \
--save-path MODEL_DIRECTORY/

##Focal loss with Scheduled gamma 5,3,1 (FLSc-531)
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset tiny_imagenet \
--first-milestone 40 --second-milestone 60 -e 100 -b 64 -tb 64 \
--model resnet50 \
--dataset-root TINY_IMAGENET_DIRECTORY \
--loss focal_loss --gamma-schedule 1 --gamma 5.0 --gamma2 3.0 --gamma3 1.0 --gamma-schedule-step1 30 --gamma-schedule-step2 70 \
--save-path MODEL_DIRECTORY/

##Focal loss with Scheduled gamma 5,3,2 (FLSc-532)
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset tiny_imagenet \
--first-milestone 40 --second-milestone 60 -e 100 -b 64 -tb 64 \
--model resnet50 \
--dataset-root TINY_IMAGENET_DIRECTORY \
--loss focal_loss --gamma-schedule 1 --gamma 5.0 --gamma2 3.0 --gamma3 2.0 --gamma-schedule-step1 30 --gamma-schedule-step2 70 \
--save-path MODEL_DIRECTORY/

##Focal loss with sample dependent gamma 5,2 (FLSD-52)
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset tiny_imagenet \
--first-milestone 40 --second-milestone 60 -e 100 -b 64 -tb 64 \
--model resnet50 \
--dataset-root TINY_IMAGENET_DIRECTORY \
--loss focal_loss_adaptive --gamma 2.0 \
--save-path MODEL_DIRECTORY/

##Focal loss with sample dependent gamma 5,3 (FLSD-53)
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset tiny_imagenet \
--first-milestone 40 --second-milestone 60 -e 100 -b 64 -tb 64 \
--model resnet50 \
--dataset-root TINY_IMAGENET_DIRECTORY \
--loss focal_loss_adaptive --gamma 3.0 \
--save-path MODEL_DIRECTORY/

