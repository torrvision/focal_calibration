########## RESNET50 ##################

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset tiny_imagenet \
--model resnet50 \
--dataset-root TINY_IMAGENET_DIRECTORY \
--save-path MODEL_DIRECTORY/ \
--saved_model_name resnet50_cross_entropy_100.model \
>> tiny_imagenet.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset tiny_imagenet \
--model resnet50 \
--dataset-root TINY_IMAGENET_DIRECTORY \
--save-path MODEL_DIRECTORY/ \
--saved_model_name resnet50_brier_score_100.model \
>> tiny_imagenet.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset tiny_imagenet \
--model resnet50 \
--dataset-root TINY_IMAGENET_DIRECTORY \
--save-path MODEL_DIRECTORY/ \
--saved_model_name resnet50_mmce_weighted_lamda_2.0_100.model \
>> tiny_imagenet.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset tiny_imagenet \
--model resnet50 \
--dataset-root TINY_IMAGENET_DIRECTORY \
--save-path MODEL_DIRECTORY/ \
--saved_model_name resnet50_focal_loss_gamma_1.0_100.model \
>> tiny_imagenet.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset tiny_imagenet \
--model resnet50 \
--dataset-root TINY_IMAGENET_DIRECTORY \
--save-path MODEL_DIRECTORY/ \
--saved_model_name resnet50_focal_loss_gamma_2.0_100.model \
>> tiny_imagenet.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset tiny_imagenet \
--model resnet50 \
--dataset-root TINY_IMAGENET_DIRECTORY \
--save-path MODEL_DIRECTORY/ \
--saved_model_name resnet50_focal_loss_gamma_3.0_100.model \
>> tiny_imagenet.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset tiny_imagenet \
--model resnet50 \
--dataset-root TINY_IMAGENET_DIRECTORY \
--save-path MODEL_DIRECTORY/ \
--saved_model_name resnet50_focal_loss_scheduled_gamma_5.0_3.0_1.0_100.model \
>> tiny_imagenet.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset tiny_imagenet \
--model resnet50 \
--dataset-root TINY_IMAGENET_DIRECTORY \
--save-path MODEL_DIRECTORY/ \
--saved_model_name resnet50_focal_loss_scheduled_gamma_5.0_3.0_2.0_100.model \
>> tiny_imagenet.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset tiny_imagenet \
--model resnet50 \
--dataset-root TINY_IMAGENET_DIRECTORY \
--save-path MODEL_DIRECTORY/ \
--saved_model_name resnet50_focal_loss_adaptive_gamma_2.0_100.model \
>> tiny_imagenet.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset tiny_imagenet \
--model resnet50 \
--dataset-root TINY_IMAGENET_DIRECTORY \
--save-path MODEL_DIRECTORY/ \
--saved_model_name resnet50_focal_loss_adaptive_gamma_3.0_100.model \
>> tiny_imagenet.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset tiny_imagenet \
--model resnet50 \
--dataset-root TINY_IMAGENET_DIRECTORY \
--save-path MODEL_DIRECTORY/ \
--saved_model_name resnet50_cross_entropy_smoothed_smoothing_0.05_100.model \
>> tiny_imagenet.txt


