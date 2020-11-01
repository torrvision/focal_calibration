########## RESNET50 ##################

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model resnet50 \
--save-path MODEL_DIRECTORY/ \
--saved_model_name resnet50_cross_entropy_350.model \
>> c10.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model resnet50 \
--save-path MODEL_DIRECTORY/ \
--saved_model_name resnet50_brier_score_350.model \
>> c10.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model resnet50 \
--save-path MODEL_DIRECTORY/ \
--saved_model_name resnet50_mmce_weighted_lamda_2.0_350.model \
>> c10.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model resnet50 \
--save-path MODEL_DIRECTORY/ \
--saved_model_name resnet50_focal_loss_gamma_1.0_350.model \
>> c10.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model resnet50 \
--save-path MODEL_DIRECTORY/ \
--saved_model_name resnet50_focal_loss_gamma_2.0_350.model \
>> c10.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model resnet50 \
--save-path MODEL_DIRECTORY/ \
--saved_model_name resnet50_focal_loss_gamma_3.0_350.model \
>> c10.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model resnet50 \
--save-path MODEL_DIRECTORY/ \
--saved_model_name resnet50_focal_loss_scheduled_gamma_1.0_350.model \
>> c10.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model resnet50 \
--save-path MODEL_DIRECTORY/ \
--saved_model_name resnet50_focal_loss_scheduled_gamma_2.0_350.model \
>> c10.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model resnet50 \
--save-path MODEL_DIRECTORY/ \
--saved_model_name resnet50_focal_loss_adaptive_532_350.model \
>> c10.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model resnet50 \
--save-path MODEL_DIRECTORY/ \
--saved_model_name resnet50_focal_loss_adaptive_53_350.model \
>> c10.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model resnet50 \
--save-path MODEL_DIRECTORY/ \
--saved_model_name resnet50_cross_entropy_smoothed_smoothing_0.05_350.model \
>> c10.txt


########## RESNET110 ##################

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model resnet110 \
--save-path MODEL_DIRECTORY/ \
--saved_model_name resnet110_cross_entropy_350.model \
>> c10.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model resnet110 \
--save-path MODEL_DIRECTORY/ \
--saved_model_name resnet110_brier_score_350.model \
>> c10.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model resnet110 \
--save-path MODEL_DIRECTORY/ \
--saved_model_name resnet110_mmce_weighted_lamda_2.0_350.model \
>> c10.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model resnet110 \
--save-path MODEL_DIRECTORY/ \
--saved_model_name resnet110_focal_loss_gamma_1.0_350.model \
>> c10.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model resnet110 \
--save-path MODEL_DIRECTORY/ \
--saved_model_name resnet110_focal_loss_gamma_2.0_350.model \
>> c10.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model resnet110 \
--save-path MODEL_DIRECTORY/ \
--saved_model_name resnet110_focal_loss_gamma_3.0_350.model \
>> c10.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model resnet110 \
--save-path MODEL_DIRECTORY/ \
--saved_model_name resnet110_focal_loss_scheduled_gamma_1.0_350.model \
>> c10.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model resnet110 \
--save-path MODEL_DIRECTORY/ \
--saved_model_name resnet110_focal_loss_scheduled_gamma_2.0_350.model \
>> c10.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model resnet110 \
--save-path MODEL_DIRECTORY/ \
--saved_model_name resnet110_focal_loss_adaptive_532_350.model \
>> c10.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model resnet110 \
--save-path MODEL_DIRECTORY/ \
--saved_model_name resnet110_focal_loss_adaptive_53_350.model \
>> c10.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model resnet110 \
--save-path MODEL_DIRECTORY/ \
--saved_model_name resnet110_cross_entropy_smoothed_smoothing_0.05_350.model \
>> c10.txt


########## WIDE-RESNET ##################

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model wide_resnet \
--save-path MODEL_DIRECTORY/ \
--saved_model_name wide_resnet_cross_entropy_350.model \
>> c10.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model wide_resnet \
--save-path MODEL_DIRECTORY/ \
--saved_model_name wide_resnet_brier_score_550.model \
>> c10.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model wide_resnet \
--save-path MODEL_DIRECTORY/ \
--saved_model_name wide_resnet_mmce_weighted_lamda_2.0_350.model \
>> c10.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model wide_resnet \
--save-path MODEL_DIRECTORY/ \
--saved_model_name wide_resnet_focal_loss_gamma_1.0_350.model \
>> c10.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model wide_resnet \
--save-path MODEL_DIRECTORY/ \
--saved_model_name wide_resnet_focal_loss_gamma_2.0_350.model \
>> c10.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model wide_resnet \
--save-path MODEL_DIRECTORY/ \
--saved_model_name wide_resnet_focal_loss_gamma_3.0_350.model \
>> c10.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model wide_resnet \
--save-path MODEL_DIRECTORY/ \
--saved_model_name wide_resnet_focal_loss_scheduled_gamma_1.0_350.model \
>> c10.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model wide_resnet \
--save-path MODEL_DIRECTORY/ \
--saved_model_name wide_resnet_focal_loss_scheduled_gamma_2.0_350.model \
>> c10.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model wide_resnet \
--save-path MODEL_DIRECTORY/ \
--saved_model_name wide_resnet_focal_loss_adaptive_532_350.model \
>> c10.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model wide_resnet \
--save-path MODEL_DIRECTORY/ \
--saved_model_name wide_resnet_focal_loss_adaptive_53_350.model \
>> c10.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model wide_resnet \
--save-path MODEL_DIRECTORY/ \
--saved_model_name wide_resnet_cross_entropy_smoothed_smoothing_0.05_350.model \
>> c10.txt


########## DENSENET121 ##################

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model densenet121 \
--save-path MODEL_DIRECTORY/ \
--saved_model_name densenet121_cross_entropy_350.model \
>> c10.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model densenet121 \
--save-path MODEL_DIRECTORY/ \
--saved_model_name densenet121_brier_score_350.model \
>> c10.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model densenet121 \
--save-path MODEL_DIRECTORY/ \
--saved_model_name densenet121_mmce_weighted_lamda_2.0_350.model \
>> c10.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model densenet121 \
--save-path MODEL_DIRECTORY/ \
--saved_model_name densenet121_focal_loss_gamma_1.0_350.model \
>> c10.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model densenet121 \
--save-path MODEL_DIRECTORY/ \
--saved_model_name densenet121_focal_loss_gamma_2.0_350.model \
>> c10.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model densenet121 \
--save-path MODEL_DIRECTORY/ \
--saved_model_name densenet121_focal_loss_gamma_3.0_350.model \
>> c10.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model densenet121 \
--save-path MODEL_DIRECTORY/ \
--saved_model_name densenet121_focal_loss_scheduled_gamma_1.0_350.model \
>> c10.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model densenet121 \
--save-path MODEL_DIRECTORY/ \
--saved_model_name densenet121_focal_loss_scheduled_gamma_2.0_350.model \
>> c10.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model densenet121 \
--save-path MODEL_DIRECTORY/ \
--saved_model_name densenet121_focal_loss_adaptive_532_350.model \
>> c10.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model densenet121 \
--save-path MODEL_DIRECTORY/ \
--saved_model_name densenet121_focal_loss_adaptive_53_350.model \
>> c10.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar10 \
--model densenet121 \
--save-path MODEL_DIRECTORY/ \
--saved_model_name densenet121_cross_entropy_smoothed_smoothing_0.05_350.model \
>> c10.txt
