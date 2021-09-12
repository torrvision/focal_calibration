#This is the script we finally used: trained for 50 epochs and used the model with the best validation accuracy.
CUDA_VISIBLE_DEVICES=0 python 20newsgroup_train.py --num_epochs 50 \
--loss mmce_weighted \
--lamda 8.0 > Outputs/focalCalibration/nlp/20ng/mmce_weighted_lamda_8.0_50.txt

CUDA_VISIBLE_DEVICES=0 python 20newsgroup_train.py --num_epochs 50 \
--loss mmce_weighted \
--lamda 2.0 > Outputs/focalCalibration/nlp/20ng/mmce_weighted_lamda_2.0_50.txt

CUDA_VISIBLE_DEVICES=0 python 20newsgroup_train.py --num_epochs 50 \
--loss cross_entropy > Outputs/focalCalibration/nlp/20ng/cross_entropy_50.txt

CUDA_VISIBLE_DEVICES=0 python 20newsgroup_train.py --num_epochs 50 \
--loss focal_loss \
--gamma 1.0 > Outputs/focalCalibration/nlp/20ng/focal_loss_gamma_1.0_50.txt

CUDA_VISIBLE_DEVICES=0 python 20newsgroup_train.py --num_epochs 50 \
--loss focal_loss \
--gamma 2.0 > Outputs/focalCalibration/nlp/20ng/focal_loss_gamma_2.0_50.txt

CUDA_VISIBLE_DEVICES=0 python 20newsgroup_train.py --num_epochs 50 \
--loss focal_loss \
--gamma 3.0 > Outputs/focalCalibration/nlp/20ng/focal_loss_gamma_3.0_50.txt

CUDA_VISIBLE_DEVICES=0 python 20newsgroup_train.py --num_epochs 50 \
--loss focal_loss \
--gamma-schedule 1 \
--gamma 5.0 \
--gamma2 3.0 \
--gamma3 1.0 > Outputs/focalCalibration/nlp/20ng/focal_loss_scheduled_gamma_5.0_3.0_1.0_50.txt

CUDA_VISIBLE_DEVICES=0 python 20newsgroup_train.py --num_epochs 50 \
--loss focal_loss \
--gamma-schedule 1 \
--gamma 5.0 \
--gamma2 3.0 \
--gamma3 2.0 > Outputs/focalCalibration/nlp/20ng/focal_loss_scheduled_gamma_5.0_3.0_2.0_50.txt

CUDA_VISIBLE_DEVICES=0 python 20newsgroup_train.py --num_epochs 50 \
--loss focal_loss_adaptive \
--gamma 3.0 > Outputs/focalCalibration/nlp/20ng/focal_loss_adaptive_gamma_3.0_50.txt

CUDA_VISIBLE_DEVICES=0 python 20newsgroup_train.py --num_epochs 50 \
--loss focal_loss_adaptive \
--gamma 2.0 > Outputs/focalCalibration/nlp/20ng/focal_loss_adaptive_gamma_2.0_50.txt

CUDA_VISIBLE_DEVICES=0 python 20newsgroup_train.py --num_epochs 50 \
--loss brier_score > Outputs/focalCalibration/nlp/20ng/brier_score_50.txt
