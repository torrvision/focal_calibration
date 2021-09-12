CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ --loss cross_entropy --max-dev-epoch 13 -ta 
CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ --loss brier_score --max-dev-epoch 20 -ta 
CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ --loss mmce_weighted --lamda 2.0 --max-dev-epoch 7 -ta 
CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ --loss focal_loss --gamma 1.0 --max-dev-epoch 23 -ta 
CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ --loss focal_loss --gamma 2.0 --max-dev-epoch 19 -ta 
CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ --loss focal_loss --gamma 3.0 --max-dev-epoch 8 -ta 
CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ --loss focal_loss --gamma-schedule 1 --gamma 5.0 --gamma2 3.0 --gamma3 1.0 --max-dev-epoch 18 -ta 
CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ --loss focal_loss --gamma-schedule 1 --gamma 5.0 --gamma2 3.0 --gamma3 2.0 --max-dev-epoch 18 -ta 
CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ --loss focal_loss_adaptive --gamma 2.0 --max-dev-epoch 15 -ta 
CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ --loss focal_loss_adaptive --gamma 3.0 --max-dev-epoch 15 -ta 
CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/Label_Smoothing/NLP/Best_Models/ --loss cross_entropy_smoothed --smoothing 0.05 --max-dev-epoch 24 -ta
CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/Label_Smoothing/NLP/Best_Models/ --loss cross_entropy_smoothed --smoothing 0.1 --max-dev-epoch 24 -ta





CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss mmce_weighted \
--lamda 8.0 -ce --max-dev-epoch 7

CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss mmce_weighted \
--lamda 8.0 -cae --max-dev-epoch 7

CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss mmce_weighted \
--lamda 8.0 -tn --max-dev-epoch 7

CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss mmce_weighted \
--lamda 8.0 -cn --max-dev-epoch 7




CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss mmce_weighted \
--lamda 2.0 -ce --max-dev-epoch 7

CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss mmce_weighted \
--lamda 2.0 -cae --max-dev-epoch 7

CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss mmce_weighted \
--lamda 2.0 -tn --max-dev-epoch 7

CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss mmce_weighted \
--lamda 2.0 -cn --max-dev-epoch 7




CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss cross_entropy -ce --max-dev-epoch 13

CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss cross_entropy -cae --max-dev-epoch 13

CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss cross_entropy -tn --max-dev-epoch 13

CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss cross_entropy -cn --max-dev-epoch 13




CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss \
--gamma 1.0 -ce --max-dev-epoch 23

CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss \
--gamma 1.0 -cae --max-dev-epoch 23

CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss \
--gamma 1.0 -tn --max-dev-epoch 23

CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss \
--gamma 1.0 -cn --max-dev-epoch 23




CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss \
--gamma 2.0 -ce --max-dev-epoch 19

CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss \
--gamma 2.0 -cae --max-dev-epoch 19

CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss \
--gamma 2.0 -tn --max-dev-epoch 19

CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss \
--gamma 2.0 -cn --max-dev-epoch 19




CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss \
--gamma 3.0 -ce --max-dev-epoch 8

CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss \
--gamma 3.0 -cae --max-dev-epoch 8

CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss \
--gamma 3.0 -tn --max-dev-epoch 8

CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss \
--gamma 3.0 -cn --max-dev-epoch 8




CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss \
--gamma-schedule 1 \
--gamma 5.0 \
--gamma2 3.0 \
--gamma3 1.0 -ce --max-dev-epoch 18

CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss \
--gamma-schedule 1 \
--gamma 5.0 \
--gamma2 3.0 \
--gamma3 1.0 -cae --max-dev-epoch 18

CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss \
--gamma-schedule 1 \
--gamma 5.0 \
--gamma2 3.0 \
--gamma3 1.0 -tn --max-dev-epoch 18

CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss \
--gamma-schedule 1 \
--gamma 5.0 \
--gamma2 3.0 \
--gamma3 1.0 -cn --max-dev-epoch 18




CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss \
--gamma-schedule 1 \
--gamma 5.0 \
--gamma2 3.0 \
--gamma3 2.0 -ce --max-dev-epoch 18

CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss \
--gamma-schedule 1 \
--gamma 5.0 \
--gamma2 3.0 \
--gamma3 2.0 -cae --max-dev-epoch 18

CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss \
--gamma-schedule 1 \
--gamma 5.0 \
--gamma2 3.0 \
--gamma3 2.0 -tn --max-dev-epoch 18

CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss \
--gamma-schedule 1 \
--gamma 5.0 \
--gamma2 3.0 \
--gamma3 2.0 -cn --max-dev-epoch 18




CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss_adaptive \
--gamma 3.0 -ce --max-dev-epoch 15

CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss_adaptive \
--gamma 3.0 -cae --max-dev-epoch 15

CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss_adaptive \
--gamma 3.0 -tn --max-dev-epoch 15

CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss_adaptive \
--gamma 3.0 -cn --max-dev-epoch 15




CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss_adaptive \
--gamma 2.0 -ce --max-dev-epoch 15

CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss_adaptive \
--gamma 2.0 -cae --max-dev-epoch 15

CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss_adaptive \
--gamma 2.0 -tn --max-dev-epoch 15

CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss_adaptive \
--gamma 2.0 -cn --max-dev-epoch 15




CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss brier_score \
-ce --max-dev-epoch 20

CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss brier_score \
-cae --max-dev-epoch 20

CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss brier_score \
-tn --max-dev-epoch 20

CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss brier_score \
-cn --max-dev-epoch 20
































CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss mmce_weighted \
--lamda 8.0 -ce --max-dev-epoch 7

CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss mmce_weighted \
--lamda 8.0 -cn --max-dev-epoch 7




CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss mmce_weighted \
--lamda 2.0 -ce --max-dev-epoch 7

CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss mmce_weighted \
--lamda 2.0 -cn --max-dev-epoch 7




CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss cross_entropy -ce --max-dev-epoch 13

CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss cross_entropy -cn --max-dev-epoch 13




CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss \
--gamma 1.0 -ce --max-dev-epoch 23

CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss \
--gamma 1.0 -cn --max-dev-epoch 23




CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss \
--gamma 2.0 -ce --max-dev-epoch 19

CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss \
--gamma 2.0 -cn --max-dev-epoch 19




CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss \
--gamma 3.0 -ce --max-dev-epoch 8

CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss \
--gamma 3.0 -cn --max-dev-epoch 8




CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss \
--gamma-schedule 1 \
--gamma 5.0 \
--gamma2 3.0 \
--gamma3 1.0 -ce --max-dev-epoch 18

CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss \
--gamma-schedule 1 \
--gamma 5.0 \
--gamma2 3.0 \
--gamma3 1.0 -cn --max-dev-epoch 18




CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss \
--gamma-schedule 1 \
--gamma 5.0 \
--gamma2 3.0 \
--gamma3 2.0 -ce --max-dev-epoch 18

CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss \
--gamma-schedule 1 \
--gamma 5.0 \
--gamma2 3.0 \
--gamma3 2.0 -cn --max-dev-epoch 18




CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss_adaptive \
--gamma 3.0 -ce --max-dev-epoch 15

CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss_adaptive \
--gamma 3.0 -cn --max-dev-epoch 15




CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss_adaptive \
--gamma 2.0 -ce --max-dev-epoch 15

CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss_adaptive \
--gamma 2.0 -cn --max-dev-epoch 15




CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss brier_score \
-ce --max-dev-epoch 20

CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss brier_score \
-cn --max-dev-epoch 20













CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss mmce_weighted \
--lamda 8.0 -tn --max-dev-epoch 7


CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss mmce_weighted \
--lamda 2.0 -tn --max-dev-epoch 7


CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss cross_entropy -tn --max-dev-epoch 13


CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss \
--gamma 1.0 -tn --max-dev-epoch 23


CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss \
--gamma 2.0 -tn --max-dev-epoch 19


CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss \
--gamma 3.0 -tn --max-dev-epoch 8


CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss \
--gamma-schedule 1 \
--gamma 5.0 \
--gamma2 3.0 \
--gamma3 1.0 -tn --max-dev-epoch 18


CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss \
--gamma-schedule 1 \
--gamma 5.0 \
--gamma2 3.0 \
--gamma3 2.0 -tn --max-dev-epoch 18


CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss_adaptive \
--gamma 3.0 -tn --max-dev-epoch 15


CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss_adaptive \
--gamma 2.0 -tn --max-dev-epoch 15

CUDA_VISIBLE_DEVICES=0 python sst_eval.py --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss brier_score \
-tn --max-dev-epoch 20

