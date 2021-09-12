# Training script finally used to train for 25 epochs
python sst_train.py --model_name constituency --epochs 25 --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss cross_entropy

python sst_train.py --model_name constituency --epochs 25 --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss \
--gamma 1.0

python sst_train.py --model_name constituency --epochs 25 --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss \
--gamma 2.0

python sst_train.py --model_name constituency --epochs 25 --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss \
--gamma 3.0

python sst_train.py --model_name constituency --epochs 25 --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss_adaptive \
--gamma 3.0

python sst_train.py --model_name constituency --epochs 25 --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss_adaptive \
--gamma 2.0

python sst_train.py --model_name constituency --epochs 25 --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss \
--gamma-schedule 1 \
--gamma 5.0 \
--gamma2 3.0 \
--gamma3 1.0

python sst_train.py --model_name constituency --epochs 25 --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss focal_loss \
--gamma-schedule 1 \
--gamma 5.0 \
--gamma2 3.0 \
--gamma3 2.0

python sst_train.py --model_name constituency --epochs 25 --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss mmce_weighted \
--lamda 8.0

python sst_train.py --model_name constituency --epochs 25 --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss mmce_weighted \
--lamda 2.0

python sst_train.py --model_name constituency --epochs 25 --saved Outputs/focalCalibration/nlp/treeLSTM/ \
--loss brier_score
