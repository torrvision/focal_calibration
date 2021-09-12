# FocalCalibration

## For SST:

Adaptation of [TreeLSTMSentiment](https://github.com/ttpro1995/TreeLSTMSentiment)

### Installations:

Refer [TreeLSTMSentiment](https://github.com/ttpro1995/TreeLSTMSentiment)

conda create -n py_v0.3_3.5 python=3.5

conda activate py_v0.3_3.5

conda install pytorch=0.3.0 torchvision cuda90 -c pytorch

conda install -c conda-forge tqdm

pip install meowlogtool

### Training and Evaluation:

Refer sst_*.sh files for training and Evaluation scripts

For label smoothing, replace Experiments/temperature_scaling_sst.py with Experiments/temperature_scaling_sst_smoothing.py; Experiments/error_bars_sst.py with Experiments/error_bars_sst_smoothing.py; treeLSTM/trainer.py with treeLSTM/trainer_smoothing.py and use pytorch_v1.0 


## For 20 Newsgroup:

Pytorch adaptation from [MMCE](https://github.com/aviralkumar2907/MMCE)

### Installations

Refer [MMCE](https://github.com/aviralkumar2907/MMCE)

Use virtualenv with Python 2.7.12, pytorch 0.4.1, tensorflow 1.4.1, keras 2.1.2

### Training and Evaluation:

Refer 20newsgroup_*.sh files for training and Evaluation scripts

## Pretrained models

All the pretrained models for all the datasets can be [downloaded from here](http://www.robots.ox.ac.uk/~viveka/focal_calibration/).
