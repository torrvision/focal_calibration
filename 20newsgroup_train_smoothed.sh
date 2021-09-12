CUDA_VISIBLE_DEVICES=0 python 20newsgroup_train.py --num_epochs 50 \
--loss cross_entropy_smoothed --smoothing 0.05 > script_output_cross_entropy_smoothed_0.05.txt

CUDA_VISIBLE_DEVICES=0 python 20newsgroup_train.py --num_epochs 50 \
--loss cross_entropy_smoothed --smoothing 0.1 > script_output_cross_entropy_smoothed_0.1.txt