CUDA_VISIBLE_DEVICES=0 python 20newsgroup_eval_error_bars.py \
--saved_model_names mmce_weighted_lamda_2.0_best.model \
--save-path Outputs/focalCalibration/nlp/20ng/ --temp 2.2 >> 20ng_mmce.txt

#CUDA_VISIBLE_DEVICES=0 python 20newsgroup_eval_error_bars.py \
#--saved_model_names mmce_weighted_lamda_8.0_best.model \
#--save-path Outputs/focalCalibration/nlp/20ng/ --temp 2.1 >> 20ng_error_bars.txt

CUDA_VISIBLE_DEVICES=0 python 20newsgroup_eval_error_bars.py \
--saved_model_names cross_entropy_best.model \
--save-path Outputs/focalCalibration/nlp/20ng/ --temp 3.4 >> 20ng_cross_entropy.txt

CUDA_VISIBLE_DEVICES=0 python 20newsgroup_eval_error_bars.py \
--saved_model_names brier_score_best.model \
--save-path Outputs/focalCalibration/nlp/20ng/ --temp 2.3 >> 20ng_brier_score.txt

CUDA_VISIBLE_DEVICES=0 python 20newsgroup_eval_error_bars.py \
--saved_model_names cross_entropy_smoothed_smoothing_0.05_best.model \
--save-path Outputs/focalCalibration/Label_Smoothing/NLP/Best_Models/ --temp 1.1 >> 20ng_ls_0.05.txt

CUDA_VISIBLE_DEVICES=0 python 20newsgroup_eval_error_bars.py \
--saved_model_names focal_loss_gamma_1.0_best.model   \
--save-path Outputs/focalCalibration/nlp/20ng/ --temp 2.6 >> 20ng_fl_g1.txt

CUDA_VISIBLE_DEVICES=0 python 20newsgroup_eval_error_bars.py \
--saved_model_names focal_loss_gamma_2.0_best.model   \
--save-path Outputs/focalCalibration/nlp/20ng/ --temp 1.6 >> 20ng_fl_g2.txt

CUDA_VISIBLE_DEVICES=0 python 20newsgroup_eval_error_bars.py \
--saved_model_names focal_loss_gamma_3.0_best.model   \
--save-path Outputs/focalCalibration/nlp/20ng/ --temp 1.5 >> 20ng_fl_g3.txt

CUDA_VISIBLE_DEVICES=0 python 20newsgroup_eval_error_bars.py \
--saved_model_names focal_loss_scheduled_gamma_5.0_3.0_1.0_best.model   \
--save-path Outputs/focalCalibration/nlp/20ng/ --temp 1.7 >> 20ng_fl_scheduled_g531.txt

CUDA_VISIBLE_DEVICES=0 python 20newsgroup_eval_error_bars.py \
--saved_model_names focal_loss_scheduled_gamma_5.0_3.0_2.0_best.model  \
--save-path Outputs/focalCalibration/nlp/20ng/ --temp 1.8 >> 20ng_fl_scheduled_g532.txt

CUDA_VISIBLE_DEVICES=0 python 20newsgroup_eval_error_bars.py \
--saved_model_names focal_loss_adaptive_gamma_3.0_best.model  \
--save-path Outputs/focalCalibration/nlp/20ng/ --temp 1.5 >> 20ng_fl_adaptive_g53.txt

CUDA_VISIBLE_DEVICES=0 python 20newsgroup_eval_error_bars.py \
--saved_model_names focal_loss_adaptive_gamma_2.0_best.model  \
--save-path Outputs/focalCalibration/nlp/20ng/ --temp 2.0 >> 20ng_fl_adaptive_g532.txt
