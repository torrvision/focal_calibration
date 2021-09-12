CUDA_VISIBLE_DEVICES=0 python 20newsgroup_eval.py --saved_model_names cross_entropy_smoothed_smoothing_0.05_best.model cross_entropy_smoothed_smoothing_0.1_best.model -ta --save-path Outputs/focalCalibration/Label_Smoothing/NLP/Best_Models/

CUDA_VISIBLE_DEVICES=0 python 20newsgroup_eval.py \
--saved_model_names mmce_weighted_lamda_8.0_best.model \
mmce_weighted_lamda_2.0_best.model \
cross_entropy_best.model \
brier_score_best.model \
focal_loss_gamma_1.0_best.model \
focal_loss_gamma_2.0_best.model \
focal_loss_gamma_3.0_best.model \
focal_loss_scheduled_gamma_5.0_3.0_1.0_best.model \
focal_loss_scheduled_gamma_5.0_3.0_2.0_best.model \
focal_loss_adaptive_gamma_3.0_best.model \
focal_loss_adaptive_gamma_2.0_best.model -ta \
--save-path Outputs/focalCalibration/nlp/20ng/


####

CUDA_VISIBLE_DEVICES=0 python 20newsgroup_eval.py \
--saved_model_names mmce_weighted_lamda_8.0_best.model \
mmce_weighted_lamda_2.0_best.model \
cross_entropy_best.model \
brier_score_best.model \
focal_loss_gamma_1.0_best.model \
focal_loss_gamma_2.0_best.model \
focal_loss_gamma_3.0_best.model \
focal_loss_scheduled_gamma_5.0_3.0_1.0_best.model \
focal_loss_scheduled_gamma_5.0_3.0_2.0_best.model \
focal_loss_adaptive_gamma_3.0_best.model \
focal_loss_adaptive_gamma_2.0_best.model -ce

CUDA_VISIBLE_DEVICES=0 python 20newsgroup_eval.py \
--saved_model_names mmce_weighted_lamda_8.0_best.model \
mmce_weighted_lamda_2.0_best.model \
cross_entropy_best.model \
brier_score_best.model \
focal_loss_gamma_1.0_best.model \
focal_loss_gamma_2.0_best.model \
focal_loss_gamma_3.0_best.model \
focal_loss_scheduled_gamma_5.0_3.0_1.0_best.model \
focal_loss_scheduled_gamma_5.0_3.0_2.0_best.model \
focal_loss_adaptive_gamma_3.0_best.model \
focal_loss_adaptive_gamma_2.0_best.model -cae

CUDA_VISIBLE_DEVICES=0 python 20newsgroup_eval.py \
--saved_model_names mmce_weighted_lamda_8.0_best.model \
mmce_weighted_lamda_2.0_best.model \
cross_entropy_best.model \
brier_score_best.model \
focal_loss_gamma_1.0_best.model \
focal_loss_gamma_2.0_best.model \
focal_loss_gamma_3.0_best.model \
focal_loss_scheduled_gamma_5.0_3.0_1.0_best.model \
focal_loss_scheduled_gamma_5.0_3.0_2.0_best.model \
focal_loss_adaptive_gamma_3.0_best.model \
focal_loss_adaptive_gamma_2.0_best.model -tn

CUDA_VISIBLE_DEVICES=0 python 20newsgroup_eval.py \
--saved_model_names mmce_weighted_lamda_8.0_best.model \
mmce_weighted_lamda_2.0_best.model \
cross_entropy_best.model \
brier_score_best.model \
focal_loss_gamma_1.0_best.model \
focal_loss_gamma_2.0_best.model \
focal_loss_gamma_3.0_best.model \
focal_loss_scheduled_gamma_5.0_3.0_1.0_best.model \
focal_loss_scheduled_gamma_5.0_3.0_2.0_best.model \
focal_loss_adaptive_gamma_3.0_best.model \
focal_loss_adaptive_gamma_2.0_best.model -cn


