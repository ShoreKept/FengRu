nohup env CUDA_VISIBLE_DEVICES=2 python main.py \
--drop_out 0.25 \
--early_stopping \
--lr 2e-4 \
--k 10 \
--results_dir /home/graduate2024/code/zxb/CLAM-master/results/EGFR_Four_UNI_d1024 \
--exp_code EGFR_Four_subtyping_fengru \
--weighted_sample --bag_loss ce --inst_loss svm \
--task EGFR_Four_subtyping_fengru \
--model_type clam_mb \
--log_data --subtyping --data_root_dir /home/graduate2024/data_new/Extracted_Features_fengru_UNI_d1024/all_features \
--embed_dim 1024 \
> out_EGFR_train_fengru_UNI_d1024_Four.log 2> error_EGFR_train_fengru_UNI_d1024_Four.log &