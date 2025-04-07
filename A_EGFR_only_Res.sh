nohup env CUDA_VISIBLE_DEVICES=1 python main.py \
--drop_out 0.25 \
--early_stopping \
--lr 2e-4 \
--k 10 \
--results_dir /home/graduate2024/code/baseline/CLAM-master/results/EGFR_only_Res3_UNI_d1024 \
--split_dir /home/graduate2024/code/baseline/CLAM-master/splits/EGFR_Five_subtyping_fengru_100 \
--exp_code EGFR_only_Res3_fengru \
--weighted_sample --bag_loss ce --inst_loss svm \
--task EGFR_Five_subtyping_fengru \
--model_type clam_mb \
--log_data --subtyping \
--data_root_dir /home/graduate2024/data_new/Extracted_Features_fengru_UNI_d1024/all_features \
--embed_dim 1024 \
> out_EGFR_train_only_Res3.log 2> error_EGFR_train_only_Res3.log &