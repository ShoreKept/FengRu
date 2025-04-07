nohup env CUDA_VISIBLE_DEVICES=0 python main.py \
--drop_out 0.25 \
--early_stopping \
--lr 2e-4 \
--k 10 \
--B 15 \
--results_dir /home/graduate2024/code/baseline/CLAM-master/results/RCC_d1024_B15 \
--split_dir /home/graduate2024/code/baseline/CLAM-master/splits/RCC_fengru_100 \
--exp_code RCC_fengru_clam_B15 \
--weighted_sample --bag_loss ce --inst_loss svm \
--task RCC_fengru \
--model_type clam_mb \
--log_data --subtyping \
--data_root_dir /home/graduate2024/data_new/TCGA_RCC_Feature_d1024/all_features \
--embed_dim 1024 \
> out_RCC_train_fengru_B15.log 2> error_RCC_train_fengru_B15.log &

