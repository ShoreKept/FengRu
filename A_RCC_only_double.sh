nohup env CUDA_VISIBLE_DEVICES=0 python main.py \
--drop_out 0.25 \
--early_stopping \
--lr 2e-4 \
--k 10 \
--results_dir /home/graduate2024/code/baseline/CLAM-master/results/RCC_only_double_UNI_d1024 \
--split_dir /home/graduate2024/code/zxb/CLAM-master/splits/RCC_fengru_100 \
--exp_code RCC_only_double_fengru \
--weighted_sample --bag_loss ce --inst_loss svm \
--task RCC_fengru \
--model_type clam_mb \
--log_data --subtyping \
--data_root_dir /home/graduate2024/data_new/TCGA_RCC_Feature_d1024/all_features \
--embed_dim 1024 \
> out_RCC_train_only_double.log 2> error_RCC_train_only_double.log &