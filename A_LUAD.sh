nohup env CUDA_VISIBLE_DEVICES=3 python main.py \
--drop_out 0.25 \
--early_stopping \
--lr 2e-4 \
--k 10 \
--B 20 \
--results_dir /home/graduate2024/code/baseline/CLAM-master/results/LUAD_d1024_old2 \
--split_dir /home/graduate2024/code/baseline/CLAM-master/splits/LUAD_fengru_100 \
--exp_code LUAD_fengru_clam \
--weighted_sample --bag_loss ce --inst_loss svm \
--task LUAD_fengru \
--model_type clam_mb \
--log_data --subtyping \
--data_root_dir /home/graduate2024/data_new/TCGA_LUAD_Feature_d1024/all_features \
--embed_dim 1024 \
> out_LUAD_train_fengru_old2.log 2> error_LUAD_train_fengru_old2.log &

