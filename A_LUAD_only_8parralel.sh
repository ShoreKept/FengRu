nohup env CUDA_VISIBLE_DEVICES=1 python main.py \
--drop_out 0.25 \
--early_stopping \
--lr 2e-4 \
--k 10 \
--results_dir /home/graduate2024/code/baseline/CLAM-master/results/LUAD_only_8parralel_UNI_d1024 \
--split_dir /home/graduate2024/code/baseline/CLAM-master/splits/LUAD_fengru_100 \
--exp_code LUAD_only_8parralel_fengru \
--weighted_sample --bag_loss ce --inst_loss svm \
--task LUAD_fengru \
--model_type clam_mb \
--log_data --subtyping \
--data_root_dir /home/graduate2024/data_new/TCGA_LUAD_Feature_d1024/all_features \
--embed_dim 1024 \
> out_LUAD_train_only_8parralel.log 2> error_LUAD_train_only_8parralel.log &