nohup env CUDA_VISIBLE_DEVICES=3 python main.py \
--drop_out 0.25 \
--early_stopping \
--lr 2e-4 \
--k 10 \
--results_dir /home/graduate2024/code/baseline/CLAM-master/results/LUAD_only_Res2x2_UNI_d1024 \
--split_dir /home/graduate2024/code/baseline/CLAM-master/splits/LUAD_fengru_100 \
--exp_code LUAD_only_Res2x2_fengru \
--weighted_sample --bag_loss ce --inst_loss svm \
--task LUAD_fengru \
--model_type clam_mb \
--log_data --subtyping \
--data_root_dir /home/graduate2024/data_new/TCGA_LUAD_Feature_d1024/all_features \
--embed_dim 1024 \
> out_LUAD_train_only_Res2x2.log 2> error_LUAD_train_only_Res2x2.log &