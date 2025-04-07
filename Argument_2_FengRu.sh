#为了处理五分类问题
#提特征
# nohup env CUDA_VISIBLE_DEVICES=3 python Extract_Features_fengru_zxb.py  > out_fengru_zxb.log 2> error_fengru_zxb.log &

# python create_split_fengru_zxb.py --task EGFR_Five_subtyping_fengru --seed 1 --k 10  #按照CLAM原代码分离数据集


#训练：
# nohup env CUDA_VISIBLE_DEVICES=2 python main.py --drop_out 0.25 --early_stopping --lr 2e-4 --k 10 --exp_code EGFR_Five_subtyping_fengru --weighted_sample --bag_loss ce --inst_loss svm --task EGFR_Five_subtyping_fengru --model_type clam_mb --log_data --subtyping --data_root_dir /home/graduate2024/data_new/Extracted_Features_fengru/all_features --embed_dim 512 > out_EGFR_train_fengru.log 2> error_EGFR_train_fengru.log &

#测试：
# nohup env CUDA_VISIBLE_DEVICES=1 python eval_new_fengru.py --k 10 --models_exp_code EGFR_Five_subtyping_fengru_s1_old9 --save_exp_code EGFR_Five_subtyping_fengru --task EGFR_Five_subtyping_fengru --model_type clam_mb --results_dir results --data_root_dir /home/graduate2024/data_new/Extracted_Features_fengru/all_features --embed_dim 512 > out_EGFR_test_fengru.log 2> error_EGFR_test_fengru.log  &

#1024维的测试：
# nohup env CUDA_VISIBLE_DEVICES=1 python eval_new_fengru.py --k 10 --models_exp_code EGFR_Four_UNI_d1024/EGFR_Four_subtyping_fengru_s1 --save_exp_code EGFR_Four_UNI_d1024 --task EGFR_Four_subtyping_fengru --model_type clam_mb --results_dir results --data_root_dir /home/graduate2024/data_new/Extracted_Features_fengru_UNI_d1024/all_features --embed_dim 1024 > out_EGFR_test_fengru.log 2> error_EGFR_test_fengru.log  &
# nohup env CUDA_VISIBLE_DEVICES=2 python eval_new_fengru.py --k 10 --models_exp_code EGFR_Five_UNI_d1024_old3/EGFR_Five_subtyping_fengru_s1 --save_exp_code EGFR_Five_UNI_d1024 --task EGFR_Five_subtyping_fengru --model_type clam_mb --results_dir results --data_root_dir /home/graduate2024/data_new/Extracted_Features_fengru_UNI_d1024/all_features --embed_dim 1024 > out_EGFR_test_fengru_1024.log 2> error_EGFR_test_fengru_1024.log  &

#RCC上的命令
python create_split_fengru_zxb.py --task LUAD_fengru --seed 1 --k 10  #按照CLAM原代码分离数据集