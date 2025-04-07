#训练：
nohup env CUDA_VISIBLE_DEVICES=3 python main.py --drop_out 0.25 --early_stopping --lr 2e-4 --k 10 --exp_code EGFR_subtyping_CLAM_2_classes --weighted_sample --bag_loss svm  --task EGFR_subtyping --model_type clam_mb --log_data --subtyping --data_root_dir /home/graduate2024/data_new/Features_mine --embed_dim 512 > out_EGFR_train.log 2> error_EGFR_train.log &

#测试：
# nohup env CUDA_VISIBLE_DEVICES=3 python eval.py --k 10 --models_exp_code EGFR_subtyping_CLAM_new_s1 --save_exp_code EGFR_subtyping_CLAM_new_cv --task EGFR_subtyping --model_type clam_mb --results_dir results --data_root_dir /home/graduate2024/data_new/Features_mine --embed_dim 512 > out_EGFR_test_zxb.log 2> error_EGFR_test_zxb.log &

