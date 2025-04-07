#未执行：
# nohup env CUDA_VISIBLE_DEVICES=2 python Extract_Features_fengru.py --data_h5_dir /home/graduate2024/data_new/Patches_result_fengru/19DEL  --data_slide_dir /home/graduate2024/data/zxb_output/19DEL --csv_path /home/graduate2024/data/CLAM_output5/19DEL/process_list_autogen.csv --feat_dir /home/graduate2024/data_new/Extracted_Features_fengru/19DEL --  --batch_size 512 --slide_ext .tiff --model_name  'conch_v1' > out_19DEL.log 2> error_19DEL.log &
# nohup env CUDA_VISIBLE_DEVICES=3 python Extract_Features_fengru.py --data_h5_dir /home/graduate2024/data_new/Patches_result_fengru/L858R  --data_slide_dir /home/graduate2024/data/zxb_output/L858R --csv_path /home/graduate2024/data/CLAM_output5/L858R/process_list_autogen.csv --feat_dir /home/graduate2024/data_new/Extracted_Features_fengru/L858R --batch_size 512 --slide_ext .tiff --model_name  'conch_v1' > out_L858R.log 2> error_L858R.log &
# nohup env CUDA_VISIBLE_DEVICES=3 python Extract_Features_fengru.py --data_h5_dir /home/graduate2024/data_new/Patches_result_fengru/Normal --data_slide_dir /home/graduate2024/data/zxb_output/Normal --csv_path /home/graduate2024/data/CLAM_output5/Normal/process_list_autogen.csv --feat_dir /home/graduate2024/data_new/Extracted_Features_fengru/Normal --batch_size 512 --slide_ext .tiff --model_name  'conch_v1' > out_Normal.log 2> error_Normal.log &
# nohup env CUDA_VISIBLE_DEVICES=2 python Extract_Features_fengru.py --data_h5_dir /home/graduate2024/data_new/Patches_result_fengru/Wild   --data_slide_dir /home/graduate2024/data/zxb_output/Wild --csv_path /home/graduate2024/data/CLAM_output5/Wild/process_list_autogen.csv --feat_dir /home/graduate2024/data_new/Extracted_Features_fengru/Wild --batch_size 512 --slide_ext .tiff --model_name  'conch_v1' > out_Wild.log 2> error_Wild.log &
# nohup env CUDA_VISIBLE_DEVICES=3 python Extract_Features_fengru.py --data_h5_dir /home/graduate2024/data_new/Patches_result_fengru/Other  --data_slide_dir /home/graduate2024/data/zxb_output/Other --csv_path /home/graduate2024/data/CLAM_output5/Other/process_list_autogen.csv --feat_dir /home/graduate2024/data_new/Extracted_Features_fengru/Other --batch_size 512 --slide_ext .tiff --model_name  'conch_v1' > out_Other.log 2> error_Other.log &

#提特征
# nohup env CUDA_VISIBLE_DEVICES=3 python Extract_Features_fengru.py  > out_fengru.log 2> error_fengru.log &

# python create_split_fengru_zxb.py --task EGFR_Four_subtyping_fengru --seed 1 --k 10  #按照CLAM原代码分离数据集


#训练：
# nohup env CUDA_VISIBLE_DEVICES=1 python main.py --drop_out 0.25 --early_stopping --lr 2e-4 --k 10 --exp_code EGFR_Four_subtyping_fengru --weighted_sample --bag_loss ce --inst_loss svm --task EGFR_Four_subtyping_fengru --model_type clam_mb --log_data --subtyping --data_root_dir /home/graduate2024/data_new/Extracted_Features_fengru/all_features --embed_dim 512 > out_EGFR_train_fengru.log 2> error_EGFR_train_fengru.log &

#测试：
nohup env CUDA_VISIBLE_DEVICES=2 python eval_new_fengru.py --k 10 --models_exp_code EGFR_Four_subtyping_fengru_s1_old4 --save_exp_code EGFR_Four_subtyping_fengru --task EGFR_Four_subtyping_fengru --model_type clam_mb --results_dir results --data_root_dir /home/graduate2024/data_new/Extracted_Features_fengru/all_features --embed_dim 512 > out_EGFR_test_fengru.log 2> error_EGFR_test_fengru.log  &
