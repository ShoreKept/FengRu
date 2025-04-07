# nohup python create_patches_fp.py --source /home/graduate2024/data/zxb_output/Normal --save_dir /home/graduate2024/data/CLAM_output5/Normal --patch_size 256 --seg --patch --stitch > out1.log 2> error1.log &
# nohup python create_patches_fp.py --source /home/graduate2024/data/zxb_output/19DEL --save_dir /home/graduate2024/data/CLAM_output5/19DEL --patch_size 256 --seg --patch --stitch > out2.log 2> error2.log &
# nohup python create_patches_fp.py --source /home/graduate2024/data/zxb_output/L858R --save_dir /home/graduate2024/data/CLAM_output5/L858R --patch_size 256 --seg --patch --stitch > out3.log 2> error3.log &
# nohup python create_patches_fp.py --source /home/graduate2024/data/zxb_output/Other --save_dir /home/graduate2024/data/CLAM_output5/Other --patch_size 256 --seg --patch --stitch > out4.log 2> error4.log &
# nohup python create_patches_fp.py --source /home/graduate2024/data/zxb_output/Wild --save_dir /home/graduate2024/data/CLAM_output5/Wild --patch_size 256 --seg --patch --stitch > out5.log 2> error5.log &

# nohup env CUDA_VISIBLE_DEVICES=2 python extract_features_fp.py --data_h5_dir /home/graduate2024/data/CLAM_output5/19DEL --data_slide_dir /home/graduate2024/data/zxb_output/19DEL --csv_path /home/graduate2024/data/CLAM_output5/19DEL/process_list_autogen.csv --feat_dir /home/graduate2024/data/CLAM_features/19DEL --batch_size 512 --slide_ext .tiff --model_name  'conch_v1' > out_19DEL.log 2> error_19DEL.log &
# nohup env CUDA_VISIBLE_DEVICES=3 python extract_features_fp.py --data_h5_dir /home/graduate2024/data/CLAM_output5/L858R --data_slide_dir /home/graduate2024/data/zxb_output/L858R --csv_path /home/graduate2024/data/CLAM_output5/L858R/process_list_autogen.csv --feat_dir /home/graduate2024/data/CLAM_features/L858R --batch_size 512 --slide_ext .tiff --model_name  'conch_v1' > out_L858R.log 2> error_L858R.log &
# nohup env CUDA_VISIBLE_DEVICES=3 python extract_features_fp.py --data_h5_dir /home/graduate2024/data/CLAM_output5/Normal --data_slide_dir /home/graduate2024/data/zxb_output/Normal --csv_path /home/graduate2024/data/CLAM_output5/Normal/process_list_autogen.csv --feat_dir /home/graduate2024/data/CLAM_features/Normal --batch_size 512 --slide_ext .tiff --model_name  'conch_v1' > out_Normal.log 2> error_Normal.log &
# nohup env CUDA_VISIBLE_DEVICES=2 python extract_features_fp.py --data_h5_dir /home/graduate2024/data/CLAM_output5/Wild --data_slide_dir /home/graduate2024/data/zxb_output/Wild --csv_path /home/graduate2024/data/CLAM_output5/Wild/process_list_autogen.csv --feat_dir /home/graduate2024/data/CLAM_features/Wild --batch_size 512 --slide_ext .tiff --model_name  'conch_v1' > out_Wild.log 2> error_Wild.log &
# nohup env CUDA_VISIBLE_DEVICES=3 python extract_features_fp.py --data_h5_dir /home/graduate2024/data/CLAM_output5/Other --data_slide_dir /home/graduate2024/data/zxb_output/Other --csv_path /home/graduate2024/data/CLAM_output5/Other/process_list_autogen.csv --feat_dir /home/graduate2024/data/CLAM_features/Other --batch_size 512 --slide_ext .tiff --model_name  'conch_v1' > out_Other.log 2> error_Other.log &

# python create_splits_seq.py --task EGFR_subtyping --seed 1 --k 10
# python main.py --task EGFR_subtyping --seed 1 --k 10 --data_root_dir /home/graduate2024/data/CLAM_all_features

#训练：
# nohup env CUDA_VISIBLE_DEVICES=3 python main.py --drop_out 0.25 --early_stopping --lr 2e-4 --k 10 --exp_code EGFR_subtyping_CLAM_50 --weighted_sample --bag_loss svm  --task EGFR_subtyping --model_type clam_sb --log_data --subtyping --data_root_dir /home/graduate2024/data/CLAM_all_features --embed_dim 512 > out_EGFR_train.log 2> error_EGFR_train.log &
#不使用svm而使用交叉熵

#测试：
#nohup env CUDA_VISIBLE_DEVICES=2 python eval.py --k 10 --models_exp_code EGFR_subtyping_CLAM_50_s1 --save_exp_code EGFR_subtyping_CLAM_50_s1_cv --task EGFR_subtyping --model_type clam_sb --results_dir results --data_root_dir /home/graduate2024/data/CLAM_all_features --embed_dim 512 > out_EGFR_test.log 2> error_EGFR_test.log &

#CUDA_VISIBLE_DEVICES=3 python create_heatmaps.py --config config_template.yaml


# nohup env CUDA_VISIBLE_DEVICES=2 python eval_new.py --k 10 --models_exp_code EGFR_subtyping_CLAM_50_s1 --save_exp_code EGFR_subtyping_CLAM_50_s1_cv --task EGFR_subtyping --model_type clam_sb --results_dir results --data_root_dir /home/graduate2024/data/CLAM_all_features --embed_dim 512 > out_EGFR_test.log 2> error_EGFR_test.log &

#测试：
nohup env CUDA_VISIBLE_DEVICES=2 python extract_features_fp.py --data_h5_dir /home/graduate2024/data/CLAM_output5/19DEL --data_slide_dir /home/graduate2024/data/EGFR/19DEL --csv_path /home/graduate2024/data/CLAM_output5/19DEL/process_list_autogen.csv --feat_dir /home/graduate2024/data/CLAM_features/19DEL --batch_size 512 --slide_ext .tiff --model_name  'conch_v1' > out_19DEL_test.log 2> error_19DEL_test.log &
