nohup env CUDA_VISIBLE_DEVICES=2 python Extract_Features_fengru_Pos.py \
--feat_dir /home/graduate2024/data_new/Extracted_Features_fengru_UNI_d1024 \
--result_dir /home/graduate2024/data_new/FeaturesWithPosition \
> out_FeaturePositon.log 2> error_FeaturePositon.log &