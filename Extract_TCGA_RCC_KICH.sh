nohup env CUDA_VISIBLE_DEVICES=0 python  Extract_Features_TCGA_RCC.py\
 --data_path /home/graduate2024/data_TCGA/TCGA_RCC/TCGA-KICH\
 > out_FeatTCGA_RCC_KICH.log 2> error_FeatTCGA_RCC_KICH.log &
