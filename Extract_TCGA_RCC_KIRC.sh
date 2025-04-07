nohup env CUDA_VISIBLE_DEVICES=3 python  Extract_Features_TCGA_RCC.py\
 --data_path /home/graduate2024/data_TCGA/TCGA_RCC/TCGA-KIRC\
 > out_FeatTCGA_RCC_KIRC.log 2> error_FeatTCGA_RCC_KIRC.log &
