nohup env CUDA_VISIBLE_DEVICES=1 python  ExtractFeature_TCGA_LUAD.py \
--begin 0 \
> out_FeatTCGA_LUAD0.log 2> error_FeatTCGA_LUAD0.log &
nohup env CUDA_VISIBLE_DEVICES=2 python  ExtractFeature_TCGA_LUAD.py \
--begin 1 \
> out_FeatTCGA_LUAD1.log 2> error_FeatTCGA_LUAD1.log &
nohup env CUDA_VISIBLE_DEVICES=3 python  ExtractFeature_TCGA_LUAD.py \
--begin 2 \
> out_FeatTCGA_LUAD2.log 2> error_FeatTCGA_LUAD2.log &  

