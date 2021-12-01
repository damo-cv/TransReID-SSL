# ViT-S+ICS
CUDA_VISIBLE_DEVICES=2,3 python examples/cluster_contrast_train_usl.py -b 256 -a vit_small -d msmt17 --iters 200 --eps 0.7 --self-norm --use-hard --hw-ratio 2 --num-instances 8 --conv-stem -pp ../../log/transreid/market/vit_small_ics_cfs_lup/transformer_120.pth --logs-dir ../../log/cluster_contrast_reid/market2msmt/vit_small_ics_cfs_lup 

# VIT-S
# CUDA_VISIBLE_DEVICES=0,1 python examples/cluster_contrast_train_usl.py -b 256 -a vit_small -d msmt17 --iters 200 --eps 0.7 --self-norm --use-hard --hw-ratio 2 --num-instances 8 -pp ../../log/transreid/market/vit_small_cfs_lup/transformer_120.pth --logs-dir ../../log/cluster_contrast_reid/market2msmt/vit_small_cfs_lup 
