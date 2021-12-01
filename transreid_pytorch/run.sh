#Single GPU
python train.py --config_file configs/market/vit_small_ics.yml

# Multiple GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 66666 train.py --config_file configs/market/vit_small_ics_ddp.yml

