python -W ignore -m torch.distributed.launch --nproc_per_node=8 main_dino.py \
--arch vit_small \
--data_path /home/michuan.lh/datasets/LUP \
--output_dir ./log/dino/lup/vit_small_full_lup \
--height 256 --width 128 \
--crop_height 128 --crop_width 64 \
--epochs 100 \
