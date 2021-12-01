python -W ignore -m torch.distributed.launch --nproc_per_node=8 main_dino.py \
--arch ours_vit_small \
--data_path /home/michuan.lh/datasets/LUP \
--filter_path ../../cfs_list.pkl \
--keep_num 2508146 \
--output_dir ./log/dino//vit_small_ics_cfs_lup \
--height 256 --width 128 \
--crop_height 128 --crop_width 64 \
--epochs 100 \
