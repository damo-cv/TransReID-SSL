python -W ignore -m torch.distributed.launch --nproc_per_node=8 main_dino.py \
--arch ours_vit_base \
--data_path /home/michuan.lh/datasets/LUP \
--output_dir ../../log/dino/vit-b_ics_cfs_lup \
--filter_path ../../cfs_list.pkl \
--keep_num 2508146 \
--height 256 --width 128 \
--crop_height 128 --crop_width 64 \
--epochs 100 \
