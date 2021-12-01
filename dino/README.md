# Self-Supervised Vision Transformers with DINO
We modify the code from [DINO](https://github.com/facebookresearch/dino). You can refer to the original repo for more details.

## Training
Please set `--data_path`, `filter_path`, `output_dir` and `--keep_num` in the shell files. 
You can set `--keep_num` to 2090122 (50%) or 2508146 (60%) for the conditional training.
60% pre-training data achieves better performance, while 50% pre-training data makes a trade-off between the performance and the computational cost.

- Training ViT-S
```bash
python -W ignore -m torch.distributed.launch --nproc_per_node=8 main_dino.py \
--arch vit_small \
--data_path /home/michuan.lh/datasets/LUP \
--output_dir ./log/dino/lup/vit_small_full_lup \
--height 256 --width 128 \
--crop_height 128 --crop_width 64 \
--epochs 100 \
```

- Training ViT-S+ICS
```bash
python -W ignore -m torch.distributed.launch --nproc_per_node=8 main_dino.py \
--arch ours_vit_small \
--data_path /home/michuan.lh/datasets/LUP \
--output_dir ./log/dino/lup/vit_small_ics_full_lup \
--height 256 --width 128 \
--crop_height 128 --crop_width 64 \
--epochs 100 \

```

- Training ViT-S+ICS+CFS
```bash
python -W ignore -m torch.distributed.launch --nproc_per_node=8 main_dino.py \
--arch ours_vit_small \
--data_path /home/michuan.lh/datasets/LUP \
--filter_path /mnt1/michuan.lh/workspace/transreid_pytorch/save/cfs_list.pkl \
--keep_num 2508146 \
--output_dir /mnt1/michuan.lh/log/dino/lup_filter/open_source/vit_small_ics_cfs_lup \
--height 256 --width 128 \
--crop_height 128 --crop_width 64 \
--epochs 100 \
```

## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

## Citation
If you find this repository useful, please consider giving a star :star: and citation :t-rex::
```
@article{caron2021emerging,
  title={Emerging Properties in Self-Supervised Vision Transformers},
  author={Caron, Mathilde and Touvron, Hugo and Misra, Ishan and J\'egou, Herv\'e  and Mairal, Julien and Bojanowski, Piotr and Joulin, Armand},
  journal={arXiv preprint arXiv:2104.14294},
  year={2021}
}
```
