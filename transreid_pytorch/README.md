# TransReID: Transformer-based Object Re-Identification [[pdf]](https://openaccess.thecvf.com/content/ICCV2021/papers/He_TransReID_Transformer-Based_Object_Re-Identification_ICCV_2021_paper.pdf)
We modify the code from [TransReID](https://github.com/damo-cv/TransReID). You can refer to the original repo for more details.

## Requirements

### Installation

```bash
(we use /torch 1.7.1 /torchvision 0.8.2 /timm 0.3.4 /cuda 10.1 / 16G or 32G V100 for training and evaluation.
Note that we use torch.cuda.amp to accelerate speed of training which requires pytorch >=1.6)
```
### Prepare Pre-trained Models 
Please download the pre-trained models and put them into your custom file folder.

## Training

We utilize 1  GPU for training. Please modify the `MODEL.PRETRAIN_PATH` and `OUTPUT_DIR` in the config file.

```bash
python train.py --config_file configs/market/vit_small_ics.yml
```

You also can speed up training with 4-GPUs training. But the performance may be reduced by 0.1~0.2% mAP.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 66666 train.py --config_file configs/market/vit_small_ics_ddp.yml
```

## Evaluation

```bash
python test.py --config_file 'choose which config to test' MODEL.DEVICE_ID "('your device id')" TEST.WEIGHT "('your path of trained checkpoints')"
```

**Some examples:**

```bash
# Market
python test.py --config_file configs/market/vit_small_ics.yml MODEL.DEVICE_ID "('0')"  TEST.WEIGHT 'XXXX/transformer_120.pth'
```

## Citation

If you find this code useful for your research, please cite our paper

```
@InProceedings{He_2021_ICCV,
    author    = {He, Shuting and Luo, Hao and Wang, Pichao and Wang, Fan and Li, Hao and Jiang, Wei},
    title     = {TransReID: Transformer-Based Object Re-Identification},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {15013-15022}
}
```
