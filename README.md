![Python >=3.6](https://img.shields.io/badge/Python->=3.6-yellow.svg)
![PyTorch >=1.7](https://img.shields.io/badge/PyTorch->=1.7-blue.svg)

# Self-Supervised Pre-Training for Transformer-Based Person Re-Identification [[pdf]](https://arxiv.org/pdf/2111.12084)
The *official* repository for [Self-Supervised Pre-Training for Transformer-Based Person Re-Identification](https://arxiv.org/pdf/2111.12084).

## Requirements

### Installation
```bash
pip install -r requirements.txt
```
We recommend to use /torch=1.7.1 /torchvision=0.8.2 /timm=0.3.4 /cuda>10.1 /faiss-gpu=1.7.1/ 16G or 32G V100 for training and evaluation. If you find some packages are missing, please install them manually.
You can refer to [DINO](https://github.com/facebookresearch/dino), [TransReID](https://github.com/damo-cv/TransReID) and [cluster-contrast-reid](https://github.com/alibaba/cluster-contrast-reid) to install the environment of pre-training, supervised ReID and unsupervised ReID, respectively.

### Prepare Datasets

```bash
mkdir data
```

Download the datasets:
- [Market-1501](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view)
- [MSMT17](https://arxiv.org/abs/1711.08565)
- [LUPerson](https://github.com/DengpanFu/LUPerson). We don't have the copyright of the LUPerson dataset. Please contact authors of LUPerson to get this dataset.
- You can download the file list ordered by the CFS score for the LUPerson. [[CFS_list.pkl]](https://drive.google.com/file/d/1D6RaiOv3F2WSABYfQB1Aa88mwGoVNa3k/view?usp=sharing)

Then unzip them and rename them under the directory like

```
data
├── market1501
│   └── bounding_box_train
│   └── bounding_box_test
│   └── ..
├── MSMT17
│   └── train
│   └── test
│   └── ..
└── LUP
    └── images 
    └── CFS_list.pkl 
```

## Pre-trained Models
| Model         | Download |
| :------:      | :------: |
| ViT-S/16      | [link](https://drive.google.com/file/d/1ODxA7mJv17UfzwfXtY9dTWNsYghoNWGB/view?usp=sharing) |
| ViT-S/16+ICS  | [link](https://drive.google.com/file/d/18FL9JaJNlo15-UksalcJRXX-0dgo4Mz4/view?usp=sharing) |
| ViT-B/16+ICS  | [link](https://drive.google.com/file/d/1ZFMCBZ-lNFMeBD5K8PtJYJfYEk5D9isd/view?usp=sharing) |

Please download pre-trained models and put them into your custom file path.

## ReID performance

We have reproduced the performance to verify the reproducibility. The reproduced results may have a gap of about 0.1~0.2% with the numbers in the paper.

### Supervised ReID

##### Market-1501
| Model         | Image Size|Paper | Reproduce | Download |
| :------:      | :------: |:------: | :------: |:------: |
| ViT-S/16      | 256*128 |91.0/96.0 | 91.2/95.8|[model](https://drive.google.com/file/d/1lED8sKkiFAHp5LEzfhUSmF_Yh-7kKFsP/view?usp=sharing) / [log](https://drive.google.com/file/d/1jcsYcglLZJragtlpA1NICpxsoFpcs9Nc/view?usp=sharing)|
| ViT-S/16+ICS  | 256*128 |91.3/96.2 | 91.4/96.2|[model](https://drive.google.com/file/d/1tdO01aYtckVP3nQZm-cpSSPPrxODCrmB/view?usp=sharing) / [log](https://drive.google.com/file/d/1HP_giuY6eVoXrPS8dEyX6ZxDuv_wipff/view?usp=sharing)|
| ViT-B/16+ICS  | 384*128 |93.2/96.7 | 93.1/96.6|[model](https://drive.google.com/file/d/1wELRg_fCrgYCD3A3kUU7KJd-D_YSknQ_/view?usp=sharing) / [log](https://drive.google.com/file/d/1rcZan0_ov4V3nm-8AGvKjiN8kSVk_PP7/view?usp=sharing)|

##### MSMT17
| Model         | Image Size|Paper | Reproduce | Download |
| :------:      | :------: |:------: | :------: |:------: |
| ViT-S/16      | 256*128 |66.1/84.6 | 66.3/84.8|[model](https://drive.google.com/file/d/11KBWzbgyx73ejiS3Sn1NNDPN_34huQwP/view?usp=sharing) / [log](https://drive.google.com/file/d/1-W864ht66MZf-CqBeKIfgT12LqK9jcsU/view?usp=sharing)|
| ViT-S/16+ICS  | 256*128 |68.1/86.1 | 68.3/86.2|[model](https://drive.google.com/file/d/1taqdhKGunBTGLTXve17UH_zfzzMjMqBb/view?usp=sharing) / [log](https://drive.google.com/file/d/11UCRK9sJsDhDTLpIiMsZf7uKTqGaXvDn/view?usp=sharing)|
| ViT-B/16+ICS  | 384*128 |75.0/89.6 | 75.1/89.6|[model](https://drive.google.com/file/d/1m5a2vG6Se2K420g3oF4Zvu6m_zfWh-X4/view?usp=sharing) / [log](https://drive.google.com/file/d/1JSr1OI7mayTrdcM8g3ZKrsC_NHmaM_oW/view?usp=sharing)|

### USL ReID

##### Market-1501
| Model         | Image Size|Paper | Reproduce | Download |
| :------:      | :------: |:------: | :------: |:------: |
| ViT-S/16      | 256*128 |88.2/94.2 | 88.4/94.6|[model](https://drive.google.com/file/d/1lB7OS0hmYUbgUSqqKjULX_cYtiOVHi0j/view?usp=sharing) / [log](https://drive.google.com/file/d/1j3kWDYJftMTDBMmXrglJtZzDLiV6NHFu/view?usp=sharing)|
| ViT-S/16+ICS  | 256*128 |89.6/95.3 | 89.5/95.3|[model](https://drive.google.com/file/d/1AcZSbEz4iDI6pG2T0GWpqZdefAUJ4oRo/view?usp=sharing) / [log](https://drive.google.com/file/d/1sCQoQJ1l8n94TnFOGuVPZxS1rWHJAetn/view?usp=sharing)|

##### MSMT17
| Model         | Image Size|Paper | Reproduce | Download |
| :------:      | :------: |:------: | :------: |:------: |
| ViT-S/16      | 256*128 |40.9/66.4 | 40.9/66.4|[model](https://drive.google.com/file/d/1YJDixKh75Igmj-euzITzhcNNBdhNprMl/view?usp=sharing) / [log](https://drive.google.com/file/d/1F2ZU4vQv-wUzHNKxYgiJGfWPb_r1m_we/view?usp=sharing)|
| ViT-S/16+ICS  | 256*128 |50.6/75.0 | 50.6/75.0|[model](https://drive.google.com/file/d/1Y32B1XcZp5JvavEf4QqbM7BYEsYCh9yq/view?usp=sharing) / [log](https://drive.google.com/file/d/1_xolAYpJtOMyGsD6apPuTcJWFF569ziv/view?usp=sharing)|

### UDA ReID

##### MSMT2Market
| Model         | Image Size|Paper | Reproduce | Download |
| :------:      | :------: |:------: | :------: |:------: |
| ViT-S/16      | 256*128 |89.4/95.4 | 89.2/95.3|[model](https://drive.google.com/file/d/1W3SqChw4hcr0I52zwSODqEBJ-a_RrmoG/view?usp=sharing) / [log](https://drive.google.com/file/d/17c0HuchH9L-TONskOlqA_KnYORDPUwLU/view?usp=sharing)|
| ViT-S/16+ICS  | 256*128 |89.9/95.5 | 89.9/95.4|[model](https://drive.google.com/file/d/1DYnLuydsjZhEgpubrD2xsQqD6gXuI90V/view?usp=sharing) / [log](https://drive.google.com/file/d/1l7wCLQGnAFuM4TkJwTWO6DiO8lQC-f2i/view?usp=sharing)|

##### Market2MSMT
| Model         | Image Size|Paper | Reproduce | Download |
| :------:      | :------: |:------: | :------: |:------: |
| ViT-S/16      | 256*128 |47.4/70.8 | 47.7/71.2|[model](https://drive.google.com/file/d/1FgdyA_Vi0TsL0EKOYLy27RyEiQ1bd2hw/view?usp=sharing) / [log](https://drive.google.com/file/d/1JAhYdUtGxxJ2zA1IFjnHIVTixRPOJ3D5/view?usp=sharing)|
| ViT-S/16+ICS  | 256*128 |57.8/79.5 | 57.8/79.4|[model](https://drive.google.com/file/d/1yHKnhK2CYMbCceeO6vNBIKIMPUTY4VbS/view?usp=sharing) / [log](https://drive.google.com/file/d/1rkDWQrHKzddb6oHAKdM6L5kP20CXtQSd/view?usp=sharing)|

## Acknowledgment
Our implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.

[LUPerson](https://github.com/DengpanFu/LUPerson), [DINO](https://github.com/facebookresearch/dino), [TransReID](https://github.com/damo-cv/TransReID), [cluster-contrast-reid](https://github.com/alibaba/cluster-contrast-reid).

## Citation

If you find this code useful for your research, please cite our paper

```
@article{luo2021self,
  title={Self-Supervised Pre-Training for Transformer-Based Person Re-Identification},
  author={Luo, Hao and Wang, Pichao and Xu, Yi and Ding, Feng and Zhou, Yanxin and Wang, Fan and Li, Hao and Jin, Rong},
  journal={arXiv preprint arXiv:2111.12084},
  year={2021}
}
```

## Contact

If you have any question, please feel free to contact us. E-mail: [michuan.lh@alibaba-inc.com](mailto:michuan.lh@alibaba-inc.com) or [haoluocsc@zju.edu.cn](mailto:haoluocsc@zju.edu.cn)
