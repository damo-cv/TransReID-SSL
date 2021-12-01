#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import pickle as pkl
import sys
import torch

if __name__ == "__main__":
    input = sys.argv[1]
    #  input = '/mnt1/michuan.lh/log/moco/sysu_200ep/ckpt_0020.pth'
    obj = torch.load(input, map_location="cpu")
    #  obj = obj["state_dict"]
    obj = obj["teacher"]
    

    newmodel = {}
    for k, v in obj.items():
        if k.startswith('module'):
            k = k.replace("module.", "")
        if not k.startswith("backbone."):
            continue
        old_k = k
        k = k.replace("backbone.", "")
        print(old_k, "->", k)
        newmodel[k] = v
    torch.save(newmodel,sys.argv[2])
