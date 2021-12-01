# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re
import os
import os.path as osp

from .bases import BaseImageDataset
from collections import defaultdict
class MM(BaseImageDataset):
    def __init__(self, root='', verbose=True, pid_begin = 0, **kwargs):
        super(MM, self).__init__()
        self.dataset_dir = osp.join(root, 'market1501')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        market_dir = '/home/michuan.lh/datasets/market1501/bounding_box_train'
        msmt_dir = '/home/michuan.lh/datasets/MSMT17/train'
        train = self.process_msmt(msmt_dir)
        train.extend(self.process_label(market_dir,b_pid=1041,b_camid=15))

        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> MM loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)


    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in sorted(img_paths):
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]

            dataset.append((img_path, pid, camid, 1))
        return dataset

    def process_label(self, root_dir, b_pid=0, b_camid=0):
        img_paths = os.listdir(root_dir)
        pattern = re.compile(r'([-\d]+)_c(\d)')
        pid_container = set()
        camid_container = set()
        EXTs = ('.jpg', '.png', '.jpeg', '.bmp', '.ppm')
        for img_path in img_paths:
            if os.path.splitext(img_path)[-1] not in EXTs: continue
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
            camid_container.add(camid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            if os.path.splitext(img_path)[-1] not in EXTs: continue
            pid, camid = map(int, pattern.search(img_path).groups())
            camid -= 1  # index starts from 0
            if pid == -1: continue  # junk images are just ignored
            pid = pid2label[pid]
            dataset.append((os.path.join(root_dir,img_path), b_pid + pid, b_camid+camid, 1))
        return dataset

    def process_msmt(self, msmt_dir):
        list_path = os.path.join(msmt_dir,'../list_train.txt')
        with open(list_path, 'r') as txt:
            lines = txt.readlines()
        dataset = []
        pid_container = set()
        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid)  # no need to relabel
            camid = int(img_path.split('_')[2])
            img_path = os.path.join(msmt_dir, img_path)
            dataset.append((img_path, pid, camid-1, 1))
        return dataset


