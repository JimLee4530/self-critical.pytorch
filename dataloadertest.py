from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
import os
import numpy as np
import random

import torch
import torch.utils.data as data

import multiprocessing

split_path = {
    'train': 'train_20170902',
    'val': 'validation_20170910',
    'test': 'test_b_20171120',
}

def get_npy_data(opt, ix, fc_file, att_file):
    # fc_feats = np.load(att_file)['x']
    fc_feats = np.zeros((opt.att_feat_size), dtype=np.double)
    att_feats = np.zeros((100, opt.att_feat_size), dtype=np.double)
    
    t_att_dict = np.load(att_file)
    t_att_feats = t_att_dict['x']
    num_bbox = t_att_dict['num_bbox']
    t_att_feats = np.transpose(t_att_feats, (1,0))
    att_feats[:num_bbox, :] = t_att_feats

    return (fc_feats, att_feats, num_bbox, ix)

class DataLoader(data.Dataset):

    def get_vocab(self):
        return self.ix_to_word

    def __init__(self, opt):
        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.seq_per_img = opt.seq_per_img

        # # open the hdf5 file
        print('DataLoader loading h5 file: ', opt.input_fc_dir, opt.input_att_dir)
        # self.h5_label_file = h5py.File(self.opt.input_label_h5, 'r', driver='core')

        print('DataLoader loading json file: ', opt.input_json)
        self.info = json.load(open(self.opt.input_json))
        self.ix_to_word = self.info['ix_to_word']
        self.vocab_size = len(self.ix_to_word)

        self.input_fc_dir = self.opt.input_fc_dir
        self.input_att_dir = self.opt.input_att_dir

        self.seq_length = self.opt.seq_length
        print('max sequence length in data is', self.seq_length)

        imgids = []
        filepath = self.opt.test_image_feat
        pathDir =  os.listdir(filepath)
        for each_name in pathDir:
            imgids.append(each_name.split('.')[0])

        self.imgids = imgids




    # It's not coherent to make DataLoader a subclass of Dataset, but essentially, we only need to implement the following to functions,
    # so that the torch.utils.data.DataLoader can load the data according the index.
    # However, it's minimum change to switch to pytorch data loading.
    def __getitem__(self, index):
        # batch_size = batch_size or self.batch_size
        """This function returns a tuple that is further passed to collate_fn
        """
        ix = index #self.split_ix[index]
        fc_feats, att_feats, num_bbox, ix = get_npy_data(self.opt, ix, \
                    os.path.join(self.input_fc_dir, split_path['val'], 'res152/pool5', str(self.imgids[ix]) + '.jpg.npy'),
                    os.path.join(self.input_att_dir, split_path['val'], 'faster_rcnn_pool5_features', str(self.imgids[ix]) + '.jpg.npz'),
                    )
        fc_feats = torch.from_numpy(fc_feats).float()
        att_feats = torch.from_numpy(att_feats).float()
        num_bbox = torch.from_numpy(np.array([num_bbox])).float()
        # indx = torch.from_numpy(np.array([index])).long()
        # fc_feats = np.stack(tmp_fc_feats)
        # att_feats = np.stack(tmp_att_feats)
        # num_bbox = np.stack(np.array([tmp_num_bbox]))
        # print(np.array([num_bbox]))
        return fc_feats, att_feats, num_bbox

    def __len__(self):
        return len(self.imgids)
