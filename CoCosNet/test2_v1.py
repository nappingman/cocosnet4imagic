# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from collections import OrderedDict
import torch
import torchvision.utils as vutils
import torch.nn.functional as F
import data
import numpy as np
from util.util import masktorgb
from options.test_options import TestOptions
from models.pix2pix_model_v1 import Pix2PixModel


from torch.utils.data import DataLoader
from dataset2 import IllustTestDataset
from pathlib import Path

opt = TestOptions().parse()
#Path("/archive/zhaowei/colorart/b/"),
#"/archive/zhaowei/evaluation/daojian_b/"
torch.manual_seed(0)
#/archive/zhaowei/evaluation/daojian_a/
#"/archive/zhaowei/colorart/a/"
dataset = IllustTestDataset(Path("/archive/zhaowei/colorart/b/"),
                         Path("/archive/zhaowei/colorart/a/"),
                         ["blend"],
                         "png",
                         256)
dataloader = DataLoader(dataset,
                        batch_size=opt.batchSize,
                        shuffle=True,
                        drop_last=True)
save_dir = './warpb_a/'
if not os.path.exists(save_dir):
  os.mkdir(save_dir)
# dataloader = data.create_dataloader(opt)
# dataloader.dataset[0]

model = Pix2PixModel(opt)
model.eval()

save_root = os.path.join(os.path.dirname(opt.checkpoints_dir), 'output')

# test
for i, data_i in enumerate(dataloader):
    name = data_i['name'][0]
    print('{} / {}'.format(i, len(dataloader)))
  
    if i * opt.batchSize >= opt.how_many:
        break
    imgs_num = data_i['label'].shape[0]
    #data_i['stage1'] = torch.ones_like(data_i['stage1'])
    
    out = model(data_i, mode='inference')
    if opt.save_per_img:
        root = save_root + '/test_per_img/'
        if not os.path.exists(root + opt.name):
            os.makedirs(root + opt.name)
        imgs = out['fake_image'].data.cpu()
        try:
            imgs = (imgs + 1) / 2
            for i in range(imgs.shape[0]):
                if opt.dataset_mode == 'deepfashion':
                    name = data_i['path'][i].split('Dataset/DeepFashion/')[-1].replace('/', '_')
                else:
                    name = os.path.basename(data_i['path'][i])
                vutils.save_image(imgs[i:i+1], root + opt.name + '/' + name,  
                        nrow=1, padding=0, normalize=False)
        except OSError as err:
            print(err)
    else:
        if not os.path.exists(save_root + '/test/' + opt.name):
            os.makedirs(save_root + '/test/' + opt.name)

        if opt.dataset_mode == 'deepfashion':
            label = data_i['label'][:,:3,:,:]
        elif opt.dataset_mode == 'celebahqedge':
            label = data_i['label'].expand(-1, 3, -1, -1).float()
        else:
            label = masktorgb(data_i['label'].cpu().numpy())
            label = torch.from_numpy(label).float() / 128 - 1
        imgs = torch.cat((out['input_semantics'].data.cpu(), out['ref_image'].data.cpu(), out['warp_out'].data.cpu(), out['fake_image'].data.cpu()), 0)
        #imgs = torch.cat((out['input_semantics'].data.cpu(), out['ref_image'].data.cpu(), out['fake_image'].data.cpu()), 0)
        
        
        #imgs = torch.cat((label.cpu(), data_i['ref'].cpu(), out['fake_image'].data.cpu()), 0)
        try:
            imgs = (imgs + 1) / 2
            vutils.save_image(imgs, save_dir+name,  
                    nrow=imgs_num, padding=0, normalize=False)            
            #vutils.save_image(imgs, save_root + '/test/' + opt.name + '/' + str(i) + '.png',  
                    #nrow=imgs_num, padding=0, normalize=False)
        except OSError as err:
            print(err)
