# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torchvision.utils import save_image
import os
import imageio
import numpy as np
import data
from util.util import mkdir
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from torch.utils.data import DataLoader
from dataset2 import IllustTestDataset
from pathlib import Path

if __name__ == '__main__':
    opt = TestOptions().parse()
    dataset = IllustTestDataset(Path("/home/jason/code/CoCosNet/imgs/colorart/b/"),
                         Path("/home/jason/code/CoCosNet/imgs/colorart/a/"),
                         ["xdog"],
                         "png",
                         256)
    dataloader = DataLoader(dataset,
                        batch_size=opt.batchSize,
                        shuffle=True,
                        drop_last=True)
    #dataloader = data.create_dataloader(opt)
    model = Pix2PixModel(opt)
    if len(opt.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
    else:
        model.to(opt.gpu_ids[0])
    model.eval()
    save_root = os.path.join(opt.checkpoints_dir, opt.name, 'test')
    mkdir(save_root)
    for i, data_i in enumerate(dataloader):
        print('{} / {}'.format(i, len(dataloader)))
        if i * opt.batchSize >= opt.how_many:
            break
        imgs_num = data_i['label'].shape[0]
        out = model(data_i, mode='inference')
        if opt.save_per_img:
            try:
                for it in range(imgs_num):
                    save_name = os.path.join(save_root, '%08d_%04d.png' % (i, it))
                    save_image(out['fake_image'][it:it+1], save_name, padding=0, normalize=True)
            except OSError as err:
                print(err)
        else:
            label = data_i['label'][:,:3,:,:]
            print(out['fake_image'].size())
            imgs = torch.cat((label.cpu(), out['ref_image'].data.cpu(), out['fake_image'].data.cpu(),out['warp_out'][3].data.cpu(),data_i['image'].cpu()), 0)
            try:
                save_name = os.path.join(save_root, '%08d.png' % i)
                save_image(imgs, save_name, nrow=imgs_num, padding=0, normalize=True)
            except OSError as err:
                print(err)
