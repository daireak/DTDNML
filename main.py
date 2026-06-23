#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Before run this file, please ensure running <python -m visdom.server> in current environment.
Then, please go to http:localhost://#display_port# to see the visualizations.
"""

import torch
import time
import os
# 如果没有 hues，可以注释掉下面这行
try:
    import hues
except ImportError:
    pass

from data import get_dataloader
from model import create_model
from options.train_options import TrainOptions
from utils.visualizer import Visualizer
import scipy.io as sio
import numpy as np
import random
from tqdm import tqdm


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
setup_seed(5)

if __name__ == "__main__":

    start_time = time.time()

    # 解析命令行传入的参数（run.sh 里的参数在这里被接收）
    train_opt = TrainOptions().parse()

    # ====== 给予基础参数默认值 (如果命令行没传的话) ======
    if not hasattr(train_opt, 'niter'): train_opt.niter = 3000
    if not hasattr(train_opt, 'niter_decay'): train_opt.niter_decay = 7000
    if not hasattr(train_opt, 'lr'): train_opt.lr = 5e-3
    if not hasattr(train_opt, 'lr_decay_iters'): train_opt.lr_decay_iters = 1000
    if not hasattr(train_opt, 'display_port'): train_opt.display_port = 8097
    if not hasattr(train_opt, 'print_freq'): train_opt.print_freq = 100
    if not hasattr(train_opt, 'save_freq'): train_opt.save_freq = 100
    
    train_opt.which_epoch = train_opt.niter + train_opt.niter_decay
    
    # 确保保存结果的文件夹存在，避免保存时报错
    save_dir = os.path.join("./checkpoints/", train_opt.name, "results")
    os.makedirs(save_dir, exist_ok=True)

    # 获取数据加载器
    train_dataloader = get_dataloader(train_opt, isTrain=True)
    dataset_size = len(train_dataloader)
    
    # 创建模型
    train_model = create_model(train_opt, train_dataloader.hsi_channels,
                               train_dataloader.msi_channels,
                               train_dataloader.lrhsi_height,
                               train_dataloader.lrhsi_width,
                               train_dataloader.sp_matrix,
                               train_dataloader.sp_range)

    train_model.setup(train_opt)
    visualizer = Visualizer(train_opt, train_dataloader.sp_matrix)

    total_steps = 0
    
    for epoch in tqdm(range(train_opt.epoch_count, train_opt.niter + train_opt.niter_decay + 1)):
    
        epoch_start_time = time.time()
        iter_start_time = time.time()
        epoch_iter = 0

        train_psnr_list = []

        for i, data in enumerate(train_dataloader):
            iter_start_time = time.time()
            total_steps += train_opt.batchsize
            epoch_iter += train_opt.batchsize

            visualizer.reset()

            train_model.set_input(data, True)
            train_model.optimize_joint_parameters(epoch)

            train_psnr = train_model.cal_psnr()
            train_psnr_list.append(train_psnr)

            if epoch % train_opt.print_freq == 0:
                losses = train_model.get_current_losses()
                t = (time.time() - iter_start_time) / train_opt.batchsize
                visualizer.print_current_losses(epoch, epoch_iter, losses, t)
                if train_opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, train_opt, losses)
                    visualizer.display_current_results(train_model.get_current_visuals(),
                                                       train_model.get_image_name(), epoch, True,
                                                       win_id=[1])

                    visualizer.plot_spectral_lines(train_model.get_current_visuals(), train_model.get_image_name(),
                                                   visual_corresponding_name=train_model.get_visual_corresponding_name(),
                                                   win_id=[2, 3])
                    visualizer.plot_psnr_sam(train_model.get_current_visuals(), train_model.get_image_name(),
                                             epoch, float(epoch_iter) / dataset_size,
                                             train_model.get_visual_corresponding_name())

                    visualizer.plot_lr(train_model.get_LR(), epoch)

        train_model.update_learning_rate()
        
    # 训练结束保存最后一次的mat结果
    rec_hhsi = train_model.get_current_visuals()[train_model.get_visual_corresponding_name()['real_hhsi']].data.cpu().float().numpy()[0]
    
    # 保存结果到 checkpoints/实验名/results 下
    mat_save_path = os.path.join(save_dir, ''.join(data['name']) + '_' + str(epoch) + '.mat')
    sio.savemat(mat_save_path, {'out': rec_hhsi.transpose(1, 2, 0)})

    print('Training completed. Full time %d sec' % (time.time() - start_time))