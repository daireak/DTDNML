#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Before run this file, please ensure running <python -m visdom.server> in current environment.
Then, please go to http:localhost://#display_port# to see the visulizations.
"""

import torch
import time
import os
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

    # 从 run.sh 获取你传进来的参数
    train_opt = TrainOptions().parse()

    # ================= 移除硬编码 =================
    # 为了让你在 run.sh 里写的参数生效，我把原作者下面强行覆盖配置的代码都注释掉了
    # 如果你想改迭代次数、学习率等，请直接在 run.sh 里加上对应的参数 (例如: --niter 3000 --lr 5e-3)
    #
    # train_opt.niter = 3000
    # train_opt.niter_decay = 7000
    # train_opt.lr = 5e-3
    # train_opt.name = 'sandiego_scale_8'
    # train_opt.data_name = "sandiego"
    # train_opt.scale_factor = 8
    # train_opt.lambda_A = 0.1
    # ==============================================

    # 一些不方便放进 bash 脚本里的固定设置可以留着
    train_opt.print_freq = 100
    train_opt.save_freq = 100
    train_opt.which_epoch = train_opt.niter + train_opt.niter_decay
    train_opt.attention_use = True
    
    train_opt.concat = 'Yes'
    train_opt.useSoftmax = 'No'
    train_opt.lambda_F = 100
    
    # 获取数据集
    train_dataloader = get_dataloader(train_opt, isTrain=True)
    dataset_size = len(train_dataloader)
    train_model = create_model(train_opt, train_dataloader.hsi_channels,
                               train_dataloader.msi_channels,
                               train_dataloader.lrhsi_height,
                               train_dataloader.lrhsi_width,
                               train_dataloader.sp_matrix,
                               train_dataloader.sp_range)

    train_model.setup(train_opt)
    
    # 初始化画图工具 (既然 bash 脚本里用了 display_id 0 关闭 visdom，这里实际上只是个骨架)
    visualizer = Visualizer(train_opt, train_dataloader.sp_matrix)

    total_steps = 0
    
    for epoch in tqdm(range(train_opt.epoch_count, train_opt.niter + train_opt.niter_decay + 1)):
    
        epoch_start_time = time.time()
        iter_data_time = time.time()
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
                
                # 如果开启了 visdom 则画图
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

    # ================= 训练结束，保存结果 =================
    rec_hhsi = train_model.get_current_visuals()[
        train_model.get_visual_corresponding_name()['real_hhsi']].data.cpu().float().numpy()[0]
    
    # 确保保存路径的文件夹存在，防止因为没有创建文件夹而导致闪退
    save_dir = os.path.join("./checkpoints/", train_opt.name, "results")
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, ''.join(data['name']) + '_' + str(epoch) + '.mat')
    sio.savemat(save_path, {'out': rec_hhsi.transpose(1, 2, 0)})

    print('Training Complete. Full time: %d sec' % (time.time() - start_time))
    print(f"Result saved to: {save_path}")