#!/bin/bash
#SBATCH --job-name=DTDNML_Train
#SBATCH --output=logs/dtdnml_%j.log
#SBATCH --error=logs/dtdnml_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00

# 激活您的虚拟环境
source /home/dengxiaogui/NTSR/venv/bin/activate

# 创建必要的文件夹
mkdir -p logs
mkdir -p checkpoints

echo "=========================================================="
echo "Starting DTDNML Training Job (No Visdom)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================================="

# ================= 实验控制区 =================
# 注意：加入了 --display_id 0 来彻底关闭画图逻辑



# 【1】PU 数据集 (缩放4倍)
python main.py --name pu_scale_4 --data_path /home/dengxiaogui/Data/PU.mat --r_path /home/dengxiaogui/Data/R.mat --mat_key img --r_key R --scale_factor 4 --num_theta 30 --batchsize 1 --display_id 0 --isCalSP Yes --lambda_A 0.1 --lambda_B 0.001 --lambda_C 0.01 --lr 5e-3 --niter 3000 --niter_decay 7000 --gpu_ids 0
#python main.py --name pu_scale_4 --data_path /home/dengxiaogui/Data/PU.mat --r_path /home/dengxiaogui/Data/R.mat --mat_key img --r_key R --scale_factor 4 --num_theta 30 --batchsize 1 --display_id 0 --isCalSP Yes --lambda_A 0.1 --lambda_B 0.001 --lambda_C 0.01 --lr 5e-3 --niter 3000 --niter_decay 7000 --gpu_ids 0
python metrics.py
# 【2】WDC 数据集 (缩放4倍)
#python main.py --name wadc_scale_4 --data_path /home/dengxiaogui/Data/WDC.mat --r_path /home/dengxiaogui/Data/R.mat --mat_key WDC --r_key R --scale_factor 4 --num_theta 30 --batchsize 1 --display_id 0 --isCalSP Yes --concat Yes --lambda_A 0.1 --lambda_B 0 --lambda_C 0--gpu_ids 0

# 【3】Chikusei 数据集 (缩放8倍)
# python main.py --name chikusei_scale_8 --data_path /home/dengxiaogui/Data/Chikusei.mat --r_path /home/dengxiaogui/Data/R.mat --mat_key chikusei --r_key R --scale_factor 8 --num_theta 30 --batchsize 1 --display_id 0 --isCalSP Yes --concat Yes --lambda_A 0.1 --lambda_B 0 --lambda_C 0--gpu_ids 0

echo "=========================================================="
echo "Job finished."
