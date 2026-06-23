import os
import numpy as np
import scipy.io as sio
from skimage.metrics import structural_similarity as skimage_ssim

# ==========================================
# 1. 核心评估函数 (完全复用你的 train_pu.py)
# ==========================================
def bandwise_psnr(img_real, img_fake, data_range=1.0):
    mse_bands = np.mean((img_real - img_fake) ** 2, axis=(0, 1))
    psnr_bands = np.zeros_like(mse_bands)
    zero_mask = (mse_bands == 0)
    non_zero_mask = ~zero_mask
    psnr_bands[non_zero_mask] = 10 * np.log10((data_range ** 2) / mse_bands[non_zero_mask])
    psnr_bands[zero_mask] = 100.0
    return np.mean(psnr_bands)

def ergas(img_fake, img_real, scale_factor):
    img_fake, img_real = np.clip(img_fake, 0.0, 1.0), np.clip(img_real, 0.0, 1.0)
    channels = img_real.shape[2]
    inner_sum = sum(((np.sqrt(np.mean((img_real[:, :, i] - img_fake[:, :, i]) ** 2)) / np.mean(img_real[:, :, i])) ** 2) for i in range(channels) if np.mean(img_real[:, :, i]) != 0)
    return 100 / scale_factor * np.sqrt(inner_sum / channels)

def cross_correlation(img_fake, img_real):
    channels = img_real.shape[2]
    cc_val = 0
    for i in range(channels):
        v1, v2 = img_fake[:, :, i].flatten(), img_real[:, :, i].flatten()
        v1, v2 = v1 - np.mean(v1), v2 - np.mean(v2)
        den = np.sqrt(np.sum(v1 ** 2) * np.sum(v2 ** 2))
        if den != 0: cc_val += np.sum(v1 * v2) / den
    return cc_val / channels

def sam(img1, img2):
    img1, img2 = img1.reshape(-1, img1.shape[-1]), img2.reshape(-1, img2.shape[-1])
    cos_theta = np.clip(np.sum(img1 * img2, axis=-1) / (np.linalg.norm(img1, axis=-1) * np.linalg.norm(img2, axis=-1) + 1e-8), -1, 1)
    return np.mean(np.arccos(cos_theta)) * 180 / np.pi

def quality_assessment(S_true, Z_pred, sf):
    Z_pred, S_true = np.clip(Z_pred, 0.0, 1.0), np.clip(S_true, 0.0, 1.0)
    psnr_v = bandwise_psnr(S_true, Z_pred, data_range=1.0)
    ssim_v = skimage_ssim(S_true, Z_pred, channel_axis=-1, data_range=1.0)
    sam_v = sam(S_true, Z_pred)
    ergas_v = ergas(Z_pred, S_true, sf)
    cc_v = cross_correlation(Z_pred, S_true)
    dd_v = np.mean(np.abs(Z_pred - S_true) * 255.0)
    return psnr_v, ssim_v, sam_v, ergas_v, cc_v, dd_v


# ==========================================
# 2. 评测主逻辑
# ==========================================
if __name__ == '__main__':
    # ------------------ 路径设置 ------------------
    # 修改为你真实的 PU 数据集路径
    gt_path = "/home/dengxiaogui/Data/PU.mat" 
    
    # 修改为你刚刚跑出来的 70MB mat 文件路径
    pred_path = "/home/dengxiaogui/DTDNML/checkpoints/pu_scale_4_fixed/results/PU_4000.mat" # 根据你的实际文件名修改
    
    scale_factor = 4
    # ----------------------------------------------

    
    # 1. 加载并对齐 Ground Truth 
    print("Loading Ground Truth...")
    gt_img = sio.loadmat(gt_path)['img']  
    
    # 🌟 修改：与 dataset.py 保持完全一致，只取前 256x256 区域进行评测
    S_true = gt_img[:256, :256, :]
    
    # 归一化
    S_true = (S_true - np.min(S_true)) / (np.max(S_true) - np.min(S_true))
    # 归一化
    S_true = (S_true - np.min(S_true)) / (np.max(S_true) - np.min(S_true))

    # 2. 加载预测结果
    print("Loading Predicted Data...")
    try:
        Z_pred = sio.loadmat(pred_path)['out']
    except Exception as e:
        print(f"读取预测文件失败，请检查路径是否正确: {e}")
        exit()

    print(f"GT Shape: {S_true.shape}, Pred Shape: {Z_pred.shape}")
    
    # 检查维度是否对齐
    if S_true.shape != Z_pred.shape:
        print("警告：GT 和 预测结果的维度不一致！请检查！")
    else:
        # 3. 计算 6 指标
        print("\nCalculating metrics... (It may take a few seconds)")
        psnr_v, ssim_v, sam_v, ergas_v, cc_v, dd_v = quality_assessment(S_true, Z_pred, scale_factor)

        print(f"\n{'='*20} DTDNML 最终评估结果 (Scale={scale_factor}) {'='*20}")
        print(f"{'PSNR':<10} | {psnr_v:<10.4f}")
        print(f"{'SSIM':<10} | {ssim_v:<10.4f}")
        print(f"{'SAM':<10} | {sam_v:<10.4f}")
        print(f"{'ERGAS':<10} | {ergas_v:<10.4f}")
        print(f"{'CC':<10} | {cc_v:<10.4f}")
        print(f"{'DD':<10} | {dd_v:<10.5f}")
        print(f"{'='*54}\n")