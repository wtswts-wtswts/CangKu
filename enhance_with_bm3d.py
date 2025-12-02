#!/usr/bin/env python3
"""
增强脚本（BM3D 优先），针对大量斑点/椒盐噪声的老照片

流程（推荐顺序）:
  1) 中值滤波（去孤立噪点）
  2) 检测极端像素(0/255)并用形态学精修 mask
  3) inpaint 修补极端像素
  4) BM3D 去噪（效果强，保细节）
  5) 可选 TV 去噪 / 轻微锐化

运行示例:
  python enhance_with_bm3d.py -i C:/Users/asus/Desktop/l.png -o out_dir --median 7 --inpaint --bm3d --sharpen

依赖:
  python -m pip install numpy opencv-python scikit-image bm3d
"""
import os
import argparse
from pathlib import Path

import cv2
import numpy as np

# 从 skimage 取工具
from skimage.util import img_as_float, img_as_ubyte
from skimage.restoration import estimate_sigma, denoise_tv_chambolle

# BM3D
try:
    from bm3d import bm3d
except Exception:
    bm3d = None  # 在运行时检查，缺失则回退到 nlm 或 opencv

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)
    return str(Path(p))

def read_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError("无法读取图像: " + path)
    return img

def write_image(path, img):
    cv2.imwrite(path, img)

def median_filter(img, ksize=3):
    k = max(1, int(ksize))
    if k % 2 == 0:
        k += 1
    return cv2.medianBlur(img, k)

def detect_salt_pepper(img, low_thresh=5, high_thresh=250):
    mask = np.zeros_like(img, dtype=np.uint8)
    mask[np.where(img <= low_thresh)] = 255
    mask[np.where(img >= high_thresh)] = 255
    return mask

def refine_mask(mask, open_kernel=3, min_area=5):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    contours, _ = cv2.findContours(mask_open.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned = np.zeros_like(mask_open)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            cv2.drawContours(cleaned, [cnt], -1, 255, -1)
    return cleaned

def inpaint_mask(img, mask, inpaint_radius=3, method='telea'):
    if np.count_nonzero(mask) == 0:
        return img.copy()
    mode = cv2.INPAINT_TELEA if method == 'telea' else cv2.INPAINT_NS
    return cv2.inpaint(img, mask, inpaint_radius, mode)

def bm3d_denoise_gray(img, sigma_factor=1.0):
    """
    使用 bm3d 去噪。bm3d 包的输入为 float 图像（0..1），sigma_psd 为标准差（0..1）。
    我们用 estimate_sigma 自动估计噪声，再乘以 sigma_factor 用作 bm3d 强度控制。
    """
    if bm3d is None:
        raise RuntimeError("BM3D 库不可用。请先 pip install bm3d")
    img_f = img_as_float(img)  # float 0..1
    try:
        sigma_est = estimate_sigma(img_f, channel_axis=None)
    except TypeError:
        sigma_est = estimate_sigma(img_f, multichannel=False)
    # sigma_est 可能是标量或数组
    sigma = float(np.mean(sigma_est)) * sigma_factor
    # bm3d 返回 float 0..1
    den = bm3d(img_f, sigma_psd=sigma, stage_arg=bm3d.BM3DStages.ALL_STAGES) if hasattr(bm3d, 'BM3DStages') else bm3d(img_f, sigma_psd=sigma)
    den_u = img_as_ubyte(np.clip(den, 0.0, 1.0))
    return den_u

def nlm_fallback(img, patch_size=7, patch_distance=11, h_factor=0.8):
    # 简单回退：使用 skimage nlm（相对慢）
    from skimage.restoration import denoise_nl_means
    img_f = img_as_float(img)
    try:
        sigma_est = estimate_sigma(img_f, channel_axis=None)
    except TypeError:
        sigma_est = estimate_sigma(img_f, multichannel=False)
    h = h_factor * float(np.mean(sigma_est))
    try:
        den = denoise_nl_means(img_f, h=h, patch_size=patch_size, patch_distance=patch_distance, fast_mode=True)
    except TypeError:
        den = denoise_nl_means(img_f, h=h, patch_size=patch_size, patch_distance=patch_distance, fast_mode=True, channel_axis=None)
    return img_as_ubyte(np.clip(den, 0.0, 1.0))

def tv_denoise_gray(img, weight=0.1):
    img_f = img_as_float(img)
    res = denoise_tv_chambolle(img_f, weight=weight)
    return img_as_ubyte(np.clip(res, 0.0, 1.0))

def unsharp_mask(img, radius=1.0, amount=1.0):
    img_f = img.astype(np.float32)
    blurred = cv2.GaussianBlur(img_f, ksize=(0,0), sigmaX=radius, sigmaY=radius)
    mask = img_f - blurred
    out = img_f + amount * mask
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def process_pipeline(input_path, out_dir, args):
    ensure_dir(out_dir)
    base = Path(input_path).stem

    img = read_gray(input_path)
    write_image(os.path.join(out_dir, f"{base}_input.png"), img)
    cur = img.copy()

    # 1) 中值
    if args.median and args.median > 1:
        cur = median_filter(cur, ksize=args.median)
        write_image(os.path.join(out_dir, f"{base}_median_k{args.median}.png"), cur)

    # 2) inpaint 修补 0/255
    if args.inpaint:
        mask = detect_salt_pepper(cur, low_thresh=args.sp_low, high_thresh=args.sp_high)
        mask_refined = refine_mask(mask, open_kernel=args.open_kernel, min_area=args.min_area)
        write_image(os.path.join(out_dir, f"{base}_mask.png"), mask_refined)
        if np.count_nonzero(mask_refined):
            cur = inpaint_mask(cur, mask_refined, inpaint_radius=args.inpaint_radius, method=args.inpaint_method)
            write_image(os.path.join(out_dir, f"{base}_inpaint.png"), cur)

    # 3) BM3D 或回退
    if args.bm3d:
        try:
            den = bm3d_denoise_gray(cur, sigma_factor=args.bm3d_factor)
            write_image(os.path.join(out_dir, f"{base}_bm3d.png"), den)
            cur = den
        except Exception as e:
            print("BM3D 失败，将回退到 nlm: ", e)
            den = nlm_fallback(cur, patch_size=args.nlm_patch, patch_distance=args.nlm_dist, h_factor=args.nlm_h)
            write_image(os.path.join(out_dir, f"{base}_nlm_fallback.png"), den)
            cur = den
    elif args.nlm:
        den = nlm_fallback(cur, patch_size=args.nlm_patch, patch_distance=args.nlm_dist, h_factor=args.nlm_h)
        write_image(os.path.join(out_dir, f"{base}_nlm.png"), den)
        cur = den
    else:
        # 可选：使用 OpenCV 快速 NLM（灰度）
        if args.cv_nlm:
            cur = cv2.fastNlMeansDenoising(cur, None, h=args.cv_h, templateWindowSize=args.cv_template, searchWindowSize=args.cv_search)
            write_image(os.path.join(out_dir, f"{base}_nlm_cv.png"), cur)

    # 4) TV 去噪（可选）
    if args.tv:
        cur = tv_denoise_gray(cur, weight=args.tv_weight)
        write_image(os.path.join(out_dir, f"{base}_tv.png"), cur)

    # 5) 锐化（可选）
    if args.sharpen:
        cur = unsharp_mask(cur, radius=args.sharpen_radius, amount=args.sharpen_amount)
        write_image(os.path.join(out_dir, f"{base}_sharpen.png"), cur)

    final_path = os.path.join(out_dir, f"{base}_enhanced.png")
    write_image(final_path, cur)
    print("处理完成，输出保存在：", final_path)
    return final_path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--median", type=int, default=7)
    p.add_argument("--inpaint", action="store_true")
    p.add_argument("--sp-low", type=int, default=5)
    p.add_argument("--sp-high", type=int, default=250)
    p.add_argument("--open-kernel", type=int, default=3)
    p.add_argument("--min-area", type=int, default=2)
    p.add_argument("--inpaint-radius", type=int, default=3)
    p.add_argument("--inpaint-method", choices=['telea','ns'], default='telea')
    p.add_argument("--bm3d", action="store_true", help="使用 BM3D（优先），需要 pip install bm3d")
    p.add_argument("--bm3d-factor", type=float, default=1.0, help="BM3D 强度系数（乘以估计的 sigma）")
    p.add_argument("--nlm", action="store_true")
    p.add_argument("--nlm-patch", type=int, default=7)
    p.add_argument("--nlm-dist", type=int, default=11)
    p.add_argument("--nlm-h", type=float, default=0.8)
    p.add_argument("--cv-nlm", action="store_true")
    p.add_argument("--cv-h", type=float, default=10.0)
    p.add_argument("--cv-template", type=int, default=7)
    p.add_argument("--cv-search", type=int, default=21)
    p.add_argument("--tv", action="store_true")
    p.add_argument("--tv-weight", type=float, default=0.08)
    p.add_argument("--sharpen", action="store_true")
    p.add_argument("--sharpen-radius", type=float, default=1.0)
    p.add_argument("--sharpen-amount", type=float, default=0.8)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process_pipeline(args.input, args.out, args)