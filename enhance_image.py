#!/usr/bin/env python3
"""
改进版图像增强脚本（针对大量椒盐噪声 + 略模糊的老照片）

主要流程（默认）:
  1) 大中值滤波（去除部分离散噪点）
  2) 检测极端值(0/255)，形态学过滤小噪点，inpaint 修补
  3) 快速非局部去噪（OpenCV）或 skimage nlm（可选）
  4) 可选 TV 平滑
  5) 轻微锐化

用法示例：
  python enhance_image_improved.py -i noisy.png -o out_dir --median 7 --inpaint --nlm --sharpen

依赖：
  pip install numpy opencv-python scikit-image
  （BM3D 更强： pip install bm3d  —— 可选）
"""
import os
import argparse
from pathlib import Path

import cv2
import numpy as np
from skimage.util import img_as_float, img_as_ubyte
from skimage.restoration import denoise_nl_means, estimate_sigma, denoise_tv_chambolle

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
    """
    简单检测极端噪点（近 0 或近 255）。
    返回二值 mask (uint8, 0/255) 表示需要修补的像素。
    low_thresh / high_thresh 可调。
    """
    mask = np.zeros_like(img, dtype=np.uint8)
    mask[np.where(img <= low_thresh)] = 255
    mask[np.where(img >= high_thresh)] = 255
    return mask

def refine_mask(mask, open_kernel=3, min_area=5):
    """
    形态学开操作去掉较大连通区域之外的孤立噪点，再移除过小连通组件（防止误删细节）。
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    # 可选去除超小连通域
    # findContours expects binary (0/255)
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

def fast_nl_means_opencv(img, h=10, templateWindowSize=7, searchWindowSize=21):
    # 仅对灰度图有效
    return cv2.fastNlMeansDenoising(img, None, h=h, templateWindowSize=templateWindowSize, searchWindowSize=searchWindowSize)

def nlm_skimage(img, patch_size=7, patch_distance=11, h_factor=0.8, fast_mode=True):
    img_f = img_as_float(img)
    try:
        sigma_est = estimate_sigma(img_f, multichannel=False)
    except TypeError:
        sigma_est = estimate_sigma(img_f, channel_axis=None)
    h = h_factor * float(np.mean(sigma_est)) if np.ndim(sigma_est) > 0 else h_factor * float(sigma_est)
    try:
        res = denoise_nl_means(img_f, h=h, patch_size=patch_size, patch_distance=patch_distance, fast_mode=fast_mode)
    except TypeError:
        res = denoise_nl_means(img_f, h=h, patch_size=patch_size, patch_distance=patch_distance, fast_mode=fast_mode, channel_axis=None)
    return img_as_ubyte(res)

def tv_denoise_gray(img, weight=0.1):
    img_f = img_as_float(img)
    try:
        res = denoise_tv_chambolle(img_f, weight=weight)
    except TypeError:
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

    # 0) 先做一个较温和的大中值，减少孤立点影响（可选）
    if args.median and args.median > 1:
        cur = median_filter(cur, ksize=args.median)
        write_image(os.path.join(out_dir, f"{base}_median_k{args.median}.png"), cur)

    # 1) 检测并修补极端值（0/255）
    if args.inpaint:
        mask = detect_salt_pepper(cur, low_thresh=args.sp_low, high_thresh=args.sp_high)
        mask_refined = refine_mask(mask, open_kernel=args.open_kernel, min_area=args.min_area)
        write_image(os.path.join(out_dir, f"{base}_mask.png"), mask_refined)
        if np.count_nonzero(mask_refined):
            cur = inpaint_mask(cur, mask_refined, inpaint_radius=args.inpaint_radius, method=args.inpaint_method)
            write_image(os.path.join(out_dir, f"{base}_inpaint.png"), cur)

    # 2) 非局部去噪（优先使用 OpenCV 的快速实现）
    if args.nlm_opencv:
        cur = fast_nl_means_opencv(cur, h=args.cv_h, templateWindowSize=args.cv_template, searchWindowSize=args.cv_search)
        write_image(os.path.join(out_dir, f"{base}_nlm_cv_h{args.cv_h}.png"), cur)
    elif args.nlm:
        try:
            cur = nlm_skimage(cur, patch_size=args.nlm_patch, patch_distance=args.nlm_dist, h_factor=args.nlm_h, fast_mode=not args.nlm_slow)
            write_image(os.path.join(out_dir, f"{base}_nlm_sk_ps{args.nlm_patch}_pd{args.nlm_dist}.png"), cur)
        except Exception as e:
            print("skimage nlm 失败，跳过：", e)

    # 3) TV 去噪（可选）
    if args.tv:
        cur = tv_denoise_gray(cur, weight=args.tv_weight)
        write_image(os.path.join(out_dir, f"{base}_tv_w{args.tv_weight}.png"), cur)

    # 4) 最终锐化（可选）
    if args.sharpen:
        cur = unsharp_mask(cur, radius=args.sharpen_radius, amount=args.sharpen_amount)
        write_image(os.path.join(out_dir, f"{base}_sharpen_r{args.sharpen_radius}_a{args.sharpen_amount}.png"), cur)

    final_path = os.path.join(out_dir, f"{base}_enhanced.png")
    write_image(final_path, cur)
    print("处理完成，输出保存在：", final_path)
    return final_path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--median", type=int, default=7, help="中值核大小（奇数）")
    p.add_argument("--inpaint", action="store_true", help="检测 0/255 噪点并 inpaint 修补")
    p.add_argument("--sp-low", type=int, default=5)
    p.add_argument("--sp-high", type=int, default=250)
    p.add_argument("--open-kernel", type=int, default=3)
    p.add_argument("--min-area", type=int, default=2)
    p.add_argument("--inpaint-radius", type=int, default=3)
    p.add_argument("--inpaint-method", choices=['telea','ns'], default='telea')
    p.add_argument("--nlm-opencv", action="store_true", help="使用 OpenCV 快速 NLM（灰度）")
    p.add_argument("--cv-h", type=float, default=10.0)
    p.add_argument("--cv-template", type=int, default=7)
    p.add_argument("--cv-search", type=int, default=21)
    p.add_argument("--nlm", action="store_true", help="使用 skimage 的 nlm（慢但效果好）")
    p.add_argument("--nlm-patch", type=int, default=7)
    p.add_argument("--nlm-dist", type=int, default=11)
    p.add_argument("--nlm-h", type=float, default=0.8)
    p.add_argument("--nlm-slow", action="store_true")
    p.add_argument("--tv", action="store_true")
    p.add_argument("--tv-weight", type=float, default=0.08)
    p.add_argument("--sharpen", action="store_true")
    p.add_argument("--sharpen-radius", type=float, default=1.0)
    p.add_argument("--sharpen-amount", type=float, default=0.8)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process_pipeline(args.input, args.out, args)