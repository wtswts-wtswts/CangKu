#!/usr/bin/env python3
"""
改进版管线：保留细节的去脉冲 -> 修补 -> 局部中值/去噪 -> 边缘保留增强

主要变化：
- 增加 median-scope（mask/global/none），默认在 inpaint 情况下只对 mask 区域做中值
- 默认更保守的 TV/CLAHE 参数，可关闭 CLAHE
- 优先使用 BM3D（若安装），否则使用 OpenCV NLM（h 默认较小）
- 可选 bilateral 边缘保留平滑
"""
import os
import argparse
from pathlib import Path

import cv2
import numpy as np

# skimage 兼容导入
try:
    from skimage.util import img_as_float, img_as_ubyte
    from skimage.restoration import estimate_sigma, denoise_tv_chambolle
except Exception as e:
    raise RuntimeError("请先安装 scikit-image：python -m pip install scikit-image") from e

# BM3D 可选
try:
    from bm3d import bm3d
except Exception:
    bm3d = None

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def read_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError("无法读取图像: " + path)
    return img

def write_image(path, img):
    cv2.imwrite(path, img)

def detect_impulse_mask(img, low_thresh=5, high_thresh=250):
    mask = np.zeros_like(img, dtype=np.uint8)
    mask[img <= low_thresh] = 255
    mask[img >= high_thresh] = 255
    return mask

def adaptive_median_one_pixel(img, y, x, max_k):
    h, w = img.shape
    orig = int(img[y, x])
    ks = list(range(3, max_k+1, 2))
    for k in ks:
        r = k//2
        y0, y1 = max(0, y-r), min(h, y+r+1)
        x0, x1 = max(0, x-r), min(w, x+r+1)
        patch = img[y0:y1, x0:x1]
        med = int(np.median(patch))
        if med > 5 and med < 250:
            return med
    return orig

def adaptive_median(img, mask, max_k=9):
    out = img.copy()
    ys, xs = np.nonzero(mask)
    for (y, x) in zip(ys, xs):
        out[y, x] = adaptive_median_one_pixel(img, y, x, max_k)
    return out

def refine_mask(mask, open_kernel=3, min_area=2):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    contours, _ = cv2.findContours(m.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned = np.zeros_like(m)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(cleaned, [cnt], -1, 255, -1)
    return cleaned

def inpaint_mask(img, mask, radius=3, method='telea'):
    if np.count_nonzero(mask) == 0:
        return img.copy()
    mode = cv2.INPAINT_TELEA if method == 'telea' else cv2.INPAINT_NS
    return cv2.inpaint(img, mask, radius, mode)

def apply_median_scope(img, mask, k, scope='mask'):
    """
    scope: 'mask' (只对 mask 区域应用中值), 'global' (整张中值), 'none'
    """
    if scope == 'none' or k <= 1:
        return img.copy()
    k = int(k) if int(k)%2==1 else int(k)+1
    if scope == 'global':
        return cv2.medianBlur(img, k)
    # mask scope: 仅替换 mask 区域
    med_full = cv2.medianBlur(img, k)
    out = img.copy()
    out[mask==255] = med_full[mask==255]
    return out

def bilateral_filter(img, d=9, sigmaColor=75, sigmaSpace=75):
    return cv2.bilateralFilter(img, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)

def bm3d_or_nlm(img, use_bm3d=False, bm3d_factor=1.0, nlm_h=7):
    if use_bm3d and bm3d is not None:
        img_f = img_as_float(img)
        try:
            sigma = estimate_sigma(img_f, channel_axis=None)
        except TypeError:
            sigma = estimate_sigma(img_f, multichannel=False)
        sigma_val = float(np.mean(sigma))
        sigma_psd = sigma_val * bm3d_factor
        try:
            den = bm3d(img_f, sigma_psd=sigma_psd)
        except Exception:
            den = bm3d(img_f, sigma_psd)
        den_u = img_as_ubyte(np.clip(den, 0.0, 1.0))
        return den_u
    else:
        # OpenCV NLM（灰度），h 设小以保留细节
        den = cv2.fastNlMeansDenoising(img, None, nlm_h, 7, 21)
        return den

def tv_and_optional_clahe(img, tv_weight=0.02, do_clahe=False, clahe_clip=1.5):
    # 更温和的 TV (默认 0.02)
    img_f = img_as_float(img)
    try:
        res = denoise_tv_chambolle(img_f, weight=tv_weight)
    except Exception:
        res = denoise_tv_chambolle(img_f, weight=tv_weight)
    res_u = img_as_ubyte(np.clip(res, 0.0, 1.0))
    if do_clahe:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8,8))
        res_c = clahe.apply(res_u)
        return res_c
    else:
        return res_u

def unsharp(img, radius=1.0, amount=0.8):
    img_f = img.astype(np.float32)
    blurred = cv2.GaussianBlur(img_f, (0,0), sigmaX=radius)
    mask = img_f - blurred
    out = img_f + amount * mask
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def process(input_path, out_dir, args):
    ensure_dir(out_dir)
    base = Path(input_path).stem

    img = read_gray(input_path)
    write_image(os.path.join(out_dir, f"{base}_input.png"), img)

    # 初始检测 mask
    mask0 = detect_impulse_mask(img, low_thresh=args.sp_low, high_thresh=args.sp_high)
    write_image(os.path.join(out_dir, f"{base}_mask0.png"), mask0)

    # 自适应中值（只替换 mask 区域）
    if args.adaptive_max and args.adaptive_max >= 3:
        img_ad = adaptive_median(img, mask0, max_k=args.adaptive_max)
        write_image(os.path.join(out_dir, f"{base}_adaptive_median.png"), img_ad)
    else:
        img_ad = img.copy()

    # refine mask (基于替换后的图再检测)
    mask2 = detect_impulse_mask(img_ad, low_thresh=args.sp_low, high_thresh=args.sp_high)
    mask2 = refine_mask(mask2, open_kernel=args.open_kernel, min_area=args.min_area)
    write_image(os.path.join(out_dir, f"{base}_mask_refined.png"), mask2)

    # inpaint
    if args.inpaint:
        img_ip = inpaint_mask(img_ad, mask2, radius=args.inpaint_radius, method=args.inpaint_method)
        write_image(os.path.join(out_dir, f"{base}_inpaint.png"), img_ip)
    else:
        img_ip = img_ad

    # 可选 bilateral（在 inpaint 后作为边缘保留平滑）
    if args.bilateral:
        img_ip = bilateral_filter(img_ip, d=args.bilateral_d, sigmaColor=args.bilateral_sc, sigmaSpace=args.bilateral_ss)
        write_image(os.path.join(out_dir, f"{base}_bilateral.png"), img_ip)

    # 中值：默认对 mask 区域（更温和），可用 --median-scope global 来全图处理
    if args.median and args.median > 1:
        img_med = apply_median_scope(img_ip, mask2, args.median, scope=args.median_scope)
        write_image(os.path.join(out_dir, f"{base}_median_scope_{args.median_scope}_k{args.median}.png"), img_med)
        img_ip = img_med

    # BM3D 或 NLM 回退
    den = bm3d_or_nlm(img_ip, use_bm3d=args.bm3d, bm3d_factor=args.bm3d_factor, nlm_h=args.nlm_h)
    write_image(os.path.join(out_dir, f"{base}_denoised.png"), den)

    # TV + 可选 CLAHE（默认不启用 CLAHE）
    tvc = tv_and_optional_clahe(den, tv_weight=args.tv_weight, do_clahe=args.clahe, clahe_clip=args.clahe_clip)
    write_image(os.path.join(out_dir, f"{base}_tv_clahe.png"), tvc)

    # 锐化（可选）
    if args.sharpen:
        out = unsharp(tvc, radius=args.sharpen_radius, amount=args.sharpen_amount)
    else:
        out = tvc
    write_image(os.path.join(out_dir, f"{base}_final.png"), out)

    print("处理完成，输出保存在：", out_dir)
    return os.path.join(out_dir, f"{base}_final.png")

def parse_args():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("-i", "--input", required=True, dest="input")
    p.add_argument("-o", "--out", required=True, dest="out")
    p.add_argument("--sp-low", type=int, default=5)
    p.add_argument("--sp-high", type=int, default=250)
    p.add_argument("--adaptive-max", type=int, default=11)
    p.add_argument("--open-kernel", type=int, default=3)
    p.add_argument("--min-area", type=int, default=2)
    p.add_argument("--inpaint", action="store_true", help="检测并 inpaint 修补检测到的极端像素")
    p.add_argument("--inpaint-radius", type=int, default=3)
    p.add_argument("--inpaint-method", choices=['telea','ns'], default='telea')
    p.add_argument("--median", type=int, default=7, help="中值核（只在 median-scope 指定区域）")
    p.add_argument("--median-scope", choices=['mask','global','none'], default='mask', help="mask: 仅替换 mask 区域；global: 全图中值；none: 不做中值")
    p.add_argument("--bilateral", action="store_true", help="在 inpaint 后使用双边滤波作边缘保留平滑")
    p.add_argument("--bilateral-d", type=int, default=9)
    p.add_argument("--bilateral-sc", type=float, default=75.0)
    p.add_argument("--bilateral-ss", type=float, default=75.0)
    p.add_argument("--bm3d", action="store_true", help="优先使用 BM3D（需安装 bm3d）")
    p.add_argument("--bm3d-factor", type=float, default=1.0)
    p.add_argument("--nlm-h", type=float, default=7.0, help="OpenCV NLM h 参数（灰度）")
    p.add_argument("--tv-weight", type=float, default=0.02)
    p.add_argument("--clahe", action="store_true", help="在 TV 后启用 CLAHE（默认关闭）")
    p.add_argument("--clahe-clip", type=float, default=1.5)
    p.add_argument("--sharpen", action="store_true")
    p.add_argument("--sharpen-radius", type=float, default=1.0)
    p.add_argument("--sharpen-amount", type=float, default=0.8)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process(args.input, args.out, args)