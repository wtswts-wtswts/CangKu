#!/usr/bin/env python3
"""
只对 mask 指定区域做去噪并平滑融合回原图（修正版，修复广播错误）

用法示例:
  python denoise_masked.py --input inpaint.png --mask mask.png -o out_dir --method nlm --nlm-h 6 --dilate 3
"""
import os
import argparse
from pathlib import Path
import cv2
import numpy as np

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

try:
    from skimage.util import img_as_float, img_as_ubyte
    from skimage.restoration import estimate_sigma
except Exception:
    img_as_float = img_as_ubyte = estimate_sigma = None

try:
    from bm3d import bm3d
except Exception:
    bm3d = None

def read_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError("无法读取: " + path)
    return img

def denoise_crop_nlm(crop, h=6):
    den = cv2.fastNlMeansDenoising(crop, None, h, 7, 21)
    return den

def denoise_crop_bm3d(crop, factor=1.0):
    if bm3d is None:
        raise RuntimeError("BM3D 未安装，请 pip install bm3d 或使用 --method nlm")
    if img_as_float is None or estimate_sigma is None:
        raise RuntimeError("需要 scikit-image，请 pip install scikit-image")
    f = img_as_float(crop)
    try:
        sigma = estimate_sigma(f, channel_axis=None)
    except TypeError:
        sigma = estimate_sigma(f, multichannel=False)
    sigma_val = float(np.mean(sigma))
    sigma_psd = sigma_val * factor
    try:
        den = bm3d(f, sigma_psd=sigma_psd)
    except Exception:
        den = bm3d(f, sigma_psd)
    den_u = img_as_ubyte(np.clip(den, 0.0, 1.0))
    return den_u

def feather_mask(mask, ksize=21):
    # mask: uint8 0/255 -> produce float mask in [0,1] with smooth edges by gaussian blur
    m = (mask.astype(np.float32)/255.0)
    # ensure odd kernel size >=1
    k = int(ksize)
    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1
    m_blur = cv2.GaussianBlur(m, (k, k), sigmaX=max(1.0, k/3.0))
    m_blur = np.clip(m_blur, 0.0, 1.0)
    return m_blur

def bbox_of_mask(mask):
    ys, xs = np.where(mask>0)
    if len(xs)==0:
        return None
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    return x0, y0, x1+1, y1+1

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True)
    p.add_argument("--mask", "-m", required=True)
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--method", choices=['nlm','bm3d'], default='nlm')
    p.add_argument("--nlm-h", type=float, default=6.0)
    p.add_argument("--bm3d-factor", type=float, default=1.0)
    p.add_argument("--dilate", type=int, default=3, help="膨胀 mask 的像素半径")
    p.add_argument("--pad", type=int, default=10, help="裁切 bbox 的额外 padding")
    p.add_argument("--feather", type=int, default=21, help="羽化半径（高斯核大小，奇数）")
    args = p.parse_args()

    ensure_dir(args.out)
    img = read_gray(args.input)
    mask = read_gray(args.mask)
    # ensure binary 0/255
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # dilate mask if requested
    if args.dilate and args.dilate>0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.dilate*2+1, args.dilate*2+1))
        mask = cv2.dilate(mask, k)

    bbox = bbox_of_mask(mask)
    if bbox is None:
        print("mask 没有非零区域，直接复制输入到输出")
        out_path = os.path.join(args.out, Path(args.input).stem + "_masked.png")
        cv2.imwrite(out_path, img)
        return

    x0,y0,x1,y1 = bbox
    # add padding
    x0 = max(0, x0 - args.pad); y0 = max(0, y0 - args.pad)
    x1 = min(img.shape[1], x1 + args.pad); y1 = min(img.shape[0], y1 + args.pad)

    crop = img[y0:y1, x0:x1].copy()
    crop_mask = mask[y0:y1, x0:x1].copy()

    # Apply denoising to crop
    if args.method == 'nlm':
        den_crop = denoise_crop_nlm(crop, h=args.nlm_h)
    else:
        den_crop = denoise_crop_bm3d(crop, factor=args.bm3d_factor)

    # Create feathered mask for blending (float HxW in [0,1])
    feather = feather_mask(crop_mask, ksize=args.feather)  # shape H,W, float

    # Ensure shapes align: expand dims for broadcasting (H,W,1)
    feather_3 = feather[..., np.newaxis]  # H,W,1
    den_crop_f = den_crop.astype(np.float32)[..., np.newaxis]  # H,W,1
    crop_f = crop.astype(np.float32)[..., np.newaxis]  # H,W,1

    # Blend: out = feather * den_crop + (1-feather) * crop (per-channel/gray)
    blended = (feather_3 * den_crop_f + (1.0 - feather_3) * crop_f)
    blended_u = np.clip(blended[...,0], 0, 255).astype(np.uint8)  # back to H,W

    # place blended back to image
    out_img = img.copy()
    out_img[y0:y1, x0:x1] = blended_u

    # save outputs
    base = Path(args.input).stem
    cv2.imwrite(os.path.join(args.out, base + "_crop_denoised.png"), den_crop)
    cv2.imwrite(os.path.join(args.out, base + "_blended_crop.png"), blended_u)
    cv2.imwrite(os.path.join(args.out, base + "_final_masked.png"), out_img)
    print("已保存：", args.out)

if __name__ == "__main__":
    main()