#!/usr/bin/env python3
"""
诊断脚本：检测是否为脉冲(0/255)噪声并比对自适应中值/大中值的实际效果。
用法:
  python diagnose_noise.py -i C:/Users/asus/Desktop/l.png -o C:/Users/asus/Desktop/out_dir --adaptive-max 11 --median 11
输出文件到 out_dir:
  - mask.png
  - adaptive_median.png
  - median_k11.png
  - diff_adaptive.png
  - diff_median.png
并在终端打印统计信息（mask 像素数、替换像素数等）
"""
import os
import argparse
from pathlib import Path
import numpy as np
import cv2

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

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

def diff_image(a, b):
    d = cv2.absdiff(a, b)
    # scale for visibility
    return d

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--adaptive-max", type=int, default=11)
    p.add_argument("--median", type=int, default=11)
    p.add_argument("--low-thresh", type=int, default=5)
    p.add_argument("--high-thresh", type=int, default=250)
    args = p.parse_args()

    ensure_dir(args.out)
    img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("无法读取输入图像:", args.input)
        return

    mask = detect_impulse_mask(img, low_thresh=args.low_thresh, high_thresh=args.high_thresh)
    mask_count = int(np.count_nonzero(mask))
    total = img.size
    print(f"输入图像形状: {img.shape}, 总像素: {total}, 检测到的极端像素数(mask): {mask_count}, 占比: {mask_count/total:.2%}")

    # 自适应中值（只替换 mask 区域）
    ad = adaptive_median(img, mask, max_k=args.adaptive_max)
    replaced_ad = np.count_nonzero(img != ad)
    print(f"自适应中值替换后不同像素数: {replaced_ad}, 占比: {replaced_ad/total:.2%}")

    # 直接大中值（全图）
    k = args.median
    if k % 2 == 0:
        k += 1
    med = cv2.medianBlur(img, k)
    replaced_med = np.count_nonzero(img != med)
    print(f"全图中值(k={k})后不同像素数: {replaced_med}, 占比: {replaced_med/total:.2%}")

    # 保存图像：原图、mask、ad, med
    cv2.imwrite(os.path.join(args.out, "orig.png"), img)
    cv2.imwrite(os.path.join(args.out, "mask.png"), mask)
    cv2.imwrite(os.path.join(args.out, f"adaptive_median_kmax{args.adaptive_max}.png"), ad)
    cv2.imwrite(os.path.join(args.out, f"median_k{k}.png"), med)

    # 差异图（可视化）
    dav = diff_image(img, ad)
    dmed = diff_image(img, med)
    cv2.imwrite(os.path.join(args.out, "diff_adaptive.png"), dav)
    cv2.imwrite(os.path.join(args.out, "diff_median.png"), dmed)

    # 打印若干像素值统计（原图、ad、中值）
    def hist_stats(x, name):
        vals, counts = np.unique(x, return_counts=True)
        top = list(zip(vals[-10:], counts[-10:])) if len(vals) > 10 else list(zip(vals, counts))
        print(f"== {name} unique values count: {len(vals)}; sample highest values: {top}")
    hist_stats(img, "orig")
    hist_stats(ad, "adaptive")
    hist_stats(med, "median")

    print("已保存到:", args.out)
    print("请把 mask.png, adaptive_median_*.png, median_*.png, diff_*.png 以及上面的输出贴给我。")

if __name__ == "__main__":
    main()