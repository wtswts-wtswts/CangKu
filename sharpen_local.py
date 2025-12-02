#!/usr/bin/env python3
"""
对整张图或指定区域做 Unsharp 锐化（可选使用 mask 只增强脸部区域）
用法:
  python sharpen_local.py -i input.png -o out.png --amount 1.2 --radius 1.0
  或只对 mask 区域增强:
  python sharpen_local.py -i input.png -m mask.png -o out.png --amount 1.2 --radius 1.0 --feather 21
"""
import cv2, numpy as np, argparse
from pathlib import Path

def unsharp(img, radius=1.0, amount=1.0):
    img_f = img.astype('float32')
    blurred = cv2.GaussianBlur(img_f, (0,0), sigmaX=radius)
    mask = img_f - blurred
    out = img_f + amount * mask
    out = np.clip(out, 0, 255).astype('uint8')
    return out

def feather_mask(mask, ksize=21):
    k = int(ksize)
    if k%2==0: k+=1
    m = (mask.astype(np.float32)/255.0)
    mb = cv2.GaussianBlur(m, (k,k), sigmaX=max(1.0,k/3.0))
    return mb

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-i","--input", required=True)
    p.add_argument("-o","--out", required=True)
    p.add_argument("-m","--mask", default=None)
    p.add_argument("--amount", type=float, default=1.0)
    p.add_argument("--radius", type=float, default=1.0)
    p.add_argument("--feather", type=int, default=21)
    args = p.parse_args()

    img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit("无法读取输入")
    if args.mask:
        mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
        m = feather_mask(mask, ksize=args.feather)[...,None]
        sharp = unsharp(img, radius=args.radius, amount=args.amount).astype('float32')
        out = (m * sharp + (1.0-m) * img).astype('uint8')
    else:
        out = unsharp(img, radius=args.radius, amount=args.amount)
    cv2.imwrite(args.out, out)
    print("saved:", args.out)

if __name__ == '__main__':
    main()