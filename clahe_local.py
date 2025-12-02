#!/usr/bin/env python3
"""
对输入图像只在 mask 区域做 CLAHE（羽化融合）
用法示例:
 python clahe_local.py -i input.png -m mask.png -o out.png --clip 1.3 --tile 8 --feather 21
"""
import cv2, numpy as np, argparse
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("-i","--input", required=True)
p.add_argument("-m","--mask", required=True)
p.add_argument("-o","--out", required=True)
p.add_argument("--clip", type=float, default=1.3)
p.add_argument("--tile", type=int, default=8)
p.add_argument("--feather", type=int, default=21)
args = p.parse_args()

img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
if img is None or mask is None:
    raise SystemExit("无法读取 input 或 mask")

# CLAHE on whole crop, then blend using feathered mask
clahe = cv2.createCLAHE(clipLimit=args.clip, tileGridSize=(args.tile,args.tile))
clahe_img = clahe.apply(img)

# feather mask
m = (mask.astype(np.float32)/255.0)
k = args.feather
if k%2==0: k += 1
m_blur = cv2.GaussianBlur(m, (k,k), sigmaX=max(1.0,k/3.0))
m_blur = np.clip(m_blur, 0.0, 1.0)[...,None]

out = (m_blur * clahe_img[...,None] + (1.0-m_blur) * img[...,None]).astype(np.uint8)[...,0]
cv2.imwrite(args.out, out)
print("写出:", args.out)