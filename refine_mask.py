#!/usr/bin/env python3
# 用法:
# python refine_mask.py in_mask.png out_mask_closed.png --kernel 3
import cv2, sys
import argparse
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("in_mask")
p.add_argument("out_mask")
p.add_argument("--kernel", type=int, default=3)
p.add_argument("--iterations", type=int, default=1)
args = p.parse_args()

m = cv2.imread(args.in_mask, cv2.IMREAD_GRAYSCALE)
if m is None:
    raise SystemExit("无法读取: " + args.in_mask)
k = args.kernel if args.kernel%2==1 else args.kernel
se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
m_closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE, se, iterations=args.iterations)
cv2.imwrite(args.out_mask, m_closed)
print("已写出:", args.out_mask)