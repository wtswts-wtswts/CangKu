#!/usr/bin/env python3
"""
GUI 用的图像处理器（整合版，含向后兼容的 wrapper）

说明：
- 主实现为 process_image_impl(...)，参数为新的细粒度控制项。
- process_image_impl 支持 progress_callback 回调（可选），用于 GUI 更新进度或日志。
- wrapper process_image_file(...) 兼容旧调用签名（denoise, deblur）并过滤/映射多余参数以避免 TypeError。
"""
import os
import sys
from pathlib import Path
import cv2
import numpy as np

# skimage 兼容导入（BM3D 分支可能依赖）
try:
    from skimage.util import img_as_float, img_as_ubyte
    from skimage.restoration import estimate_sigma
except Exception:
    img_as_float = img_as_ubyte = estimate_sigma = None

try:
    from bm3d import bm3d
except Exception:
    bm3d = None

# -------------------- 基础工具 --------------------
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

def feather_mask(mask, ksize=21):
    m = (mask.astype(np.float32)/255.0)
    k = int(ksize)
    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1
    m_blur = cv2.GaussianBlur(m, (k, k), sigmaX=max(1.0, k/3.0))
    return np.clip(m_blur, 0.0, 1.0)

# -------------------- 去噪函数 --------------------
def denoise_crop_nlm(crop, h=6.0):
    return cv2.fastNlMeansDenoising(crop, None, h, 7, 21)

def denoise_crop_bm3d(crop, factor=1.0):
    if bm3d is None:
        raise RuntimeError("BM3D 未安装，请 pip install bm3d 或切换到 NLM")
    if img_as_float is None or estimate_sigma is None:
        raise RuntimeError("需要 scikit-image（estimate_sigma），请 pip install scikit-image")
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
    return img_as_ubyte(np.clip(den, 0.0, 1.0))

# -------------------- 融合与后处理 --------------------
def _ensure_3ch_gray(arr):
    """
    将二维灰度数组 (H,W) 转为 (H,W,1) float32，以便与 (H,W,1) 掩码安全广播。
    如果已是 (H,W,1)，则转换为 float32 并返回。
    """
    a = np.asarray(arr)
    if a.ndim == 2:
        return a.astype(np.float32)[..., np.newaxis]
    elif a.ndim == 3 and a.shape[2] == 1:
        return a.astype(np.float32)
    else:
        # 如果是意外形状，尝试转换到 float32 并加最后一维
        return a.astype(np.float32)[..., np.newaxis]

def blend_crop_back(img, crop_coords, den_crop, crop_mask, feather_k=21):
    """
    把 den_crop (HxW) 融合回 img 的 bbox 部分，使用 crop_mask (HxW) 羽化融合。
    此处确保所有数组在融合前被扩展为 (H,W,1) 以保证广播安全。
    """
    x0,y0,x1,y1 = crop_coords
    # feather 2D -> 2D float, 再扩展
    feather = feather_mask(crop_mask, ksize=feather_k)
    den_f3 = _ensure_3ch_gray(den_crop)
    crop_f3 = _ensure_3ch_gray(img[y0:y1, x0:x1])
    feather3 = feather[..., np.newaxis]  # H,W,1
    # 现在都为 (H,W,1)
    blended = (feather3 * den_f3 + (1.0 - feather3) * crop_f3)[...,0].astype(np.uint8)
    out = img.copy()
    out[y0:y1, x0:x1] = blended
    return out

def unsharp_local(img, mask=None, radius=1.0, amount=1.0, feather_k=21):
    """
    局部锐化；当提供 mask 时，使用羽化掩码把锐化结果和原图融合。
    确保在融合前把所有参与数组扩展为 (H,W,1)。
    """
    out = (img.astype(np.float32) - cv2.GaussianBlur(img.astype(np.float32), (0,0), sigmaX=radius))
    out = (img.astype(np.float32) + amount * out)
    out = np.clip(out, 0, 255).astype(np.uint8)

    if mask is not None:
        f = feather_mask(mask, ksize=feather_k)
        # 做融合时确保维度匹配
        out3 = _ensure_3ch_gray(out)
        img3 = _ensure_3ch_gray(img)
        f3 = f[..., np.newaxis]
        blended = (f3 * out3 + (1.0 - f3) * img3)[...,0].astype(np.uint8)
        return blended
    else:
        return out

def clahe_local(img, mask, clip=1.3, tile=8, feather_k=21):
    """
    在 mask 区域做 CLAHE 并用羽化掩码融合回原图。
    """
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile,tile))
    cimg = clahe.apply(img)
    f = feather_mask(mask, ksize=feather_k)
    # 确保维度一致再融合
    c3 = _ensure_3ch_gray(cimg)
    i3 = _ensure_3ch_gray(img)
    f3 = f[..., np.newaxis]
    blended = (f3 * c3 + (1.0 - f3) * i3)[...,0].astype(np.uint8)
    return blended

# -------------------- 主实现（参数化、内部使用） --------------------
def process_image_impl(input_path, out_dir,
                       adaptive_max=15,
                       inpaint_radius=3,
                       inpaint_method='telea',
                       median_scope='mask',   # 'mask'|'global'|'none'
                       median_k=7,
                       use_bm3d=False,
                       bm3d_factor=1.0,
                       nlm_h=5.0,
                       dilate_mask=2,
                       feather_k=15,
                       tv_weight=0.02,
                       do_clahe=False,
                       clahe_clip=1.2,
                       do_sharpen=True,
                       sharpen_amount=1.15,
                       sharpen_radius=0.8,
                       progress_callback=None):
    """
    新版管线的主体实现。返回最终 output path，并写出中间文件以便调试对比。
    支持 progress_callback(msg: str) 回调（可选）。
    """
    def _progress(msg):
        if callable(progress_callback):
            try:
                progress_callback(msg)
            except Exception:
                pass

    ensure_dir(out_dir)
    base = Path(input_path).stem
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError("无法读取图像: " + input_path)
    cv2.imwrite(os.path.join(out_dir, f"{base}_input.png"), img)
    _progress(f"读取图像，shape={img.shape}")

    # 1) 检测与自适应中值替换（仅替换 mask 区域）
    mask0 = detect_impulse_mask(img)
    cv2.imwrite(os.path.join(out_dir, f"{base}_mask0.png"), mask0)
    _progress(f"检测到极端像素: {int(np.count_nonzero(mask0))} 个")

    img_ad = adaptive_median(img, mask0, max_k=adaptive_max)
    cv2.imwrite(os.path.join(out_dir, f"{base}_adaptive_median.png"), img_ad)
    _progress("自适应中值替换完成")

    # 2) refine mask 并膨胀
    mask2 = detect_impulse_mask(img_ad)
    mask2 = refine_mask(mask2)
    if dilate_mask and dilate_mask>0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_mask*2+1, dilate_mask*2+1))
        mask2 = cv2.dilate(mask2, k)
    cv2.imwrite(os.path.join(out_dir, f"{base}_mask_refined.png"), mask2)
    _progress(f"mask 精修完成: {int(np.count_nonzero(mask2))} 个像素需要处理")

    # 3) inpaint
    img_ip = inpaint_mask(img_ad, mask2, radius=inpaint_radius, method=inpaint_method)
    cv2.imwrite(os.path.join(out_dir, f"{base}_inpaint.png"), img_ip)
    _progress("inpaint 修补完成")

    # 4) 中值作用域
    if median_k and median_k > 1:
        k = median_k if median_k % 2 == 1 else median_k+1
        if median_scope == 'global':
            img_med = cv2.medianBlur(img_ip, k)
        elif median_scope == 'mask':
            med_full = cv2.medianBlur(img_ip, k)
            img_med = img_ip.copy()
            img_med[mask2==255] = med_full[mask2==255]
        else:
            img_med = img_ip
        img_ip = img_med
        cv2.imwrite(os.path.join(out_dir, f"{base}_median_scope_{median_scope}_k{k}.png"), img_ip)
        _progress(f"中值处理完成（scope={median_scope}, k={k}）")

    # 5) 局部去噪：裁切 mask bbox，使用 BM3D 或 NLM
    ys,xs = np.where(mask2>0)
    if len(xs) == 0:
        den = img_ip.copy()
        cv2.imwrite(os.path.join(out_dir, f"{base}_denoised.png"), den)
        _progress("无需局部去噪")
    else:
        y0,y1 = ys.min(), ys.max()
        x0,x1 = xs.min(), xs.max()
        pad = 10
        x0 = max(0, x0-pad); y0 = max(0, y0-pad)
        x1 = min(img.shape[1], x1+pad); y1 = min(img.shape[0], y1+pad)
        crop = img_ip[y0:y1, x0:x1].copy()
        crop_mask = mask2[y0:y1, x0:x1].copy()
        _progress("开始局部去噪（裁切区域并处理）")
        if use_bm3d and bm3d is not None:
            den_crop = denoise_crop_bm3d(crop, factor=bm3d_factor)
        else:
            den_crop = denoise_crop_nlm(crop, h=nlm_h)
        cv2.imwrite(os.path.join(out_dir, f"{base}_crop_denoised.png"), den_crop)
        den = blend_crop_back(img_ip, (x0,y0,x1,y1), den_crop, crop_mask, feather_k=feather_k)
        cv2.imwrite(os.path.join(out_dir, f"{base}_denoised.png"), den)
        _progress("局部去噪并融合完成")

    # 6) 可选局部 CLAHE
    if do_clahe:
        den = clahe_local(den, mask2, clip=clahe_clip, feather_k=feather_k)
        cv2.imwrite(os.path.join(out_dir, f"{base}_tv_clahe.png"), den)
        _progress("局部 CLAHE 完成")

    # 7) 局部锐化
    if do_sharpen:
        den = unsharp_local(den, mask=mask2, radius=sharpen_radius, amount=sharpen_amount, feather_k=feather_k)
        cv2.imwrite(os.path.join(out_dir, f"{base}_final.png"), den)
        _progress("最终锐化完成，写出 final")
    else:
        cv2.imwrite(os.path.join(out_dir, f"{base}_final.png"), den)
        _progress("写出 final")

    return os.path.join(out_dir, f"{base}_final.png")

# -------------------- 向后兼容 wrapper --------------------
def process_image_file(input_path, out_dir, denoise=True, deblur=False, progress_callback=None, **kwargs):
    """
    兼容旧 GUI 的调用签名。
    - denoise (bool): 若为 False，则尽量跳过去噪/中值步骤（把 median_k 设为 1, nlm_h 设为 0）
    - deblur (bool): 旧版本可能期望去模糊，这个 wrapper 不实现 RL deblur；若需要去模糊请在 GUI 中使用新的参数或告诉我以集成 deblur.
    - progress_callback (callable): 若提供，会被传入并由 process_image_impl 用于进度日志.
    - kwargs: 其他新版参数也可以直接传入（会覆盖默认值）；未知参数会被过滤掉以防止 TypeError.
    """
    # 基本默认，新版参数表的默认值
    params = {
        "adaptive_max": 15,
        "inpaint_radius": 3,
        "inpaint_method": "telea",
        "median_scope": "mask",
        "median_k": 7,
        "use_bm3d": False,
        "bm3d_factor": 1.0,
        "nlm_h": 5.0,
        "dilate_mask": 2,
        "feather_k": 15,
        "tv_weight": 0.02,
        "do_clahe": False,
        "clahe_clip": 1.2,
        "do_sharpen": True,
        "sharpen_amount": 1.15,
        "sharpen_radius": 0.8,
        # allow explicit progress callback to be passed through
        "progress_callback": progress_callback
    }

    # 只保留已知参数，过滤未知 kwargs（避免向 process_image_impl 传入不支持的参数）
    allowed = set(params.keys())
    for k, v in kwargs.items():
        if k in allowed:
            params[k] = v
        else:
            # 忽略未知参数但可打印以便调试
            print(f"忽略未知参数: {k}", file=sys.stderr)

    # 处理 denoise 布尔：若 denoise 为 False，则关闭去噪相关步骤
    if denoise is False:
        params["median_k"] = 1          # 不做中值
        params["median_scope"] = "none"
        params["use_bm3d"] = False
        params["nlm_h"] = 0.0
        params["do_sharpen"] = False

    # 处理 deblur 布尔：当前实现未集成 deblur（可扩展）
    if deblur:
        print("警告: GUI 请求 deblur=True，但当前 pipeline 未启用去模糊。若需要此功能请告知以集成去卷积方法.", file=sys.stderr)

    # 调用实现（只传入 process_image_impl 支持的参数）
    impl_kwargs = {k: params[k] for k in params}
    return process_image_impl(input_path, out_dir, **impl_kwargs)