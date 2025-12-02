import os
import numpy as np
import soundfile as sf
import librosa
from scipy import signal

def _write_wav(path, y, sr):
    sf.write(path, y, sr)

def spectral_gating_denoise(y, sr, n_fft=2048, hop_length=512, prop_decrease=1.0):
    # 基于简单谱门控：估计噪声谱并衰减低于阈值的谱能量
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = np.abs(S), np.angle(S)
    # 噪声估计：取每频带最小值的中位数作为近似
    noise_spec = np.median(magnitude[:, :max(1, int(0.5*sr/hop_length))], axis=1, keepdims=True)
    # 阈值
    mask_gain = np.maximum(0.0, magnitude - prop_decrease * noise_spec) / (magnitude + 1e-8)
    S_denoised = mask_gain * magnitude * np.exp(1j*phase)
    y_hat = librosa.istft(S_denoised, hop_length=hop_length)
    return y_hat

def spectral_subtract(y, sr, n_fft=2048, hop_length=512):
    # 基于谱减的简单实现，噪声剖面从前0.5秒估计
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = np.abs(S), np.angle(S)
    frames_for_noise = int(min(magnitude.shape[1], (sr//hop_length)//2))
    noise_profile = np.mean(magnitude[:, :frames_for_noise], axis=1, keepdims=True)
    magnitude_sub = magnitude - noise_profile
    magnitude_sub[magnitude_sub < 0] = 0
    S_out = magnitude_sub * np.exp(1j*phase)
    y_hat = librosa.istft(S_out, hop_length=hop_length)
    return y_hat

def process_audio(input_path, out_dir, denoise=True, remove_background=False, progress_callback=None):
    y, sr = librosa.load(input_path, sr=None, mono=True)
    if progress_callback:
        progress_callback("载入音频完成，采样率: %d, 时长: %.2fs" % (sr, len(y)/sr))

    y_proc = y
    if denoise and remove_background:
        if progress_callback: progress_callback("先进行去背景（谱减）...")
        y_proc = spectral_subtract(y_proc, sr)
        if progress_callback: progress_callback("背景去除完成，进行去噪（谱门控）...")
        y_proc = spectral_gating_denoise(y_proc, sr)
    elif denoise:
        if progress_callback: progress_callback("进行去噪（谱门控）...")
        y_proc = spectral_gating_denoise(y_proc, sr)
    elif remove_background:
        if progress_callback: progress_callback("进行去背景（谱减）...")
        y_proc = spectral_subtract(y_proc, sr)
    else:
        if progress_callback: progress_callback("未选择任何处理，复制原始文件。")

    base = os.path.splitext(os.path.basename(input_path))[0]
    out_path = os.path.join(out_dir, base + '_processed.wav')
    _write_wav(out_path, y_proc, sr)
    if progress_callback: progress_callback("写出文件: " + out_path)
    return out_path