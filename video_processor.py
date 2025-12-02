import os
import tempfile
import numpy as np
import audio_processor, image_processor

def _import_moviepy():
    """
    尝试以多种方式导入 VideoFileClip / AudioFileClip，以兼容不同版本的 moviepy。
    返回 (VideoFileClip, AudioFileClip) 或抛出 RuntimeError。
    """
    errors = []
    # 方案 1: 经典方式
    try:
        from moviepy.editor import VideoFileClip, AudioFileClip
        return VideoFileClip, AudioFileClip
    except Exception as e:
        errors.append(("moviepy.editor", e))
    # 方案 2: 顶层直接导出（有些版本会把类放在顶层）
    try:
        from moviepy import VideoFileClip, AudioFileClip
        return VideoFileClip, AudioFileClip
    except Exception as e:
        errors.append(("moviepy (top-level)", e))
    # 方案 3: 直接从具体子模块导入
    try:
        from moviepy.video.io.VideoFileClip import VideoFileClip
        from moviepy.audio.io.AudioFileClip import AudioFileClip
        return VideoFileClip, AudioFileClip
    except Exception as e:
        errors.append(("moviepy.video.io / moviepy.audio.io", e))

    # 全部失败，抛出友好错误并附上尝试信息
    msg_lines = [
        "无法导入 moviepy 的 VideoFileClip/AudioFileClip（尝试了多种导入方式均失败）。",
        "请在当前 Python 虚拟环境中安装兼容版本的 moviepy 与 imageio-ffmpeg，或将 moviepy 降级到 1.0.3。",
        "安装命令示例：",
        "  python -m pip install --upgrade --force-reinstall moviepy imageio-ffmpeg",
        "或降级（更兼容老代码）：",
        "  python -m pip install --upgrade --force-reinstall moviepy==1.0.3 imageio-ffmpeg",
        "",
        "内部导入尝试的错误信息："
    ]
    for name, err in errors:
        msg_lines.append(f"- {name}: {type(err).__name__}: {err}")
    raise RuntimeError("\n".join(msg_lines))


def process_video(input_path, out_dir, audio_opts=None, image_opts=None, progress_callback=None):
    audio_opts = audio_opts or {}
    image_opts = image_opts or {}

    if progress_callback:
        progress_callback("准备处理视频...")

    # 延迟导入 moviepy（并做多重回退）
    try:
        VideoFileClip, AudioFileClip = _import_moviepy()
    except Exception as e:
        # 如果没有 moviepy，则给出友好提示（由调用方捕获）
        raise RuntimeError(str(e)) from e

    if progress_callback:
        progress_callback("载入视频文件...")
    clip = VideoFileClip(input_path)

    # 处理音频（若选中）
    processed_audio_path = None
    if audio_opts.get('denoise') or audio_opts.get('remove_background'):
        if progress_callback:
            progress_callback("提取并处理音轨...")
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_audio = tmp.name
        # moviepy 写音频会调用 ffmpeg
        clip.audio.write_audiofile(tmp_audio, logger=None)
        processed_audio_path = audio_processor.process_audio(
            tmp_audio, out_dir,
            denoise=audio_opts.get('denoise', False),
            remove_background=audio_opts.get('remove_background', False),
            progress_callback=(lambda t: progress_callback("Audio: " + t) if progress_callback else None)
        )
    else:
        if progress_callback:
            progress_callback("跳过音频处理（未选中）")

    # 处理视频帧（若选中）
    if image_opts.get('denoise') or image_opts.get('deblur'):
        if progress_callback:
            progress_callback("开始逐帧处理视频（可能较慢）...")
        def _proc_frame(frame):
            import cv2
            # moviepy 提供的 frame 是 RGB ndarray，OpenCV 使用 BGR
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out = bgr
            if image_opts.get('denoise') and image_opts.get('deblur'):
                out = image_processor.denoise_image_cv2(out)
                out = image_processor.deblur_richardson_lucy(out)
            elif image_opts.get('denoise'):
                out = image_processor.denoise_image_cv2(out)
            elif image_opts.get('deblur'):
                out = image_processor.deblur_richardson_lucy(out)
            rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            return rgb

        new_clip = clip.fl_image(_proc_frame)
    else:
        if progress_callback:
            progress_callback("跳过帧图像处理（未选中）")
        new_clip = clip

    # 合成处理后的音频（如果有）
    if processed_audio_path:
        if progress_callback:
            progress_callback("将处理后的音频合成回视频...")
        new_audio = AudioFileClip(processed_audio_path)
        final = new_clip.set_audio(new_audio)
    else:
        final = new_clip

    base = os.path.splitext(os.path.basename(input_path))[0]
    out_path = os.path.join(out_dir, base + '_processed.mp4')
    if progress_callback:
        progress_callback("开始写出最终视频（耗时视分辨率和长度而定）...")
    final.write_videofile(out_path, codec='libx264', audio_codec='aac', verbose=False, logger=None)
    if progress_callback:
        progress_callback("写出完成: " + out_path)
    return out_path