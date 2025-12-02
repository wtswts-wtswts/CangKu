import os
import tempfile
import threading
import PySimpleGUI as sg
import audio_processor, image_processor, video_processor

# 简单日志函数
def log(window, text):
    window['-LOG-'].print(text)

def run_audio(input_path, out_dir, opts, window):
    try:
        log(window, f"开始音频处理: {input_path}")
        out_path = audio_processor.process_audio(input_path, out_dir,
                                                 denoise=opts.get('denoise', False),
                                                 remove_background=opts.get('remove_background', False),
                                                 progress_callback=lambda t: log(window, t))
        log(window, f"音频处理完成：{out_path}")
    except Exception as e:
        log(window, f"音频处理出错: {e}")

def run_image(input_path, out_dir, opts, window):
    try:
        log(window, f"开始图像处理: {input_path}")
        out_path = image_processor.process_image_file(input_path, out_dir,
                                                      denoise=opts.get('denoise', False),
                                                      deblur=opts.get('deblur', False),
                                                      progress_callback=lambda t: log(window, t))
        log(window, f"图像处理完成：{out_path}")
    except Exception as e:
        log(window, f"图像处理出错: {e}")

def run_video(input_path, out_dir, opts, window):
    try:
        log(window, f"开始视频处理: {input_path}")
        out_path = video_processor.process_video(input_path, out_dir,
                                                 audio_opts=opts.get('audio', {}),
                                                 image_opts=opts.get('image', {}),
                                                 progress_callback=lambda t: log(window, t))
        log(window, f"视频处理完成：{out_path}")
    except Exception as e:
        log(window, f"视频处理出错: {e}")

def main():
    sg.theme('LightBlue2')
    layout = [
        [sg.Text('输入文件'), sg.Input(key='-IN-'), sg.FileBrowse()],
        [sg.Text('输出目录（可选）'), sg.Input(key='-OUT-'), sg.FolderBrowse()],
        [sg.Frame('处理模式', [
            [sg.Radio('Audio', 'MODE', key='-MODE-AUDIO-', default=True),
             sg.Radio('Image', 'MODE', key='-MODE-IMAGE-'),
             sg.Radio('Video', 'MODE', key='-MODE-VIDEO-')]
        ])],
        [sg.Frame('Audio 选项', [
            [sg.Checkbox('去噪（spectral gating）', key='-A-DENOISE-')],
            [sg.Checkbox('去背景音（谱减）', key='-A-REMOVE-BG-')],
        ], key='-FRAME-A-')],
        [sg.Frame('Image 选项', [
            [sg.Checkbox('去噪（NLMeans）', key='-I-DENOISE-')],
            [sg.Checkbox('去模糊（Richardson-Lucy）', key='-I-DEBLUR-')],
        ], key='-FRAME-I-', visible=False)],
        [sg.Frame('Video 选项', [
            [sg.Checkbox('处理音频', key='-V-AUDIO-', default=True)],
            [sg.Checkbox('处理帧图像', key='-V-IMAGE-', default=True)],
            [sg.Text('—— 图像选项 ——')],
            [sg.Checkbox('去噪', key='-V-I-DENOISE-'), sg.Checkbox('去模糊', key='-V-I-DEBLUR-')],
            [sg.Text('—— 音频选项 ——')],
            [sg.Checkbox('去噪', key='-V-A-DENOISE-'), sg.Checkbox('去背景音', key='-V-A-REMOVE-BG-')],
        ], key='-FRAME-V-', visible=False)],
        [sg.Button('Start'), sg.Button('Exit')],
        [sg.Multiline(size=(80, 15), key='-LOG-', autoscroll=True, disabled=False)]
    ]

    window = sg.Window('DSP 作业工具', layout, finalize=True)

    while True:
        event, values = window.read(timeout=100)
        if event in (sg.WIN_CLOSED, 'Exit'):
            break

        # 切换选项卡可见性
        if values['-MODE-AUDIO-']:
            window['-FRAME-A-'].update(visible=True)
            window['-FRAME-I-'].update(visible=False)
            window['-FRAME-V-'].update(visible=False)
        elif values['-MODE-IMAGE-']:
            window['-FRAME-A-'].update(visible=False)
            window['-FRAME-I-'].update(visible=True)
            window['-FRAME-V-'].update(visible=False)
        else:
            window['-FRAME-A-'].update(visible=False)
            window['-FRAME-I-'].update(visible=False)
            window['-FRAME-V-'].update(visible=True)

        if event == 'Start':
            in_file = values['-IN-']
            if not in_file or not os.path.exists(in_file):
                log(window, '请先选择有效的输入文件。')
                continue
            out_dir = values['-OUT-'] if values['-OUT-'] else os.path.dirname(in_file)
            if not out_dir:
                out_dir = '.'
            os.makedirs(out_dir, exist_ok=True)

            # 根据模式调用不同处理线程
            if values['-MODE-AUDIO-']:
                opts = {'denoise': values['-A-DENOISE-'], 'remove_background': values['-A-REMOVE-BG-']}
                threading.Thread(target=run_audio, args=(in_file, out_dir, opts, window), daemon=True).start()
            elif values['-MODE-IMAGE-']:
                opts = {'denoise': values['-I-DENOISE-'], 'deblur': values['-I-DEBLUR-']}
                threading.Thread(target=run_image, args=(in_file, out_dir, opts, window), daemon=True).start()
            else:
                audio_opts = {'denoise': values['-V-A-DENOISE-'], 'remove_background': values['-V-A-REMOVE-BG-']}
                image_opts = {'denoise': values['-V-I-DENOISE-'], 'deblur': values['-V-I-DEBLUR-']}
                opts = {'audio': audio_opts, 'image': image_opts}
                threading.Thread(target=run_video, args=(in_file, out_dir, opts, window), daemon=True).start()

    window.close()

if __name__ == '__main__':
    main()