import concurrent.futures
import os
from pytube import YouTube
from pytube import Channel
from typing import Union, Callable, Pattern
import whisper
from whisper import Whisper

def downloadVideoAudio(_videoLink: str, _regex: Union[Pattern, Callable[[str], str]], _channelName: str, _count=0):
    # queues up the file to be downloaded
    yt = YouTube(_videoLink)
    audio_file = yt.streams.filter(only_audio=True, file_extension='mp4').order_by('abr').asc().first()

    # makes certain checks
    if audio_file.default_filename == 'Video Not Available.mp4':
        if _count > 10:
            print(f'Download failed {_videoLink=}')
            return
        downloadVideoAudio(_videoLink, _regex, _channelName, _count + 1)
        return
    if isinstance(_regex, Pattern):
        if (a := _regex.search(audio_file.default_filename)) is None:
            print(f'regex output: {a}\n {audio_file.default_filename=} {_videoLink=} {_count=}')
            return
        a = a.string[a.regs[1][0]:a.regs[1][1]]
    else:
        if (a := _regex(audio_file.default_filename)) is False:
            print(f'function output: {a}\n {audio_file.default_filename=}')
            return
    # downloads the file
    _videoLink = f"{_videoLink.split('=')[-1]}_{a.replace(' ', '_')}"
    if os.path.exists(f'{_channelName}/{_videoLink}.mp4'):
        return
    print(f'Downloading: {_videoLink=}')
    audio_file.download(output_path=_channelName, filename=f'{_videoLink}.mp4')


def runModel(_model: Whisper, _file: str) -> str:
    result = _model.transcribe(_file, language="en")["text"]
    with open(f'{_file.replace(".mp4", ".txt")}', 'w') as f:
        f.write(result)

    # os.rename(_file, _file.split("/")[0]+"/processed_"+_file.split("/")[1])
    return result


def transcribeAudio(_model: str, _dir: str) -> [str]:
    if _model not in whisper.available_models():
        raise ValueError(str(f"{_model} has to be in {whisper.available_models()}"))
    model = whisper.load_model(_model)
    results = list()
    for file in os.listdir(_dir):
        # checks if it's been processes before
        if file.startswith("processed_"):
            continue
        # only process mp4 files
        if file.endswith(".mp4"):
            results.append(runModel(model, f"{_dir}/{file}",))

    return results


def downloadChannelAudio(
        _channel: str,
        _regex: Union[Pattern, Callable[[str], Union[str, bool]]],
        _count: int,
        _threads: int):

    _count = max(1, _count)

    if not os.path.exists(_channel):
        os.makedirs(_channel)

    with concurrent.futures.ProcessPoolExecutor(max_workers=_threads) as executor:
        for video in Channel(f"https://www.youtube.com/c/{_channel}/videos")[:_count]:
            # checks if it's been downloaded before
            _break = False
            for vid in os.listdir(_channel):
                if vid.endswith('.mp4'):
                    if video.split("=")[-1] in vid:
                        _break = True
            if not _break:
                executor.submit(downloadVideoAudio, video, _regex, _channel)