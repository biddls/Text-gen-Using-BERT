import concurrent.futures
import os
from pytube import YouTube
from pytube import Channel
from typing import Union, Callable, Pattern
import whisper


# tesing
from time import time


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


def downloadChannelAudio(_channel: str, _regex: Union[Pattern, Callable[[str], str]], _count: int, _threads: int):
    _count = -1 if _count < 1 else _count

    if not os.path.exists(_channel):
        os.makedirs(_channel)
    s = time()
    if _threads > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=_threads) as executor:
            for video in Channel(f"https://www.youtube.com/c/{_channel}/videos")[:_count]:
                # checks if it's been downloaded before
                _break = False
                for vid in os.listdir(_channel):
                    if video.split("=")[-1] in vid:
                        _break = True
                if not _break:
                    executor.submit(downloadVideoAudio, video, _regex, _channel)
    else:
        for video in Channel(f"https://www.youtube.com/c/{_channel}/videos")[:_count]:
            # checks if it's been downloaded before
            _break = False
            for vid in os.listdir(_channel):
                if video.split("=")[-1] in vid:
                    _break = True
            if not _break:
                downloadVideoAudio(video, _regex, _channel)
    print(f"it took {time() - s} seconds to process the downloads")

    model = whisper.load_model("base")
    print("HAI")
    result = model.transcribe("TheDailyGwei\\8wVaBFvUxyg_The_Daily_Gwei_Refuel_471.mp4")
    print(result["text"])
