import concurrent.futures
import os
import re

from pytube import YouTube
from pytube import Channel
from typing import Union, Callable, Pattern
import whisper
from whisper import Whisper
from tqdm import tqdm
import glob


def downloadVideoAudio(
        _fileName: str,
        _regex: Union[Pattern, Callable[[str], str]],
        _channelName: str,
        _count=0,
        _testing: bool = False):
    yt = YouTube(_fileName)
    audio_file = yt.streams.filter(only_audio=True, file_extension='mp4').order_by('abr').asc().first()

    # makes certain checks
    if audio_file.default_filename == 'Video Not Available.mp4':
        if _count > 10:
            if _testing:
                print(f'Download failed {_fileName=}')
            return
        downloadVideoAudio(_fileName, _regex, _channelName, _count + 1)
        return
    if isinstance(_regex, Pattern):
        if (a := _regex.search(audio_file.default_filename)) is None:
            if _testing:
                print(f'regex output: {a}\n {audio_file.default_filename=} {_fileName=} {_count=}')
            return
        a = a.string[a.regs[1][0]:a.regs[1][1]]
    else:
        if (a := _regex(audio_file.default_filename)) is False:
            if _testing:
                print(f'function output: {a}\n {audio_file.default_filename=}')
            return

    _fileName = f"{_fileName.split('=')[-1]}_{a.replace(' ', '_')}.mp4"
    _channelName = f'{_channelName}/audio'

    # if the video has already been downloaded skip it
    if not os.path.exists(f'{_channelName}/{_fileName}'):
        print(f'Downloading: {_fileName=}')
        audio_file.download(output_path=_channelName, filename=_fileName)


def runModel(
        _model: Whisper,
        _file: str) -> str:
    _file = _file.replace('\\', "/")
    result = _model.transcribe(_file, language="en")["text"]

    # saves output of the model to a text file
    with open(_file.replace("/audio/", "/text/").replace(".mp4", ".txt"), 'w') as f:
        f.write(result)

    # adds tag to audio file that it's been processed
    os.rename(_file, _file.replace("/audio/", "/processed/"))
    return result


def transcribeDirAudio(_model: str, _dir: str) -> [str]:
    model = whisper.load_model(_model)
    results = list()

    files = list(glob.iglob(f'{_dir}/audio/*.mp4'))
    for _file in tqdm(files, desc=f"Speach to text inf using {_model}"):
        results.append(runModel(model, _file, ))

    return results


def downloadChannelAudio(
        _channel: str,
        _regex: Union[Pattern, Callable[[str], Union[str, bool]]],
        _count: int,
        _threads: int,
        _testing: bool = False):
    if not os.path.exists(_channel):
        prepDirs(_channel)

    downloaded = list(glob.iglob(f'{_channel}/**/*.mp4', recursive=True))
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, _threads)) as executor:
        for video in Channel(f'https://www.youtube.com/c/{_channel}/videos')[:max(1, _count)]:
            # checks if it's been downloaded before
            _break = False
            for vid in downloaded:
                if video.split("=")[-1] in vid:
                    _break = True
            if not _break:
                executor.submit(downloadVideoAudio, video, _regex, _channel, _testing=_testing)


def prepDirs(_channelName: str):
    dirs = ['', 'audio', 'processed', 'text']
    for _dir in dirs:
        _dir = f'{_channelName}/{_dir}'
        if not os.path.exists(_dir):
            os.mkdir(_dir)


if __name__ == "__main__":
    channel = 'TheDailyGwei'
    test = re.compile('(The Daily Gwei Refuel \d+)?=* - Ethereum Updates.mp4')
    downloadChannelAudio(channel, test, 5, 2)
    results = transcribeDirAudio('medium.en', channel)
