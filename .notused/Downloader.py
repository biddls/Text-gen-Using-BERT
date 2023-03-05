import concurrent.futures
import os
from typing import Pattern
import whisper
from whisper import Whisper
from tqdm import tqdm
import glob
import yt_dlp


def downloadVideoAudio(
        _fileName: str,
        _channel: str):
    ydl_opts = {
        'paths': {'home': f'{_channel}/audio'},
        'format': 'm4a/bestaudio/best',
        'postprocessors': [{  # Extract audio using ffmpeg
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'm4a',
        }]
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([_fileName])
    return


def runModel(
        _model: Whisper,
        _file: str) -> str:
    _file = _file.replace('\\', "/")
    result = _model.transcribe(_file, language="en")["text"]

    # saves output of the model to a text file
    with open(_file.replace("/audio/", "/text/").replace(".m4a", ".txt"), 'w') as f:
        print(result)
        f.write(result)

    # adds tag to audio file that it's been processed
    os.rename(_file, _file.replace("/audio/", "/processed/"))
    return result


def transcribeDirAudio(_model: str, _dir: str) -> [str]:
    model = whisper.load_model(_model)
    results = list()

    files = list(glob.iglob(f'{_dir}/audio/*.m4a'))[::-1]
    for _file in tqdm(files, desc=f"Speach to text inf using {_model}"):
        results.append(runModel(model, _file, ))

    return results


def downloadChannelAudio(
        _channel: str,
        _test: Pattern,
        _count: int,
        _threads: int,
        _testing: bool = False):
    if not os.path.exists(_channel):
        prepDirs(_channel)

    # downloaded = list(glob.iglob(f'{_channel}/**/*.m4a', recursive=True))
    # reads in a text file called data.txt that contains the urls of the videos
    with open(f'{_channel}/urls.txt', 'r') as f:
        urls = f.read().splitlines()

    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, _threads)) as executor:
        for video in urls:
            executor.submit(downloadVideoAudio, video, _channel)


def prepDirs(_channelName: str):
    dirs = ['', 'audio', 'processed', 'text']
    for _dir in dirs:
        _dir = f'{_channelName}/{_dir}'
        if not os.path.exists(_dir):
            os.mkdir(_dir)
