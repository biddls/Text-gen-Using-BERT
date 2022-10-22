import concurrent.futures
import os
from pytube import YouTube
from pytube import Channel
from typing import Union, Callable, Pattern
import whisper
from whisper import Whisper
from tqdm import tqdm


def downloadVideoAudio(
        _fileName: str,
        _regex: Union[Pattern, Callable[[str], str]],
        _channelName: str,
        _count=0):

    yt = YouTube(_fileName)
    audio_file = yt.streams.filter(only_audio=True, file_extension='mp4').order_by('abr').asc().first()

    # makes certain checks
    if audio_file.default_filename == 'Video Not Available.mp4':
        if _count > 10:
            print(f'Download failed {_fileName=}')
            return
        downloadVideoAudio(_fileName, _regex, _channelName, _count + 1)
        return
    if isinstance(_regex, Pattern):
        if (a := _regex.search(audio_file.default_filename)) is None:
            print(f'regex output: {a}\n {audio_file.default_filename=} {_fileName=} {_count=}')
            return
        a = a.string[a.regs[1][0]:a.regs[1][1]]
    else:
        if (a := _regex(audio_file.default_filename)) is False:
            print(f'function output: {a}\n {audio_file.default_filename=}')
            return

    _fileName = f"{_fileName.split('=')[-1]}_{a.replace(' ', '_')}.mp4"

    # if the video has already been downloaded skip it
    if not os.path.exists(f'{_channelName}/{_fileName}'):
        print(f'Downloading: {_fileName=}')
        audio_file.download(output_path=_channelName, filename=_fileName)


def runModel(
        _model: Whisper,
        _file: str) -> str:
    result = _model.transcribe(_file, language="en")["text"]

    # saves output of the model to a text file
    with open(f'{_file.replace(".mp4", ".txt")}', 'w') as f:
        f.write(result)

    # adds tag to audio file that it's been processed
    os.rename(_file, _file.replace("/", "/_"))
    return result


def transcribeDirAudio(_model: str, _dir: str) -> [str]:
    model = whisper.load_model(_model)
    results = list()

    files = [file for file in os.listdir(_dir) if (not file.startswith("_") and file.endswith(".mp4"))]
    for _file in tqdm(files, desc=f"Speach to text inf using {_model}"):
        results.append(runModel(model, f"{_dir}/{_file}",))

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
