from pytube import YouTube
from pytube import Channel
from tqdm import tqdm
import os
from time import sleep


if __name__ == "__main__":
    if not os.path.exists('Daily_Gwei'):
        os.makedirs('Daily_Gwei')
    os.chdir("Daily_Gwei")
    if not os.getcwd().endswith("Daily_Gwei"):
        raise FileNotFoundError("Not in corrent dir")

    for video in Channel("https://www.youtube.com/c/TheDailyGwei/videos"):
        yt = YouTube(video)
        audio_file = yt.streams.filter(only_audio=True).order_by('abr').desc().first()
        print(audio_file.default_filename, video)
    exit(0)

    channel = tqdm(Channel("https://www.youtube.com/c/TheDailyGwei/videos"))
    for video in channel:
        # print(video)
        yt = YouTube(video)
        audio_file = yt.streams.filter(only_audio=True).order_by('abr').desc().first()
        # print(audio_file)
        channel.set_description(audio_file.default_filename.split(" - ")[-2])
        sleep(0.5)
        # exit(0)
        # audio_file.download()

