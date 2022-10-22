import time

import whisper

import Downloader
import re

if __name__ == "__main__":
    test = re.compile('(The Daily Gwei Refuel \d+)?=* - Ethereum Updates.mp4')
    Downloader.downloadChannelAudio('TheDailyGwei', test, 10, 10)
    s = time.time()
    print(Downloader.transcribeAudio('medium.en', 'TheDailyGwei')[0][:200])
    print(f"{'medium.en'} took {time.time() - s} seconds")
    # for result in results:
    #     print(result)
