import time
import Downloader
import re

if __name__ == "__main__":
    test = re.compile('(The Daily Gwei Refuel \d+)?=* - Ethereum Updates.mp4')
    Downloader.downloadChannelAudio('TheDailyGwei', test, 10, 10)
    # results = Downloader.transcribeAudio('medium.en', 'TheDailyGwei')
    results = Downloader.transcribeAudio('tiny.en', 'TheDailyGwei')
    for result in results:
        print(result)
