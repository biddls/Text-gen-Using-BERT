from Downloader import downloadChannelAudio, transcribeDirAudio
import re

if __name__ == "__main__":
    test = re.compile('(The Daily Gwei Refuel \d+)?=* - Ethereum Updates.mp4')
    downloadChannelAudio('TheDailyGwei', test, 10, 2)
    results = transcribeDirAudio('medium.en', 'TheDailyGwei')
