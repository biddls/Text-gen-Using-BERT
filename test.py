from Downloader import downloadChannelAudio, transcribeDirAudio
import re

if __name__ == "__main__":
    channel = 'TheDailyGwei'
    test = re.compile('(The Daily Gwei Refuel \d+)?=* - Ethereum Updates.mp4')
    downloadChannelAudio(channel, test, 200, 10)
    results = transcribeDirAudio('medium.en', channel)
