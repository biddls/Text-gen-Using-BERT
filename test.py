import Downloader
import re


if __name__ == "__main__":
    test = re.compile("(The Daily Gwei Refuel \d+)?=* - Ethereum Updates.mp4")
    Downloader.downloadChannelAudio("TheDailyGwei", test, 10, 10)
