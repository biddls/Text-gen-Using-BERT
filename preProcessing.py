import glob
import json
import os
import re
import webbrowser


def getText(_dir: str) -> dict:
    files = list(glob.iglob(f'{_dir}/text/*.txt'))
    a = dict()
    with open(f'{_dir}/data.txt', 'w') as f:
        for file, text in zip(files, getTextGen(_dir)):
            a[file] = text[0]
            f.write(f'{text[0]}\n')

    with open(f'{_dir}/data.json', 'w') as outfile:
        json.dump(a, outfile)
    return a


def getTextGen(_dir: str) -> [str, str]:
    files = list(glob.iglob(f'{_dir}/text/*.txt'))
    for file in files:
        with open(file, 'r') as f:
            yield f.readlines()[0], file


class pre_processing:
    def __init__(self, _dir: str):
        files = getText(_dir)
        self.files = list(zip(files.keys(), files.values()))

    def __enter__(self):
        # stuff
        return self

    def __next__(self):
        if len(self.files) > 0:
            return self.files.pop()
        else:
            raise StopIteration

    def __iter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __delitem__(self, key):
        if os.path.exists(key):
            os.remove(key)
        else:
            raise FileNotFoundError(f'{key} not found')


if __name__ == "__main__":
    channel = "TheDailyGwei"
    getText(channel)
    with pre_processing(channel) as pre:
        for text, file in pre:
            print(file.split("\\")[-1].split(".txt")[0], text[:250])
            webbrowser.open(file)
