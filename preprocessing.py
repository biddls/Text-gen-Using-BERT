import glob
import json


def getText(_dir: str):
    files = list(glob.iglob(f'{_dir}/text/*.txt'))
    a = dict()
    with open(f'{_dir}/data.txt', 'w') as f:
        for file, text in zip(files, getTextGen(_dir)):
            a[file] = text
            f.write(f'{text}\n')

    with open(f'{_dir}/data.json', 'w') as outfile:
        json.dump(a, outfile)
    return a


def getTextGen(_dir: str):
    files = list(glob.iglob(f'{_dir}/text/*.txt'))
    for file in files:
        with open(file, 'r') as f:
            yield f.readlines()[0]


if __name__ == "__main__":
    getText("TheDailyGwei")
    for x in getTextGen("TheDailyGwei"):
        print(x[:250])
