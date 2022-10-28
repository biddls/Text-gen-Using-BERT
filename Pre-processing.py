import glob
import json
import re
import webbrowser



def getText(_dir: str):
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


if __name__ == "__main__":
    getText("TheDailyGwei")
    for text, file in getTextGen("TheDailyGwei"):
        if not re.search("(L|l)et\'?s get into it\.?", text):
            print(file.split("\\")[-1].split(".txt")[0], text[:250])
            webbrowser.open(file)
