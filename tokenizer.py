import glob
import json
import os
from nltk.corpus import words
from nltk.tokenize import WhitespaceTokenizer
import numpy as np


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
        self._dir = _dir
        files = getText(_dir)
        self.files = list(zip(files.keys(), files.values()))
        self.dict = files
        with open(f"{_dir}/config.json", "r") as json_file:
            self.config = json.load(json_file)
        for x in ["percentages", "numbers", "money", "website", "other"]:
            self.config[x] = list()

    def __enter__(self):
        return self

    # Returns the key and text of the file
    def __next__(self):
        if len(self.files) > 0:
            return self.files.pop()
        else:
            raise StopIteration

    def __len__(self):
        return len(self.files)

    def __iter__(self):
        return self

    # Deletes the file from the directory if it exists
    def __delitem__(self, key):
        if os.path.exists(key):
            os.remove(key)
        else:
            raise FileNotFoundError(f'{key} not found')

    # Sets the value of the dictionary
    def __setitem__(self, key, value):
        self.dict[key] = value

    def __getitem__(self, item):
        return self.dict[item]

    # saves the data to the directory
    def __exit__(self, exc_type, exc_val, exc_tb):
        with open(f'{self._dir}/data.json', 'w') as json_file:
            json.dump(self.dict, json_file, indent=4)
        with open(f'{self._dir}/data.txt', 'w') as f:
            for _text in self.dict.values():
                f.write(f'{_text}\n')
        with open(f'{self._dir}/config.json', 'w') as json_file:
            json.dump(self.config, json_file, indent=4)

    # takes in an array and returns a paired list of words with their config label
    def labelText(self, _text: [str]) -> [(str, str)]:
        d = enchant.Dict("en_US")
        text_return = list()

        # removes all "words" that are in the english dictionary
        _text = list(set(_text) - set(words.words()))

        # cleans the text and only returns the problem ones
        for word in _text:
            # skip anything with a token
            if re.search(r"^<.>$", word):
                continue
            # if it has 2 or more "." in it
            if re.search(r"\.{2,}", word):
                word = word.replace(".", "")
            # if it is of length 2 and is in "0-9a-zA-Z" range
            if re.search(r"^[0-9a-zA-Z]{2}$", word):
                continue
            # if the word has characters in it
            if re.search(r"[a-zA-Z]+", word):
                # if word has '.' in it
                if re.search(r"(.+\.)+.+", word):
                    if word not in self.config[self.config["<W>"]]:
                        self.config[self.config["<W>"]].append(word)
                else:
                    # if the word ends in punctuation
                    while word[-1] in ",!.;?":
                        word = word[:-1]
                    if not d.check(word):
                        if word not in self.config[self.config["<?>"]]:
                            self.config[self.config["<?>"]].append(word)
                        text_return.append((word, '<w>'))
            else:
                if re.search(r"\%\.?", word):
                    word = word[:-1] if word[-1] in ",!.;?" else word
                    if word not in self.config[self.config["<#>"]]:
                        self.config[self.config["<#>"]].append(word)
                else:
                    if re.search(r"[0-9]+", word):
                        word = word[:-1] if word[-1] == ".?" else word
                        word = word.replace(',', '')
                        try:
                            if word not in self.config[self.config["<#>"]]:
                                self.config[self.config["<#>"]].append(word)
                        except ValueError:
                            if word not in self.config[self.config["<#>"]]:
                                self.config[self.config["<#>"]].append(word)

        return text_return


if __name__ == "__main__":
    import webbrowser
    import re
    from tqdm import tqdm
    import enchant

    channel = "TheDailyGwei"
    tokenizer = WhitespaceTokenizer()
    total = list()
    length = 0
    with pre_processing(channel) as pre:
        for file, text in tqdm(pre, position=0):
            # if it cannot find a match it opens a text editor
            if not (match := re.search("(L|l)et\'?s get into it\.?", text)):
                print(file.split("\\")[-1].split(".txt")[0], text[:250])
                webbrowser.open(file)

            # if the position of the match is not at the beginning of the text
            match = match.span()
            leng = match[1]
            if leng > 300:
                raise IndexError(f'{text[:leng]} match pos is too long \n{file}')
            else:
                text = f'<I>{text[leng:]}'
                text = tokenizer.tokenize(text)
                length += len(text)
                problem = pre.labelText(text)
                total += problem

                pre[file] = text
    total = list(set(total))
    [print(word) for word in total]
    print(len(total))
    print(length)
    with open(f"{channel}/toFix.txt", "w+") as f:
        for line in total:
            f.write(str(line))
            f.write("\n")
    print('done')
