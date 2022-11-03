import glob
import json
import os
from nltk.corpus import words
from nltk.tokenize import WhitespaceTokenizer


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


# takes in an array and returns a paired list of words with their config label
def cleanText(_text: [str]) -> [(str, str)]:
    d = enchant.Dict("en_US")
    text_return = list()

    # removes all "words" that are in the english dictionary
    _text = list(set(_text) - set(words.words()))

    # cleans the text and only returns the problem ones
    for word in _text:
        # if the word has characters in it
        if re.search(r"[a-zA-Z]+", word):
            # if the word ends in punctuation
            word = word[:-1] if word[-1] in ",!.;?" else word
            if not d.check(word):
                text_return.append((word, 'txt'))
        else:
            if re.search(r"\%$", word):
                text_return.append((word, '<%>'))
            else:
                text_return.append((word, '?'))

    return text_return


class pre_processing:
    def __init__(self, _dir: str):
        self._dir = _dir
        files = getText(_dir)
        self.files = list(zip(files.keys(), files.values()))
        self.dict = files

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
        with open(f'{self._dir}/data.json', 'w') as outfile:
            json.dump(self.dict, outfile)
        with open(f'{self._dir}/data.txt', 'w') as f:
            for _text in self.dict.values():
                f.write(f'{_text}\n')


if __name__ == "__main__":
    import webbrowser
    import re
    from tqdm import tqdm
    import enchant

    channel = "TheDailyGwei"
    tokenizer = WhitespaceTokenizer()
    total = list()
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
                problem = cleanText(text)
                total += problem

                pre[file] = text
    total = list(set(total))
    [print(word) for word in total]
    print(len(total))
    print('done')
