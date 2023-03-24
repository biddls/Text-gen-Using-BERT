import os
from typing import List
import torch.utils.data
from glob import glob

from tqdm import tqdm


def split_proportionally(num: int, proportions: List[int]) -> List[int]:
    # calculate the total sum of integers that need to be split
    if sum(proportions) != 100:
        raise ValueError("Proportions must sum to 100")
    proportions = [p / 100 for p in proportions]
    # calculate the integer value of each proportion
    integer_proportions = [int(num * p) for p in proportions]
    # calculate the remaining integer value that needs to be allocated
    remaining = num - sum(integer_proportions)
    # distribute the remaining value among the integer proportions, based on the proportion values
    proportions = sorted(range(len(proportions)), key=lambda x: proportions[x], reverse=True)
    for i in proportions[:remaining]:
        integer_proportions[i] += 1
    if sum(integer_proportions) != num:
        raise ValueError("Integer proportions do not sum to num")
    return integer_proportions


def dataset_checks():
    # downloads the BBC news data set
    if not os.path.exists("bbc"):
        # download the dataset
        from io import BytesIO
        from zipfile import ZipFile
        from urllib.request import urlopen

        # open url
        resp = urlopen("http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip")
        # read zipfile
        zipfile = ZipFile(BytesIO(resp.read()))
        with zipfile as zip_ref:
            zip_ref.extractall("")
        os.remove("bbc/README.TXT")

    # split the pre-processed data set into individual files
    if not os.path.exists("onion/data"):
        import onion.split as onion_split
        onion_split.splitOnion()
        if not os.path.exists("onion/data"):
            raise FileNotFoundError("onion/data not found")


class Text_DataSet(torch.utils.data.Dataset):
    def __init__(self, _files, tokenizer):
        # load data
        corpus = []
        files = list(glob(_files))
        _len = len(files)
        match _len:
            case _ if _len > 1:
                for file in tqdm(files, desc=f"Loading files from {_files}"):
                    with open(file, 'r', encoding="UTF-8") as f:
                        corpus.append(f.read())
            case 1:
                with open(files[0], 'r', encoding="UTF-8") as f:
                    corpus = f.readlines()

                corpus = [x.split(' #~# ')[1] for x in corpus]
            case _:
                raise FileNotFoundError(f"No file Found {_files}")

        if tokenizer is not None:
            inputs = tokenizer(
                corpus,
                add_special_tokens=True,
                padding='max_length',
                return_tensors='pt',
                truncation=True,
            )
        else:
            raise TypeError("Tokenizer Not set")

        inputs['labels'] = inputs.input_ids.detach().clone()

        # create random array of floats with equal dimensions to input_ids tensor
        rand = torch.rand(inputs.input_ids.shape)

        # create mask array
        # doesn't mask over any PAD, CLS or SEP tokens (0, 101, 102)
        mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * \
                    (inputs.input_ids != 102) * (inputs.input_ids != 0)

        selection = []

        for i in range(inputs.input_ids.shape[0]):
            selection.append(
                torch.flatten(mask_arr[i].nonzero()).tolist()
            )

        for i in range(inputs.input_ids.shape[0]):
            inputs.input_ids[i, selection[i]] = 103

        self.encodings = inputs
        print(f'Dataset "{_files}" loaded')

    def __getitem__(self, idx):
        return {key: val[idx].clone().detach() for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


if __name__ == "__main__":
    onion = Text_DataSet('onion/NewsWebScrape.txt', None)
