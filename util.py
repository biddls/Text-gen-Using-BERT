import os
from typing import List


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
        #download the dataset
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