import glob
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def tokeniseFile(file: str) -> [str]:
    # Open the file
    with open(file, "r", encoding= 'unicode_escape') as f:
        text = f.read()

    # Tokenise the text
    encoding = tokenizer.encode(text)

    return encoding

def tokeniseDir(dir: str) -> [str]:
    # Get all the files in the directory
    files = glob.glob(f"{dir}/text/*.txt")

    # Tokenise each file
    files = [tokeniseFile(file) for file in files]

    return files


if __name__ == "__main__":
    tokens = tokeniseDir("sirswag")
    for x in tokens:
        print(tokenizer.convert_ids_to_tokens(x[:11]))