from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api import TranscriptsDisabled
from tqdm import tqdm
import glob
from transformers import BertTokenizer


def loadTransScriptions(file: str) -> [str]:
    with open(f"{file}.txt", "r") as f:
        urls = f.readlines()

    urls = [i[:11] for i in urls]
    output = []
    with open("script.txt", "w", encoding="UTF-8") as f:
        for index, url in tqdm(enumerate(urls), total=len(urls)):
            try:
                script = YouTubeTranscriptApi.get_transcript(url)
                script = ' '.join([i['text'].replace('\n', ' ') for i in script])
                script = script.replace("[Music] ", "")
                script = script.replace("Captions by www.SubPLY.com . . . . . . . . . . . . . ", "")
                script = script.replace("  ", " ")
                f.write(script + "\n")
                output.append(script)
            except TranscriptsDisabled:
                print(f"Failed to get transcript for the url: {url} at: {index}")
                pass

    return output


def tokeniseFile(file: str) -> [str]:
    # Open the file
    with open(file, "r", encoding= 'unicode_escape') as f:
        while True:
            text = f.readline()
            if text == "":
                break

            # Tokenise the text
            encoding = tokenizer.encode(text)

            yield encoding


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # loadTransScriptions("temp")
    # runs the BERT tokenizer on the script.txt file and trunkates to 512 tokens
    tokens = [token[:512] for token in tokeniseFile("script.txt")]