from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api import TranscriptsDisabled
from tqdm import tqdm


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


if __name__ == '__main__':
    loadTransScriptions("temp")