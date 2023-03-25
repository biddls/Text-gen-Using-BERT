import os

def splitOnion(file: str, directory: str = "onion"):

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(file, "r", encoding="UTF-8") as f:
        a = f.readlines()

    for index, line in enumerate(a):
        line = line.replace("\n", " ")
        try:
            with open(f"{directory}/{index}.txt", "w", encoding="UTF-8") as f:
                f.write(line)
        except UnicodeEncodeError as e:
            print(index)
            raise e

if __name__ == "__main__":
    splitOnion("NewsWebScrape.txt", "WebData/")
