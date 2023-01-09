import os
def splitOnion():
    if os.getcwd() != "onion":
        os.chdir("onion")

    if not os.path.exists("data"):
        os.mkdir("data")

    with open("script.txt", "r") as f:
        a = f.readlines()

    for index, line in enumerate(a):
        line = line.replace("\n", " ")
        with open(f"data/{index}.txt", "w") as f:
            f.write(line)

    os.chdir("..")

if __name__ == "__main__":
    splitOnion()
