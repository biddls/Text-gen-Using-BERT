from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
import itertools
from multiprocessing import Pool

monthlink_class = "sc-zpw6hx-0"
articleLink_class = "sc-1w8kdgf-0"
newshead_class = "sc-1efpnfq-0"
newsbody_class = "sc-77igqf-0"
video_class = "lhhce6-0"

url = "https://www.theonion.com"


def get_links(_url: str, _class: str) -> [str]:
    page = requests.get(_url)  # this might throw an exception if something goes wrong.
    soup = BeautifulSoup(page.text, 'html.parser')
    try:
        link_div = soup.find_all('div', attrs={'class': lambda e: e.startswith(_class) if e else False})[0]
    except IndexError:
        raise IndexError(f"Error: {_url}, with {_class}")
    links = link_div.findAll('a')
    links = [link.get('href') for link in links]
    return links


def extractText(_url: str, _done: bool = True) -> (str, str):
    try:
        page = requests.get(_url)  # this might throw an exception if something goes wrong.
    except requests.exceptions.SSLError or requests.exceptions.ConnectionError:
        if _done:
            return extractText(_url, _done=False)
        else:
            return "", ""
    soup = BeautifulSoup(page.text, 'html.parser')
    try:
        head = soup.find_all('h1', attrs={'class': lambda e: e.startswith(newshead_class) if e else False})[0].text
        body = soup.find_all('p', attrs={'class': lambda e: e.startswith(newsbody_class) if e else False})[0].text
        if head == body:
            return "", ""
        else:
            return head, body
    except IndexError as e:
        a = soup.find_all('div', attrs={'class': lambda e: e.startswith(video_class) if e else False})
        if not a:
            return "", ""
        print(f"Error: {_url}")
        raise e


def batched_extractText(_urls: [str], _batch_size: int, _p: Pool) -> [str]:
    return _p.map_async(extractText, _urls).get()


if __name__ == "__main__":
    monthLinks = get_links(url + "/sitemap", monthlink_class)
    print(f"{len(monthLinks)} months have been found.")
    print(f"Oldest is {monthLinks[-1].replace('/sitemap/', '')}")
    print(f"and newest is {monthLinks[0].replace('/sitemap/', '')}")
    monthLinks = [url + link for link in monthLinks]

    articleLinks = [get_links(monthLink, articleLink_class) for monthLink in tqdm(monthLinks, desc="Months")]
    articleLinks = list(itertools.chain(*articleLinks))

    print(f"{len(articleLinks)} articles have been found.")

    text = []
    batch_size = 60
    batch_counter = tqdm(range(0, len(articleLinks), batch_size), total=len(articleLinks), desc="Articles")
    with Pool(batch_size) as p:
        for x in batch_counter:
            text += batched_extractText(articleLinks[x:x + batch_size], batch_size, p)
            batch_counter.update(batch_size)
    text = [x for x in text if x != ("", "")]
    with open("onion/NewsWebScrape.txt", mode="w", encoding="utf-8") as f:
        for article in text:
            if article:
                f.write(f"{article[0]} #~# {article[1]}\n")

    print(f"{len(articleLinks)} articles where found, and {len(text)} articles where written to file.")
