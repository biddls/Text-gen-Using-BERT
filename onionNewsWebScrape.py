from bs4 import BeautifulSoup
import requests
from typing import Tuple, List
from tqdm import tqdm
import itertools
from multiprocessing import Pool

monthLinkClass = "sc-zpw6hx-0"
articleLinkClass = "sc-1w8kdgf-0"
newsHeadClass = "sc-1efpnfq-0"
newsBodyClass = "sc-77igqf-0"
videoClass = "lhhce6-0"

url = "https://www.theonion.com"


# Function to get all links from a page with _url of class _class
def get_links(_url: str, _class: str) -> List[str]:
    """
    This function takes in a URL string and a class string and returns a list
    of all links from a page with that URL and class.

    Args:
        _url (str): A string representing the URL of the page to scrape.
        _class (str): A string representing the class of the div containing the links.

    Returns:
        List[str]: A list of strings representing the URLs of the links found.
    """
    # Make a request to the given URL
    page = requests.get(_url)  # This might throw an exception if something goes wrong.

    # Create a BeautifulSoup object to parse the HTML content of the page
    soup = BeautifulSoup(page.text, 'html.parser')

    # Find the div with the given class (using a lambda function to match partial matches)
    try:
        link_div = soup.find_all('div', attrs={'class': lambda e: e.startswith(_class) if e else False})[0]
    except IndexError:
        raise IndexError(f"Error: {_url}, with {_class}")

    # Find all the links within the div and extract their URLs
    links = link_div.findAll('a')
    links = [link.get('href') for link in links]

    # Return the list of URLs found
    return links


def extractText(_url: str, _done: bool = True) -> Tuple[str, str]:
    """
    This function takes in a URL string and a boolean flag (default=True) and returns a tuple
    of two strings: the first is the text of the page's H1 heading with class 'newsHeadClass',
    and the second is the text of the page's first paragraph with class 'newsBodyClass'.

    If the function encounters an SSLError or ConnectionError, it will attempt to retry the
    request with the _done flag set to False (to prevent infinite recursion), and if this also
    fails, it will return an empty tuple of strings.

    If the page does not have an H1 heading or first paragraph with the expected class, the
    function will return an empty tuple of strings.

    Args:
        _url (str): A string representing the URL of the page to scrape.
        _done (bool): A boolean flag indicating whether the function has already attempted
                      to retry the request (default=True).

    Returns:
        Tuple[str, str]: A tuple of two strings representing the text of the H1 heading and
                         first paragraph with the expected classes (or empty strings if not found).
    """
    try:
        # Make a request to the given URL
        page = requests.get(_url)  # This might throw an exception if something goes wrong.
    except (requests.exceptions.SSLError, requests.exceptions.ConnectionError):
        # If the request fails due to an SSL error or connection error, and we haven't already
        # retried the request, try again with the _done flag set to False.
        if _done:
            return extractText(_url, _done=False)
        else:
            # If we've already retried the request, and it still failed, return an empty tuple of strings.
            return "", ""

    # Create a BeautifulSoup object to parse the HTML content of the page
    soup = BeautifulSoup(page.text, 'html.parser')

    try:
        # Find the H1 heading with the expected class
        head = soup.find_all('h1', attrs={'class': lambda _e: _e.startswith(newsHeadClass) if _e else False})[0].text

        # Find the first paragraph with the expected class
        body = soup.find_all('p', attrs={'class': lambda _e: _e.startswith(newsBodyClass) if _e else False})[0].text

        # If the H1 heading is the same as the body, assume we haven't found the expected elements
        if head == body:
            return "", ""
        else:
            # Return the text of the H1 heading and first paragraph
            return head, body

    except IndexError as e:
        # If we couldn't find the expected elements, check if there is a video on the page
        a = soup.find_all('div', attrs={'class': lambda _e: _e.startswith(videoClass) if _e else False})
        if not a:
            # If there is no video, return an empty tuple of strings
            return "", ""
        # If there is a video, print an error message and raise the IndexError
        print(f"Error: {_url}")
        raise e


def batched_extractText(_urls: List[str], _p: Pool) -> List[str]:
    """
    This function takes in a list of URL strings and a multiprocessing Pool object, and returns
    a list of strings representing the text of the H1 heading and first paragraph for each page.

    The function uses the multiprocessing Pool object to parallelize the extraction of text from
    the pages, and returns the results as a list of strings.

    Args:
        _urls (List[str]): A list of strings representing the URLs of the pages to scrape.
        _p (Pool): A multiprocessing Pool object used to parallelize the extraction of text.

    Returns:
        List[str]: A list of strings representing the text of the H1 heading and first paragraph
                   for each page.
    """
    # Use the map_async method of the multiprocessing Pool object to parallelize the extraction
    # of text from the pages.
    results = _p.map_async(extractText, _urls).get()

    # Return the results as a list of strings.
    return results


def main() -> None:
    """
    Scrape news article titles and bodies from The Onion website and save them to a file.

    Returns:
    None

    """
    # Get the links to the monthly sitemaps from the main page, and print some information about them.
    monthLinks = get_links(url + "/sitemap", monthLinkClass)
    print(f"{len(monthLinks)} months have been found.")
    print(f"Oldest is {monthLinks[-1].replace('/sitemap/', '')}")
    print(f"and newest is {monthLinks[0].replace('/sitemap/', '')}")

    # Construct the full URLs for the monthly sitemaps.
    monthLinks = [url + link for link in monthLinks]

    # Get the links to the individual articles from the monthly sitemaps, and print some information
    # about them.
    articleLinks = [get_links(monthLink, articleLinkClass) for monthLink in tqdm(monthLinks, desc="Months")]
    articleLinks = list(itertools.chain(*articleLinks))
    print(f"{len(articleLinks)} articles have been found.")

    # Extract the text of the H1 heading and first paragraph for each article, using multiprocessing
    # to speed up the process.
    text = []
    batch_size = 60
    batch_counter = tqdm(range(0, len(articleLinks), batch_size), total=len(articleLinks), desc="Articles")
    with Pool(batch_size) as p:
        for x in batch_counter:
            text += batched_extractText(articleLinks[x:x + batch_size], p)
            batch_counter.update(batch_size)

    # Filter out any articles that didn't have both a non-empty heading and a non-empty body text.
    text = [x for x in text if x != ("", "")]

    # Write the text of each article to a file.
    with open("onion/NewsWebScrape.txt", mode="w", encoding="utf-8") as f:
        for article in text:
            if article:
                f.write(f"{article[0]} #~# {article[1]}\n")

    # Print some information about the number of articles found and written to file.
    print(f"{len(articleLinks)} articles where found, and {len(text)} articles where written to file.")


if __name__ == "__main__":
    main()
