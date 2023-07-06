################################################################################
### Step 1
################################################################################

import requests
import re
import urllib.request
from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
from typing import List
from concurrent.futures import ThreadPoolExecutor, Future
import retry

from chatgpt_util import ChatGPTUtil

token_count = 0
# Regex pattern to match a URL
HTTP_URL_PATTERN = r'^http[s]{0,1}://.+$'

# Create a class to parse the HTML and get the hyperlinks
class HyperlinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        # Create a list to store the hyperlinks
        self.hyperlinks = []

    # Override the HTMLParser's handle_starttag method to get the hyperlinks
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)

        # If the tag is an anchor tag and it has an href attribute, add the href attribute to the list of hyperlinks
        if tag == "a" and "href" in attrs:
            self.hyperlinks.append(attrs["href"])

# Function to get the hyperlinks from a URL
@retry.retry(tries=3, delay=1, backoff=2)
def get_hyperlinks(url):
    
    # Try to open the URL and read the HTML
    try:
        # Open the URL and read the HTML
        with urllib.request.urlopen(url) as response:

            # If the response is not HTML, return an empty list
            if not response.info().get('Content-Type').startswith("text/html"):
                return []
            
            # Decode the HTML
            html = response.read().decode('utf-8')
    except Exception as e:
        print(e)
        return []

    # Create the HTML Parser and then Parse the HTML to get hyperlinks
    parser = HyperlinkParser()
    parser.feed(html)

    return parser.hyperlinks

def get_domain_hyperlinks(local_domain, url):
    clean_links = []
    for link in set(get_hyperlinks(url)):
        clean_link = None

        # If the link is a URL, check if it is within the same domain
        if re.search(HTTP_URL_PATTERN, link):
            # Parse the URL and check if the domain is the same
            url_obj = urlparse(link)
            if url_obj.netloc == local_domain:
                clean_link = link

        # If the link is not a URL, check if it is a relative link
        else:
            if link.startswith("/"):
                link = link[1:]
            elif (
                link.startswith("#")
                or link.startswith("mailto:")
                or link.startswith("tel:")
            ):
                continue
            clean_link = "https://" + local_domain + "/" + link

        if clean_link is not None:
            if clean_link.endswith("/"):
                clean_link = clean_link[:-1]
            clean_links.append(clean_link)

    # Return the list of hyperlinks that are within the same domain
    return list(set(clean_links))


def summarize(text: str, chat_gpt_util: ChatGPTUtil) -> str:
    _SYSTEM_PROMPT = """\
    You will be provided with the text scraped from a companies website in the following format:
    ### COMPANY TEXT ###
    <company text>
    ####################
    
    Given this context, summarize the page's main points in 3-5 bullet points. If the content is not 
    related to core buisness offerings (eg, a blog post), please output nothing.
    """
    
    _USER_PROMPT = """\
        ### COMPANY TEXT ###
        {COMPANY_TEXT}
        ####################
        """

    formatted_user_prompt = _USER_PROMPT.format(COMPANY_TEXT=text)
    messages = [
        {'role': 'system', 'content': _SYSTEM_PROMPT},
        {'role': 'user', 'content': formatted_user_prompt}]

    summary = chat_gpt_util.get_chat_completion(
        messages=messages,
        model='gpt-4')[0]
    return summary

def _get_summarized_page(url: str, chat_gpt_util: ChatGPTUtil) -> str:
    
    try:
        print(url) # for debugging and to see the progress
        # Get the text from the URL using BeautifulSoup
        soup = BeautifulSoup(requests.get(url).text, "html.parser")

        # Get the text but remove the tags
        text = soup.get_text()

        # If the crawler gets to a page that requires JavaScript, it will stop the crawl
        if ("You need to enable JavaScript to run this app." in text):
            print("Unable to parse page " + url + " due to JavaScript being required")
        
        # Otherwise, write the text to the file in the text directory
        
        summarized_text = summarize(text, chat_gpt_util)
        cleaned_text = remove_newlines(summarized_text)    
        return cleaned_text
    except Exception as e:
        print(f"Error getting page {url}: {e}")
        return ""


def crawl(url, chat_gpt_util: ChatGPTUtil, tokenizer, max_num_tokens: int = 3_000):
    # Parse the URL and get the domain
    local_domain = urlparse(url).netloc
    # Create a queue to store the URLs to crawl
    queue = deque([url])
    # Create a set to store the URLs that have already been seen (no duplicates)
    seen = set([url])
    # While the queue is not empty, continue crawling
    futures: List[Future] = []
    
    website_text = ""
    token_count = 0
    
    with ThreadPoolExecutor() as executor:
        
        while queue: 
            url = queue.pop()
            futures.append(executor.submit(_get_summarized_page, url, chat_gpt_util))
            
            for link in get_domain_hyperlinks(local_domain, url):
                if link not in seen:
                    queue.append(link)
                    seen.add(link)
                    
            if len(futures) == 10 or not queue:
                
                print(f"pausing at 10")
                results = [future.result() for future in futures]
                futures = []
                for page_text in results:
                    page_token_count = len(tokenizer.encode(page_text))
                    if token_count + page_token_count > max_num_tokens:
                        return website_text
                    website_text += page_text
                    token_count += page_token_count
                    print(f"token count: {token_count}")
                
                if not queue:
                    print(f"returning at end of queue")
                    return website_text
            
            
    
def remove_newlines(text: str) -> str:
    text = text.replace('\n', ' ')
    text = text.replace('\\n', ' ')
    text = text.replace('  ', ' ')
    text = text.replace('  ', ' ')
    return text


def get_ai_suggestions(text: str, chat_gpt_util: ChatGPTUtil) -> str:
    
    _SYSTEM_PROMPT = """\
    You will be provided with the text scraped from a companies website in the following format:
    ### COMPANY TEXT ###
    <company text>
    ####################
    
    Given this context, summarize the buisness's main services in 2-3 sentences. 
    
    Then, based on the buisness's services, suggest potential AI use cases for the buisness. 
    For each use case, explain in laymans's terms how the AI system would work and how it would benefit the buisness.
    """
    
    _USER_PROMPT = """\
        ### COMPANY TEXT ###
        {COMPANY_TEXT}
        ####################
        """

    formatted_user_prompt = _USER_PROMPT.format(COMPANY_TEXT=text)
    messages = [
        {'role': 'system', 'content': _SYSTEM_PROMPT},
        {'role': 'user', 'content': formatted_user_prompt}]

    ai_suggestions = chat_gpt_util.get_chat_completion(
        messages=messages,
        model='gpt-4')[0]
    return ai_suggestions

