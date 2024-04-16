import asyncio
from pyppeteer import launch
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import os


async def get_all_website_links(url):
    browser = await launch(headless=True)
    page = await browser.newPage()
    await page.goto(url)
    page_content = await page.content()
    soup = BeautifulSoup(page_content, "html.parser")
    links = set()
    for a_tag in soup.findAll("a"):
        href = a_tag.get("href")
        if href and href.startswith("http"):
            links.add(href)
    await browser.close()
    return links


async def crawl(url, max_depth=1):
    visited_urls = set()

    async def _crawl(urls, depth):
        if depth > max_depth:
            return
        new_urls = set()
        for url in urls:
            if url not in visited_urls:
                visited_urls.add(url)
                print(f"Crawling: {url}")
                try:
                    links = await get_all_website_links(url)
                    new_urls.update(links)
                    browser = await launch(headless=True)
                    page = await browser.newPage()
                    await page.goto(url)
                    page_content = await page.content()
                    soup = BeautifulSoup(page_content, 'html.parser')
                    text = soup.get_text()
                    path = urlparse(url).path
                    if not path:
                        path = "/index"
                    filename = f"{domain}/{path.replace('/', '_')}.txt"
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    with open(filename, "w", encoding='utf-8') as f:
                        f.write(text)
                    await browser.close()
                except Exception as e:
                    print(f"Failed to crawl {url}: {e}")
        await _crawl(new_urls, depth + 1)

    domain = urlparse(url).netloc
    os.makedirs(domain, exist_ok=True)
    await _crawl([url], 0)


# Usage
asyncio.run(crawl("https://www.elal.com/heb/baggage", max_depth=2))
