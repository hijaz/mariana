import re
from typing import Any

import httpx
from bs4 import BeautifulSoup
from langchain.tools import tool
from markdownify import markdownify as md

from mariana.utils.config import load_config

SKIP_DOMAINS = {
    "youtube.com",
    "www.youtube.com",
    "twitter.com",
    "x.com",
    "www.x.com",
    "instagram.com",
    "www.instagram.com",
    "facebook.com",
    "www.facebook.com",
    "tiktok.com",
    "www.tiktok.com",
}


def raw_search(query: str, port: int, num_results: int = 5) -> list[dict[str, str]]:
    try:
        params = {
            "q": query,
            "format": "json",
            "language": "en",
            "safesearch": 0,
        }
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"http://localhost:{port}/search", params=params)
            resp.raise_for_status()
            data = resp.json()

        results = []
        for item in data.get("results", [])[:num_results]:
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("snippet", ""),
                }
            )
        return results
    except Exception:
        return []


def raw_scrape(url: str, max_chars: int = 4000) -> str:
    if any(host in url for host in SKIP_DOMAINS):
        return ""

    try:
        headers = {"User-Agent": "MarianaResearch/0.1 (local research agent)"}
        resp = httpx.get(url, headers=headers, timeout=12.0, follow_redirects=True)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag_name in ["script", "style", "nav", "footer", "header", "aside", "form", "noscript", "iframe"]:
            for el in soup.find_all(tag_name):
                el.decompose()

        target = soup.find("article") or soup.find("main") or soup.find(id="content") or soup.find(class_="content") or soup.body
        if target is None:
            return ""

        markdown = md(str(target), heading_style="ATX", strip=["a", "img"])
        markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip()
        return markdown[:max_chars]
    except Exception:
        return ""


@tool
def searxng_search(query: str) -> str:
    "Search the web using local SearXNG instance. Args: query: search query"
    cfg = load_config()
    results = raw_search(query, cfg.searxng_port, cfg.max_results)
    if not results:
        return "No results found."

    formatted = []
    for idx, item in enumerate(results, start=1):
        formatted.append(f"{idx}. {item['title']}\n{item['url']}\n{item['snippet']}")
    return "\n\n".join(formatted)


@tool
def scrape_page(url: str) -> str:
    "Fetch and extract text content from a URL. Args: url: target URL"
    cfg = load_config()
    return raw_scrape(url, cfg.max_page_chars)
