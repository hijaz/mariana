import hashlib
import io
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import trafilatura
from langchain.tools import tool
from markdownify import markdownify
from readability import Document

from mariana.utils.config import load_config

# ── Domain block list ─────────────────────────────────────────────────────────

ALWAYS_BLOCKED = {
    # Paywalled academic
    "sciencedirect.com",
    "researchgate.net",
    "jstor.org",
    "ieee.org",
    "springer.com",
    "nature.com",
    "thelancet.com",
    "cell.com",
    "pubmed.ncbi.nlm.nih.gov",
    # Social / media
    "facebook.com",
    "twitter.com",
    "x.com",
    "instagram.com",
    "tiktok.com",
    "youtube.com",
    "reddit.com",
    # Junk / low-quality
    "quora.com",
    "realitypathing.com",
    "ihavenotv.com",
    "pinterest.com",
    "linkedin.com",
}

# ── Domain quality scoring ────────────────────────────────────────────────────

DOMAIN_SCORES: dict[str, float] = {
    "wikipedia.org": 0.9,
    "britannica.com": 0.85,
    "iaea.org": 0.9,
    "iter.org": 0.9,
    "energy.gov": 0.9,
    "nasa.gov": 0.9,
    "nationalacademies.org": 0.9,
    "mit.edu": 0.9,
    "stanford.edu": 0.9,
    "bbc.com": 0.75,
    "reuters.com": 0.75,
    "theguardian.com": 0.7,
    "reddit.com": 0.15,
    "quora.com": 0.15,
    "facebook.com": 0.0,
}


def score_domain(url: str) -> float:
    """Return a quality score 0–1 for a URL's domain."""
    try:
        domain = urlparse(url).netloc.replace("www.", "")
    except Exception:
        return 0.5
    if domain in DOMAIN_SCORES:
        return DOMAIN_SCORES[domain]
    if domain.endswith(".gov"):
        return 0.85
    if domain.endswith(".edu") or domain.endswith(".ac.uk"):
        return 0.85
    return 0.5


def _should_skip(url: str) -> bool:
    """Return True if this URL should never be scraped."""
    try:
        domain = urlparse(url).netloc.replace("www.", "")
    except Exception:
        return True
    return any(blocked in domain for blocked in ALWAYS_BLOCKED)


# ── Scrape cache ──────────────────────────────────────────────────────────────

_CACHE_DIR = Path.home() / ".mariana" / "scrape_cache"
_CACHE_TTL = timedelta(days=7)


def _cache_key(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


def _cache_get(url: str) -> str | None:
    path = _CACHE_DIR / _cache_key(url)
    if not path.exists():
        return None
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
        age = datetime.now() - datetime.fromisoformat(meta["scraped_at"])
        if age > _CACHE_TTL:
            return None
        return meta["content"]
    except Exception:
        return None


def _cache_set(url: str, content: str) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / _cache_key(url)
    try:
        path.write_text(
            json.dumps({"url": url, "scraped_at": datetime.now().isoformat(), "content": content}),
            encoding="utf-8",
        )
    except Exception:
        pass


# ── Browser-realistic headers ─────────────────────────────────────────────────

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


# ── PDF helpers ───────────────────────────────────────────────────────────────

def _is_pdf_url(url: str) -> bool:
    if url.lower().split("?")[0].endswith(".pdf"):
        return True
    try:
        resp = httpx.head(url, headers=_HEADERS, timeout=5.0, follow_redirects=True)
        return "pdf" in resp.headers.get("content-type", "").lower()
    except Exception:
        return False


def _scrape_pdf(url: str, max_chars: int) -> str:
    try:
        import pypdf

        resp = httpx.get(url, headers=_HEADERS, timeout=20.0, follow_redirects=True)
        reader = pypdf.PdfReader(io.BytesIO(resp.content))
        pages = [page.extract_text() or "" for page in reader.pages[:10]]
        text = "\n\n".join(p for p in pages if p.strip())
        return text[:max_chars]
    except Exception:
        return ""


# ── Post-scrape noise cleaning ───────────────────────────────────────────────

_NOISE_PATTERNS = [
    r'^(accept|cookie policy|subscribe|sign up|log in|enable javascript).*\n',
    r'\$\d+\.\d+ per (month|year)',
    r'(this page requires javascript|please enable)',
    r'^advertisement\s*\n',
    r'^share this article.*\n',
]


def clean_scraped_content(text: str) -> str:
    for pattern in _NOISE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ── Core scraper ──────────────────────────────────────────────────────────────

def _do_scrape(url: str, max_chars: int) -> str:
    if _is_pdf_url(url):
        return _scrape_pdf(url, max_chars)

    # Tier 1: trafilatura — density-based, handles most sites including Wikipedia
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=True,
                no_fallback=False,
                favor_recall=True,
            )
            if text and len(text) > 200:
                return clean_scraped_content(text[:max_chars])
    except Exception:
        pass

    # Tier 2: readability-lxml fallback
    try:
        resp = httpx.get(url, headers=_HEADERS, timeout=12.0, follow_redirects=True)
        # Strip null bytes and XML-incompatible control characters before parsing
        safe_html = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', resp.text)
        doc = Document(safe_html)
        text = markdownify(doc.summary(), strip=["a", "img"])
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if len(text) > 200:
            return clean_scraped_content(text[:max_chars])
    except Exception:
        pass

    return ""


def raw_scrape(url: str, max_chars: int = 20000) -> str:
    """Fetch and extract text from a URL. Returns empty string if blocked or fails."""
    if _should_skip(url):
        return ""

    cached = _cache_get(url)
    if cached is not None:
        return cached[:max_chars]

    content = _do_scrape(url, max_chars)
    _cache_set(url, content)
    return content


# ── Search ────────────────────────────────────────────────────────────────────

def raw_search(query: str, port: int, num_results: int = 5) -> list[dict[str, Any]]:
    """Call local SearXNG and return filtered, scored results."""
    try:
        params = {"q": query, "format": "json", "language": "en", "safesearch": 0}
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"http://localhost:{port}/search", params=params)
            resp.raise_for_status()
            data = resp.json()

        results = []
        for item in data.get("results", []):
            url = item.get("url", "")
            if _should_skip(url):
                continue
            results.append({
                "title": item.get("title", ""),
                "url": url,
                "snippet": item.get("snippet", ""),
                "score": score_domain(url),
            })
            if len(results) >= num_results:
                break
        # Sort by domain quality so best sources come first
        results.sort(key=lambda r: r["score"], reverse=True)
        return results
    except Exception:
        return []


def filter_for_diversity(
    results: list[dict],
    section_domain_counts: dict,
    session_domain_counts: dict,
    max_per_section: int = 1,
    max_per_session: int = 2,
) -> list[dict]:
    """Filter results so no domain dominates a section or the whole session."""
    filtered = []
    for r in results:
        domain = urlparse(r["url"]).netloc.replace("www.", "")
        if (
            section_domain_counts.get(domain, 0) < max_per_section
            and session_domain_counts.get(domain, 0) < max_per_session
        ):
            filtered.append(r)
            section_domain_counts[domain] = section_domain_counts.get(domain, 0) + 1
            session_domain_counts[domain] = session_domain_counts.get(domain, 0) + 1
    return filtered


# ── LangChain tools (for agent use) ──────────────────────────────────────────────────


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
