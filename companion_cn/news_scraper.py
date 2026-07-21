#!/usr/bin/env python3
"""
News scraper for 6 Chinese official media sources.
Scrapes every 5 hours, summarizes with Qwen, deduplicates, stores in SQLite.

Sources: 新华网, 人民网, 央视新闻, 央广网, 中国军网, 中国政府网,
         环球网, 参考消息, 海外网

Usage:
  python scripts/news_scraper.py          # run once, scrape all sources
  python scripts/news_scraper.py --daemon # run in a loop, every 5 hours
  python scripts/news_scraper.py --once   # (default) scrape once and exit
"""
import sqlite3
import os
import sys
import re
import json
import time
import hashlib
import logging
from datetime import datetime, timezone, timedelta
from xml.etree import ElementTree as ET
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import feedparser
from bs4 import BeautifulSoup
from openai import OpenAI

from .config import QWEN_MODEL, QWEN_URL, QWEN_CHAT_TEMPLATE_KWARGS

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "news.db")
LOG_PATH = os.path.join(BASE_DIR, "data", "news_scraper.log")

# ── Qwen client ───────────────────────────────────────────────────────────────
# Reuse the same endpoint as the companion service.  Keeping a second hard-coded
# address here made the scraper silently fail after the main LLM moved.
_llm = OpenAI(base_url=QWEN_URL, api_key="x")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("news_scraper")

# ── Database ──────────────────────────────────────────────────────────────────


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS news_articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                summary TEXT DEFAULT '',
                url TEXT UNIQUE NOT NULL,
                source TEXT NOT NULL,
                category TEXT DEFAULT '',
                country TEXT DEFAULT '',
                published_at TEXT DEFAULT '',
                scraped_at TEXT NOT NULL,
                title_hash TEXT DEFAULT '',
                is_summarized INTEGER DEFAULT 0
            )
        """)
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(news_articles)").fetchall()
        }
        if "country" not in columns:
            conn.execute("ALTER TABLE news_articles ADD COLUMN country TEXT DEFAULT ''")
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_news_source ON news_articles(source)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_news_published ON news_articles(published_at DESC)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_news_hash ON news_articles(title_hash)
        """)
        conn.commit()


def article_exists(url: str) -> bool:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT 1 FROM news_articles WHERE url=? LIMIT 1", (url,)).fetchone()
        return row is not None


def similar_title_exists(title: str, threshold: float = 0.7) -> bool:
    """Check if a very similar title already exists (cross-source dedup)."""
    h = _title_hash(title)
    with sqlite3.connect(DB_PATH) as conn:
        # Check recent articles (last 3 days) from other sources
        cutoff = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
        rows = conn.execute(
            "SELECT title_hash, source FROM news_articles WHERE published_at > ? AND title_hash != ''",
            (cutoff,),
        ).fetchall()
    for (existing_hash, _) in rows:
        if _hash_similarity(h, existing_hash) > threshold:
            return True
    return False


def insert_article(title: str, url: str, source: str, category: str = "",
                   published_at: str = "", country: str = "") -> Optional[int]:
    """Insert a new article. Returns id, or None if duplicate."""
    now = datetime.now(timezone.utc).isoformat()
    title_hash = _title_hash(title)
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """INSERT INTO news_articles
                   (title, url, source, category, published_at, scraped_at, title_hash, country)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (title, url, source, category, published_at, now, title_hash, country),
            )
            conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def get_unsummarized(limit: int = 30) -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM news_articles WHERE is_summarized=0 AND summary='' ORDER BY id LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def save_summary(article_id: int, summary: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE news_articles SET summary=?, is_summarized=1 WHERE id=?",
            (summary, article_id),
        )
        conn.commit()


# ── Title hashing & similarity ────────────────────────────────────────────────


def _title_hash(title: str) -> str:
    """Hash a normalized title for dedup comparison."""
    normalized = _normalize_title(title)
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def _normalize_title(title: str) -> str:
    """Strip punctuation, whitespace, and common prefixes for comparison."""
    t = title.strip()
    # Remove common prefixes
    for prefix in ["（受权发布）", "（权威发布）", "【", "）", "】", "「", "」"]:
        t = t.replace(prefix, "")
    # Keep only Chinese chars, letters, digits
    t = re.sub(r"[^\u4e00-\u9fff\w]", "", t)
    return t.lower()


def _hash_similarity(h1: str, h2: str) -> float:
    """Jaccard similarity of two MD5 hashes' character bigrams.
    Quick coarse filter — real dedup uses exact URL match."""
    if h1 == h2:
        return 1.0
    # Compare bigrams of the hex strings
    bg1 = set(h1[i : i + 2] for i in range(len(h1) - 1))
    bg2 = set(h2[i : i + 2] for i in range(len(h2) - 1))
    if not bg1 or not bg2:
        return 0.0
    return len(bg1 & bg2) / len(bg1 | bg2)


# ── HTTP helpers ──────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9",
}

_session = requests.Session()
_session.headers.update(HEADERS)


def fetch_url(url: str, timeout: int = 15) -> Optional[str]:
    """Fetch a URL and return text content, or None on failure."""
    try:
        r = _session.get(url, timeout=timeout)
        r.raise_for_status()
        # Detect encoding
        r.encoding = r.apparent_encoding or "utf-8"
        return r.text
    except Exception as e:
        log.warning(f"Failed to fetch {url}: {e}")
        return None


# ── RSS parsers ───────────────────────────────────────────────────────────────


def parse_rss_feed(url: str, source: str) -> list[dict]:
    """Parse an RSS/Atom feed and return list of article dicts."""
    articles = []
    try:
        feed = feedparser.parse(url)
        if feed.bozo and not feed.entries:
            log.warning(f"RSS parse warning for {url}: {feed.bozo_exception}")
        for entry in feed.entries[:30]:  # Max 30 per source per run
            title = entry.get("title", "").strip()
            link = entry.get("link", "")
            published = entry.get("published", "") or entry.get("updated", "")
            summary = entry.get("summary", "") or entry.get("description", "")
            # Clean summary HTML tags
            summary = re.sub(r"<[^>]+>", "", summary)[:300]
            if title and link:
                articles.append({
                    "title": title,
                    "url": link,
                    "source": source,
                    "category": _guess_category(title, summary),
                    "published_at": _parse_time(published),
                    "raw_summary": summary,
                })
        log.info(f"  RSS {source}: {len(articles)} articles from {url}")
    except Exception as e:
        log.warning(f"  RSS {source} failed: {e}")
    return articles


def _parse_time(s: str) -> str:
    """Try to parse a time string into ISO format."""
    if not s:
        return ""
    try:
        from dateutil.parser import parse as dt_parse  # optional
        return dt_parse(s).isoformat()
    except Exception:
        pass
    # Try struct_time from feedparser
    if hasattr(s, "tm_year"):
        try:
            dt = datetime(*s[:6], tzinfo=timezone.utc)
            return dt.isoformat()
        except Exception:
            pass
    return s[:25] if len(s) > 10 else ""


# ── HTML scrapers ─────────────────────────────────────────────────────────────

# Each source has a scraping function that returns list[dict]


def scrape_xinhuanet() -> list[dict]:
    """新华网 — try RSS first, fall back to HTML."""
    articles = []
    # Try RSS
    rss_urls = [
        "http://www.xinhuanet.com/politics/xhll.xml",  # 时政
    ]
    for url in rss_urls:
        articles.extend(parse_rss_feed(url, "新华网"))

    # Fallback: HTML scraping of politics page
    if len(articles) < 5:
        html = fetch_url("https://www.xinhuanet.com/politics/")
        if html:
            soup = BeautifulSoup(html, "lxml")
            for a in soup.select("a[href]"):
                title = a.get_text(strip=True)
                href = a.get("href", "")
                if not title or len(title) < 6 or not href:
                    continue
                # Filter: must look like a news article link
                if not any(kw in title for kw in ["习", "国", "政", "会", "发", "新", "中", "部", "局", "省", "市", "委", "军"]):
                    continue
                full_url = href if href.startswith("http") else f"https://www.xinhuanet.com{href}"
                articles.append({
                    "title": title,
                    "url": full_url,
                    "source": "新华网",
                    "category": "时政",
                    "published_at": "",
                    "raw_summary": "",
                })
    return articles[:30]


def scrape_people() -> list[dict]:
    """人民网 — try RSS first, fallback to HTML."""
    articles = []
    rss_urls = [
        "http://politics.people.com.cn/rss/politics.xml",
        "http://politics.people.com.cn/rss/GB/1024/index.xml",
    ]
    for url in rss_urls:
        articles.extend(parse_rss_feed(url, "人民网"))

    if len(articles) < 5:
        # Try main site with Referer
        _session.headers["Referer"] = "https://www.people.com.cn/"
        html = fetch_url("https://www.people.com.cn/")
        if not html:
            html = fetch_url("http://world.people.com.cn/")
        if html:
            soup = BeautifulSoup(html, "lxml")
            for a in soup.select("a[href]"):
                title = a.get_text(strip=True)
                href = a.get("href", "")
                if not title or len(title) < 8 or not href:
                    continue
                if not any(kw in title for kw in ["习", "国", "政", "会", "发", "新", "中", "部", "局", "省", "市", "委"]):
                    continue
                full_url = href if href.startswith("http") else f"https://www.people.com.cn{href}"
                articles.append({
                    "title": title,
                    "url": full_url,
                    "source": "人民网",
                    "category": "时政",
                    "published_at": "",
                    "raw_summary": "",
                })
        _session.headers.pop("Referer", None)
    return articles[:30]


def scrape_cctv() -> list[dict]:
    """央视新闻 — HTML scraping (no reliable RSS)."""
    articles = []
    html = fetch_url("https://news.cctv.com/")
    if html:
        soup = BeautifulSoup(html, "lxml")
        for a in soup.select("a[href]"):
            title = a.get_text(strip=True)
            href = a.get("href", "")
            if not title or len(title) < 6 or not href:
                continue
            if "cctv.com" not in href and not href.startswith("/"):
                continue
            full_url = href if href.startswith("http") else f"https://news.cctv.com{href}"
            articles.append({
                "title": title,
                "url": full_url,
                "source": "央视新闻",
                "category": _guess_category(title, ""),
                "published_at": "",
                "raw_summary": "",
            })
    return articles[:30]


def scrape_cnr() -> list[dict]:
    """央广网 — RSS + HTML."""
    articles = []
    rss_urls = ["http://news.cnr.cn/"]
    for url in rss_urls:
        articles.extend(parse_rss_feed(url, "央广网"))

    if len(articles) < 5:
        html = fetch_url("https://news.cnr.cn/")
        if html:
            soup = BeautifulSoup(html, "lxml")
            for a in soup.select("a[href]"):
                title = a.get_text(strip=True)
                href = a.get("href", "")
                if not title or len(title) < 6 or not href:
                    continue
                full_url = href if href.startswith("http") else f"https://news.cnr.cn{href}"
                articles.append({
                    "title": title,
                    "url": full_url,
                    "source": "央广网",
                    "category": "",
                    "published_at": "",
                    "raw_summary": "",
                })
    return articles[:30]


def scrape_81cn() -> list[dict]:
    """中国军网 — HTML scraping."""
    articles = []
    html = fetch_url("http://www.81.cn/")
    if html:
        soup = BeautifulSoup(html, "lxml")
        for a in soup.select("a[href]"):
            title = a.get_text(strip=True)
            href = a.get("href", "")
            if not title or len(title) < 6 or not href:
                continue
            full_url = href if href.startswith("http") else f"http://www.81.cn{href}"
            articles.append({
                "title": title,
                "url": full_url,
                "source": "中国军网",
                "category": "军事",
                "published_at": "",
                "raw_summary": "",
            })
    return articles[:30]


def scrape_govcn() -> list[dict]:
    """中国政府网 — RSS + HTML."""
    articles = []
    # Try known RSS endpoints
    rss_urls = [
        "http://www.gov.cn/zhuanti/rss.htm",
    ]
    for url in rss_urls:
        articles.extend(parse_rss_feed(url, "中国政府网"))

    if len(articles) < 5:
        html = fetch_url("https://www.gov.cn/")
        if html:
            soup = BeautifulSoup(html, "lxml")
            for a in soup.select("a[href]"):
                title = a.get_text(strip=True)
                href = a.get("href", "")
                if not title or len(title) < 6 or not href:
                    continue
                full_url = href if href.startswith("http") else f"https://www.gov.cn{href}"
                articles.append({
                    "title": title,
                    "url": full_url,
                    "source": "中国政府网",
                    "category": "政务",
                    "published_at": "",
                    "raw_summary": "",
                })
    return articles[:30]


def _guess_category(title: str, summary: str) -> str:
    """Guess news category from title/summary keywords."""
    text = title + summary
    cats = {
        "时政": ["习近", "政治局", "国务院", "总理", "外交", "访问", "会谈"],
        "军事": ["军队", "国防", "军事", "演习", "海军", "空军", "火箭军", "战区"],
        "经济": ["经济", "GDP", "增长", "贸易", "金融", "股市", "企业", "产业"],
        "民生": ["医疗", "教育", "养老", "住房", "交通", "环保", "天气", "疫情"],
        "国际": ["美国", "日本", "俄罗", "欧洲", "国际", "世界", "联合国"],
        "科技": ["科技", "AI", "人工智能", "航天", "卫星", "5G", "芯片", "互联网"],
        "文化": ["文化", "历史", "文物", "非遗", "艺术", "文学", "电影"],
    }
    for cat, keywords in cats.items():
        if any(kw in text for kw in keywords):
            return cat
    return "综合"


# ── Country classification ───────────────────────────────────────────────────

# Chinese names for 100+ countries/regions — keyword matching on title+summary
_COUNTRY_MAP = {
    "美国": ["美国", "美方", "华盛顿", "白宫", "五角大楼", "纽约", "洛杉矶", "加州"],
    "日本": ["日本", "东京", "安倍", "岸田", "日元", "日方"],
    "韩国": ["韩国", "首尔", "韩方", "尹锡悦", "青瓦台"],
    "朝鲜": ["朝鲜", "平壤", "金正恩", "朝方"],
    "俄罗斯": ["俄罗斯", "俄方", "莫斯科", "普京", "克里姆林宫"],
    "英国": ["英国", "伦敦", "英方", "唐宁街", "首相"],
    "法国": ["法国", "巴黎", "法方", "马克龙", "爱丽舍宫"],
    "德国": ["德国", "柏林", "德方", "默茨", "朔尔茨", "慕尼黑"],
    "印度": ["印度", "新德里", "印方", "莫迪"],
    "乌克兰": ["乌克兰", "基辅", "乌方", "泽连斯基"],
    "以色列": ["以色列", "特拉维夫", "耶路撒冷", "以方"],
    "巴勒斯坦": ["巴勒斯坦", "加沙", "哈马斯"],
    "伊朗": ["伊朗", "德黑兰", "伊方"],
    "沙特": ["沙特", "利雅得"],
    "阿联酋": ["阿联酋", "迪拜", "阿布扎比"],
    "澳大利亚": ["澳大利亚", "澳洲", "悉尼", "堪培拉", "澳方"],
    "加拿大": ["加拿大", "多伦多", "渥太华", "加方"],
    "巴西": ["巴西", "巴西利亚", "圣保罗"],
    "南非": ["南非", "约翰内斯堡", "开普敦"],
    "欧盟": ["欧盟", "欧洲", "布鲁塞尔", "欧方"],
    "东盟": ["东盟", "东南亚"],
    "联合国": ["联合国", "安理会"],
    "土耳其": ["土耳其", "安卡拉", "伊斯坦布尔"],
    "巴基斯坦": ["巴基斯坦", "伊斯兰堡", "巴方"],
    "越南": ["越南", "河内", "胡志明"],
    "泰国": ["泰国", "曼谷"],
    "缅甸": ["缅甸", "仰光", "内比都"],
    "菲律宾": ["菲律宾", "马尼拉"],
    "印尼": ["印尼", "印度尼西亚", "雅加达"],
    "马来西亚": ["马来西亚", "吉隆坡"],
    "新加坡": ["新加坡"],
    "墨西哥": ["墨西哥", "墨西哥城"],
    "阿根廷": ["阿根廷", "布宜诺斯艾利斯"],
    "意大利": ["意大利", "罗马", "米兰"],
    "西班牙": ["西班牙", "马德里"],
    "荷兰": ["荷兰", "阿姆斯特丹"],
    "瑞士": ["瑞士", "日内瓦", "苏黎世"],
    "瑞典": ["瑞典", "斯德哥尔摩"],
    "挪威": ["挪威", "奥斯陆"],
    "波兰": ["波兰", "华沙"],
    "捷克": ["捷克", "布拉格"],
    "希腊": ["希腊", "雅典"],
    "埃及": ["埃及", "开罗"],
    "尼日利亚": ["尼日利亚", "阿布贾"],
    "肯尼亚": ["肯尼亚", "内罗毕"],
    "埃塞俄比亚": ["埃塞俄比亚"],
    "伊拉克": ["伊拉克", "巴格达"],
    "叙利亚": ["叙利亚", "大马士革"],
    "阿富汗": ["阿富汗", "喀布尔", "塔利班"],
    "台湾": ["台湾", "台北", "台方"],
    "香港": ["香港"],
    "澳门": ["澳门"],
}

# Country aliases — canonical name from any occurrence
_COUNTRY_ALIASES = {
    "美": "美国", "日": "日本", "韩": "韩国", "俄": "俄罗斯",
    "英": "英国", "法": "法国", "德": "德国", "印": "印度",
    "乌": "乌克兰", "以": "以色列", "伊": "伊朗", "澳": "澳大利亚",
}


def _classify_country(title: str, summary: str = "") -> str:
    """Identify which country/region a news article is about.
    Returns country name in Chinese, or empty string if unclear."""
    text = title + " " + summary
    scores = {}
    for country, keywords in _COUNTRY_MAP.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scores[country] = score
    if not scores:
        return ""
    # Return the country with most keyword matches
    return max(scores, key=scores.get)


# ── Summarization with Qwen ───────────────────────────────────────────────────

SUMMARY_PROMPT = """把下面这则新闻总结成1-2句中文摘要（不超过80字）。只输出摘要，不要任何解释。

新闻标题：{title}

新闻内容或摘要：{raw}

摘要："""


def summarize_articles(articles: list[dict]) -> list[dict]:
    """Batch summarize unsummarized articles. Returns updated list."""
    if not articles:
        return articles

    to_summarize = [a for a in articles if not a.get("summary") and a.get("title")]
    if not to_summarize:
        return articles

    log.info(f"  Summarizing {len(to_summarize)} articles with Qwen...")
    results = []
    for art in to_summarize:
        try:
            raw = art.get("raw_summary", "") or art.get("title", "")
            prompt = SUMMARY_PROMPT.format(title=art["title"], raw=raw[:200])
            r = _llm.chat.completions.create(
                model=QWEN_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=120,
                extra_body={"chat_template_kwargs": QWEN_CHAT_TEMPLATE_KWARGS},
                timeout=10,
            )
            summary = r.choices[0].message.content.strip()
            art["summary"] = summary
        except Exception as e:
            log.warning(f"  Summarize failed for '{art['title'][:30]}': {e}")
            art["summary"] = ""
    return articles


# ── International news sources ─────────────────────────────────────────────────


def scrape_huanqiu() -> list[dict]:
    """环球网 — major Chinese international news outlet."""
    articles = []
    html = fetch_url("https://www.huanqiu.com/")
    if not html:
        html = fetch_url("https://world.huanqiu.com/")
    if html:
        soup = BeautifulSoup(html, "lxml")
        for a in soup.select("a[href]"):
            title = a.get_text(strip=True)
            href = a.get("href", "")
            if not title or len(title) < 6 or not href:
                continue
            if "huanqiu.com" not in href and not href.startswith("/"):
                continue
            full_url = href if href.startswith("http") else f"https://www.huanqiu.com{href}"
            country = _classify_country(title, "")
            articles.append({
                "title": title,
                "url": full_url,
                "source": "环球网",
                "category": "国际",
                "country": country,
                "published_at": "",
                "raw_summary": "",
            })
    return articles[:30]


def scrape_cankaoxiaoxi() -> list[dict]:
    """参考消息 — digests of foreign media reports."""
    articles = []
    html = fetch_url("https://www.cankaoxiaoxi.com/")
    if html:
        soup = BeautifulSoup(html, "lxml")
        for a in soup.select("a[href]"):
            title = a.get_text(strip=True)
            href = a.get("href", "")
            if not title or len(title) < 6 or not href:
                continue
            full_url = href if href.startswith("http") else f"https://www.cankaoxiaoxi.com{href}"
            country = _classify_country(title, "")
            articles.append({
                "title": title,
                "url": full_url,
                "source": "参考消息",
                "category": "国际",
                "country": country,
                "published_at": "",
                "raw_summary": "",
            })
    return articles[:30]


def scrape_haiwainet() -> list[dict]:
    """海外网 — People's Daily overseas edition."""
    articles = []
    html = fetch_url("https://www.haiwainet.cn/")
    if html:
        soup = BeautifulSoup(html, "lxml")
        for a in soup.select("a[href]"):
            title = a.get_text(strip=True)
            href = a.get("href", "")
            if not title or len(title) < 6 or not href:
                continue
            full_url = href if href.startswith("http") else f"https://www.haiwainet.cn{href}"
            country = _classify_country(title, "")
            articles.append({
                "title": title,
                "url": full_url,
                "source": "海外网",
                "category": "国际",
                "country": country,
                "published_at": "",
                "raw_summary": "",
            })
    return articles[:30]


# ── Main scraper ──────────────────────────────────────────────────────────────

SOURCES = [
    ("新华网", scrape_xinhuanet),
    ("人民网", scrape_people),
    ("央视新闻", scrape_cctv),
    ("央广网", scrape_cnr),
    ("中国军网", scrape_81cn),
    ("中国政府网", scrape_govcn),
    ("环球网", scrape_huanqiu),
    ("参考消息", scrape_cankaoxiaoxi),
    ("海外网", scrape_haiwainet),
]


def scrape_all() -> dict:
    """Scrape all sources. Returns stats dict."""
    init_db()
    stats = {"sources": {}, "total_new": 0, "total_dup": 0, "errors": []}

    for source_name, scraper in SOURCES:
        log.info(f"Scraping {source_name}...")
        try:
            articles = scraper()
            new_count = 0
            for art in articles:
                # Check URL-based dedup
                if article_exists(art["url"]):
                    stats["total_dup"] += 1
                    continue
                # Check title similarity (cross-source)
                if similar_title_exists(art["title"]):
                    stats["total_dup"] += 1
                    continue
                # Insert
                if insert_article(
                    title=art["title"],
                    url=art["url"],
                    source=art["source"],
                    category=art.get("category", ""),
                    published_at=art.get("published_at", ""),
                    country=art.get("country", ""),
                ):
                    new_count += 1
                    stats["total_new"] += 1
            stats["sources"][source_name] = new_count
            log.info(f"  {source_name}: {new_count} new, from {len(articles)} scraped")
        except Exception as e:
            log.error(f"  {source_name} ERROR: {e}")
            stats["errors"].append({"source": source_name, "error": str(e)})
            stats["sources"][source_name] = 0

    # Summarize new articles
    unsummarized = get_unsummarized(limit=30)
    if unsummarized:
        summarized = summarize_articles(unsummarized)
        for art in summarized:
            if art.get("summary"):
                save_summary(art["id"], art["summary"])

    log.info(f"Done. New: {stats['total_new']}, Duplicates: {stats['total_dup']}, "
             f"Summarized: {len(unsummarized)}")
    return stats


def daemon_loop(interval_hours: int = 5):
    """Run scraper continuously every N hours."""
    log.info(f"Starting daemon mode — scrape every {interval_hours} hours")
    while True:
        try:
            scrape_all()
        except Exception as e:
            log.error(f"Scrape cycle failed: {e}")
        next_run = datetime.now() + timedelta(hours=interval_hours)
        log.info(f"Next scrape at {next_run.strftime('%H:%M')}")
        time.sleep(interval_hours * 3600)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="News scraper for Chinese official media")
    parser.add_argument("--daemon", action="store_true", help="Run continuously every 5 hours")
    parser.add_argument("--interval", type=int, default=5, help="Hours between scrapes (daemon mode)")
    parser.add_argument("--once", action="store_true", default=True, help="Run once and exit (default)")
    args = parser.parse_args()

    print(f"DB path: {DB_PATH}")
    print(f"Log path: {LOG_PATH}")

    if args.daemon:
        daemon_loop(args.interval)
    else:
        stats = scrape_all()
        print(f"\nResults: {json.dumps(stats, ensure_ascii=False, indent=2)}")
