"""External tools: time, weather, news — called by tool_detect node."""
import os
import sqlite3
import requests
import re
import time
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone


def get_time_info() -> dict:
    """Current date/time for system prompt injection. Always succeeds."""
    now = datetime.now()
    wdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    return {
        "type": "time",
        "text": f"现在是{now.year}年{now.month}月{now.day}日 {wdays[now.weekday()]} {now.hour:02d}:{now.minute:02d}",
        "ok": True,
    }


def get_weather(city: str = "北京") -> dict:
    """Fetch current weather from wttr.in. Free, no key needed."""
    try:
        r = requests.get(
            f"https://wttr.in/{city}?format=j1",
            timeout=6,
            headers={"User-Agent": "companion-bot/1.0"},
        )
        r.raise_for_status()
        cur = r.json()["current_condition"][0]
        return {
            "type": "weather",
            "text": (
                f"{city}天气：{cur['weatherDesc'][0]['value']}，"
                f"温度{cur['temp_C']}°C（体感{cur['FeelsLikeC']}°C），"
                f"湿度{cur['humidity']}%，{cur['winddir16Point']}风{cur['windspeedKmph']}km/h"
            ),
            "ok": True,
        }
    except Exception:
        return {"type": "weather", "text": "", "ok": False}


_DB_PATH = os.path.join(os.path.dirname(__file__), "data", "news.db")


# ---------------------------------------------------------------------------
# Current office holders — live official sources, never model memory
# ---------------------------------------------------------------------------
_OFFICIAL_CACHE_TTL_SECONDS = 6 * 60 * 60
_OFFICIAL_CACHE: dict[str, dict] = {}

_OFFICIAL_SPECS = (
    {
        "office": "美国总统",
        "keywords": ("美国总统", "美國總統"),
        "url": "https://www.whitehouse.gov/administration/",
        "source": "美国白宫官网",
        "parser": "us",
    },
    {
        "office": "澳门特别行政区行政长官",
        "keywords": ("澳门特首", "澳門特首", "澳门行政长官", "澳門行政長官"),
        "url": "https://www.gov.mo/zh-hant/about-government/chief-executive-principal-officials-legislature-and-judiciary/",
        "source": "澳门特别行政区政府官网",
        "parser": "macau",
    },
    {
        "office": "日本首相",
        "keywords": ("日本首相", "日本总理", "日本總理"),
        "url": "https://japan.kantei.go.jp/",
        "source": "日本首相官邸官网",
        "parser": "japan",
    },
    {
        "office": "英国首相",
        "keywords": ("英国首相", "英國首相", "英国总理", "英國總理"),
        "url": "https://www.gov.uk/government/ministers/prime-minister",
        "source": "英国政府官网",
        "parser": "uk",
    },
    {
        "office": "法国总统",
        "keywords": ("法国总统", "法國總統"),
        "url": "https://www.elysee.fr/en/french-presidency/the-presidents-of-the-republic",
        "source": "法国总统府官网",
        "parser": "france",
    },
    {
        "office": "韩国总统",
        "keywords": ("韩国总统", "韓國總統"),
        "url": "https://en.president.go.kr/president",
        "source": "韩国总统府官网",
        "parser": "korea",
    },
    {
        "office": "俄罗斯总统",
        "keywords": ("俄罗斯总统", "俄羅斯總統"),
        "url": "https://en.kremlin.ru/",
        "source": "俄罗斯总统官网",
        "parser": "russia",
    },
    {
        "office": "香港特别行政区行政长官",
        "keywords": ("香港特首", "香港行政长官", "香港行政長官"),
        "url": "https://www.ceo.gov.hk/en/",
        "source": "香港特别行政区行政长官办公室官网",
        "parser": "hong_kong",
    },
)


def is_current_official_query(user_query: str) -> bool:
    """Whether a query asks for one of the live office-holder lookups."""
    return any(keyword in user_query for spec in _OFFICIAL_SPECS for keyword in spec["keywords"])


def expand_current_official_followup(user_query: str, history: list[dict]) -> str:
    """Expand a short country follow-up after an office-holder question.

    For example, after "日本首相是谁", "英国呢" means "英国首相是谁".
    It is deliberately limited to short follow-ups and a recent supported
    office query, so ordinary country/news questions still use their own path.
    """
    text = user_query.strip()
    if len(text) > 12 or not any(marker in text for marker in ("呢", "那", "也", "？", "?")):
        return ""
    recent = history[-8:]
    if recent and recent[-1].get("role") == "user" and recent[-1].get("content") == user_query:
        recent = recent[:-1]
    has_office_context = any(
        item.get("role") == "user" and is_current_official_query(item.get("content", ""))
        for item in recent
    )
    if not has_office_context:
        return ""
    aliases = (
        ("美国", "美国总统"), ("澳门", "澳门特首"), ("日本", "日本首相"),
        ("英国", "英国首相"), ("法国", "法国总统"), ("韩国", "韩国总统"),
        ("俄罗斯", "俄罗斯总统"), ("香港", "香港特首"),
    )
    return next((query for country, query in aliases if country in text), "")


def get_current_official(user_query: str) -> dict:
    """Look up supported office holders from their official websites.

    The result is intentionally conservative: no successful fetch and parse
    means no answerable fact. This prevents stale model knowledge or news
    articles from being presented as a current office holder.
    """
    spec = next(
        (item for item in _OFFICIAL_SPECS if any(keyword in user_query for keyword in item["keywords"])),
        None,
    )
    if not spec:
        return {"type": "official", "ok": False, "verified": False, "text": ""}

    office = spec["office"]
    cached = _OFFICIAL_CACHE.get(office)
    now = time.time()
    if cached and now - cached["cached_at"] < _OFFICIAL_CACHE_TTL_SECONDS:
        return cached["result"]

    try:
        response = requests.get(
            spec["url"],
            timeout=10,
            headers={"User-Agent": "CompanionCN/1.0 (+current-office-lookup)"},
        )
        response.raise_for_status()
        page_text = BeautifulSoup(response.content, "html.parser").get_text(" ", strip=True)

        parser = spec["parser"]
        if parser == "us":
            match = re.search(
                r"President\s+([A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){1,2})",
                page_text,
            )
            name = match.group(1).strip() if match else ""
        elif parser == "macau":
            # The official page contains navigation labels before the actual
            # person. Accept only a 2-5 character name after 行政長官.
            candidates = re.findall(r"行政長官\s+([^\s，,。；;（）()]{2,8})", page_text)
            name = next(
                (candidate.strip() for candidate in candidates
                 if 2 <= len(candidate.strip()) <= 5 and "主要" not in candidate and "立法" not in candidate),
                "",
            )
        elif parser == "japan":
            # The homepage contains news titles such as "Prime Minister X
            # spoke...". Locate the newest Cabinet page and extract the
            # dedicated Prime Minister heading instead.
            cabinet_numbers = re.findall(r"/(\d{2,3})/meibo/", response.text)
            if not cabinet_numbers:
                raise ValueError("current cabinet page link not found")
            cabinet_number = max(cabinet_numbers, key=int)
            cabinet_url = f"https://japan.kantei.go.jp/{cabinet_number}/meibo/index.html"
            cabinet_response = requests.get(
                cabinet_url,
                timeout=10,
                headers={"User-Agent": "CompanionCN/1.0 (+current-office-lookup)"},
            )
            cabinet_response.raise_for_status()
            cabinet_text = BeautifulSoup(cabinet_response.content, "html.parser").get_text(" ", strip=True)
            match = re.search(
                r"The\s+\d+(?:st|nd|rd|th)\s+Prime Minister\s+([A-Z]{2,}\s+[A-Z][A-Za-z.'-]+)",
                cabinet_text,
            )
            name = match.group(1).strip() if match else ""
        elif parser == "uk":
            match = re.search(
                r"Current role holder\s*:\s*(?:The Rt Hon )?([A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){1,4})",
                page_text,
            )
            name = match.group(1).strip() if match else ""
        elif parser == "france":
            match = re.search(r"([A-Z][a-z]+\s+[A-Z][a-z]+)\s+In office", page_text)
            name = match.group(1).strip() if match else ""
        elif parser == "korea":
            match = re.search(
                r"President\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})",
                page_text,
            )
            name = match.group(1).strip() if match else ""
        elif parser == "russia":
            match = re.search(
                r"([A-Z][a-z]+\s+[A-Z][a-z]+)\s+President of Russia|President of Russia\s+([A-Z][a-z]+\s+[A-Z][a-z]+)",
                page_text,
            )
            name = (match.group(1) or match.group(2)).strip() if match else ""
        elif parser == "hong_kong":
            match = re.search(r"(?:Image:\s*)?Chief Executive,?\s*([A-Z][A-Za-z ]+)", page_text)
            name = match.group(1).strip() if match else ""
        if not name:
            raise ValueError("official page did not contain a parseable office holder")

        checked_at = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")
        result = {
            "type": "official",
            "ok": True,
            "verified": True,
            "office": office,
            "name": name,
            "source": spec["source"],
            "url": spec["url"],
            "checked_at": checked_at,
            "text": (
                f"【实时职务查询】\n{office}：{name}\n"
                f"来源：{spec['source']}\n查询时间：{checked_at}\n"
                "[只依据以上实时查询回答；不要使用模型记忆、新闻库或猜测补充。]"
            ),
        }
    except Exception:
        result = {
            "type": "official",
            "ok": True,
            "verified": False,
            "office": office,
            "text": (
                f"【实时职务查询】\n{office}：当前未查到可靠的实时官方资料。\n"
                "[直接说无法确认，不要猜测姓名，不要让用户去问亲友。]"
            ),
        }

    _OFFICIAL_CACHE[office] = {"cached_at": now, "result": result}
    return result


def format_current_official_answer(result: dict) -> str:
    """Render verified office-holder data without sending it back to the LLM."""
    if result.get("verified"):
        return (
            f"根据{result['source']}（{result['checked_at']}查询），"
            f"{result['office']}是{result['name']}。"
        )
    return "我现在没查到可靠的实时官方资料，不能确定。"


def get_news(user_query: str = "") -> dict:
    """Fetch recent news headlines. Reads from local scraped DB first,
    falls back to external hot-search APIs if DB is empty.
    If user_query contains country/topic keywords, matching articles are boosted."""
    # Layer 1: local scraped news (from official media, summarized)
    try:
        if os.path.exists(_DB_PATH):
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=72)).isoformat()
            with sqlite3.connect(_DB_PATH) as conn:
                conn.row_factory = sqlite3.Row

                # Check if user is asking about specific countries
                asked_countries = []
                if user_query:
                    cn_countries = [
                        "美国", "日本", "韩国", "朝鲜", "俄罗斯", "英国", "法国", "德国",
                        "印度", "乌克兰", "以色列", "伊朗", "澳大利亚", "加拿大",
                        "欧盟", "东盟", "越南", "泰国", "菲律宾", "土耳其",
                        "中国", "香港", "澳门",
                    ]
                    asked_countries = [c for c in cn_countries if c in user_query]

                # Get diverse recent news: 2 per source
                rows = conn.execute(
                    """SELECT id, title, summary, source, category, country, url FROM (
                           SELECT *, ROW_NUMBER() OVER (PARTITION BY source ORDER BY id DESC) AS rn
                           FROM news_articles
                       ) WHERE rn <= 2
                       ORDER BY id DESC LIMIT 18""",
                ).fetchall()

                # If user asked about specific country, also fetch matching articles
                extra_rows = []
                if asked_countries:
                    for c in asked_countries:
                        extra = conn.execute(
                            """SELECT id, title, summary, source, category, country, url
                               FROM news_articles WHERE country=?
                               ORDER BY id DESC LIMIT 5""",
                            (c,),
                        ).fetchall()
                        extra_rows.extend(extra)

                # Merge: extra rows first, then diverse rows, dedup by title
                all_rows = list(extra_rows) + [dict(r) for r in rows]
            if all_rows:
                seen = set()
                lines = []
                articles = []
                for r in all_rows:
                    title = r["title"][:40]
                    if title in seen:
                        continue
                    seen.add(title)
                    articles.append({"id": r["id"], "title": r["title"]})
                    source = r["source"]
                    country = r["country"] or ""
                    url = r["url"] or ""
                    summary = r["summary"][:50] if r["summary"] else ""
                    country_tag = f"【{country}】" if country else ""
                    url_suffix = f" | 原文链接：{url}" if url else ""
                    if summary:
                        lines.append(f"· {country_tag}[{source}] {title} — {summary}{url_suffix}")
                    else:
                        lines.append(f"· {country_tag}[{source}] {title}{url_suffix}")
                if lines:
                    return {
                        "type": "news",
                        "text": ("近日要闻（来源：新华网/人民网/央视/央广/军网/政府网/环球网/参考消息/海外网）：\n"
                                 + "\n".join(lines[:12])
                                 + "\n\n[铁规矩：以上是数据库里的全部新闻。"
                                 + "只说你看到的，不要添加任何数据库里没有的人名、地名、数字、细节。"
                                 + "如果对方问的国家/话题在数据库里没有——直接说「这方面今儿数据库里还没啥消息」，别现编。]"),
                        "ok": True,
                        "articles": articles[:12],
                    }
    except Exception:
        pass

    # Layer 2: fallback — free hot-search aggregator
    try:
        r = requests.get(
            "https://tenapi.cn/v2/baiduhot",
            timeout=6,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        data = r.json()
        if data.get("code") == 200 and data.get("data"):
            items = data["data"][:8]
            headlines = "\n".join(f"· {it['name']}" for it in items)
            return {"type": "news", "text": f"今日热搜：\n{headlines}", "ok": True}
    except Exception:
        pass

    # Layer 3: alternate free API (Weibo hot list)
    try:
        r = requests.get(
            "https://api.vvhan.com/api/hotlist?type=wbHot",
            timeout=6,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        data = r.json()
        if data.get("success") and data.get("data"):
            items = data["data"][:8]
            headlines = "\n".join(f"· {it.get('title', it.get('name', ''))}" for it in items)
            return {"type": "news", "text": f"今日热点：\n{headlines}", "ok": True}
    except Exception:
        pass

    return {"type": "news", "text": "", "ok": False}
