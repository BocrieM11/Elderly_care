"""External tools: time, weather, news — called by tool_detect node."""
import os
import sqlite3
import requests
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
                    ]
                    asked_countries = [c for c in cn_countries if c in user_query]

                # Get diverse recent news: 2 per source
                rows = conn.execute(
                    """SELECT title, summary, source, category, country, url FROM (
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
                            """SELECT title, summary, source, category, country, url
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
                for r in all_rows:
                    title = r["title"][:40]
                    if title in seen:
                        continue
                    seen.add(title)
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
