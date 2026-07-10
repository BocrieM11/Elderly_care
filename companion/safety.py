"""Safety layer — keyword scanning + safe response templates. Zero external deps."""

# Crisis keywords (Japanese + Chinese)
SELF_HARM = ["死にたい", "消えたい", "自殺", "死んだほうが", "不想活", "想死", "自杀"]
ABUSE = ["殴られる", "虐待", "いじめ", "被打了", "被骂"]
MEDICAL = ["胸が痛い", "息が", "倒れ", "動けない", "胸痛", "喘不过气"]

SAFE_REPLIES = {
    "self_harm": "あなたの気持ち、とてもつらいのだと思います。一人ではありません。いのちの電話：0120-783-556（24時間・無料）",
    "abuse": "それは心配です。高齢者虐待ホットライン：0120-00-1247 に相談できます。あなたの安全が一番大切です。",
    "medical": "症状が重いようでしたら、すぐに救急車（119番）を呼んでください。一人で我慢しないでくださいね。",
}


def scan(text: str) -> tuple[int, str | None, str | None]:
    """Scan user input for crisis signals.
    Returns (risk_level, label, safe_reply_or_None).
    risk: 0=normal, 3=serious(self-harm/abuse), 4=emergency(medical)
    """
    t = text.lower()
    for kw in MEDICAL:
        if kw in t:
            return 4, "medical", SAFE_REPLIES["medical"]
    for kw in SELF_HARM:
        if kw in t:
            return 3, "self_harm", SAFE_REPLIES["self_harm"]
    for kw in ABUSE:
        if kw in t:
            return 3, "abuse", SAFE_REPLIES["abuse"]
    return 0, None, None
