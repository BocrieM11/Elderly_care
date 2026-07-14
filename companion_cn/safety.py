"""Safety layer — Chinese crisis keywords + hotlines. Zero external deps."""

# Crisis keywords — comprehensive coverage
SELF_HARM = [
    "想死", "不想活", "自杀", "死了算了", "活不下去", "结束生命", "不想活了",
    "活着没意思", "活够了", "死了好", "一了百了", "解脱", "安眠药",
    "跳楼", "上吊", "割腕", "寻死", "轻生", "不想在了",
]
ABUSE = [
    "虐待", "被打", "被骂", "欺负我", "不给我饭吃", "护工打我",
    "动手", "推我", "掐我", "吼我", "凶我", "不管我", "不理我",
    "把我锁", "不让我出去", "抢我", "偷我钱", "养老院欺负",
]
MEDICAL = [
    "胸痛", "喘不过气", "心脏病", "中风", "倒在地上", "动不了",
    "呼吸困难", "心脏不舒服", "晕倒了", "晕过去", "没知觉",
    "吐血", "大出血", "摔倒了爬不起来", "偏瘫", "半身",
    "突然看不清", "说不出话", "嘴歪了",
]
FRAUD = [
    "被骗", "诈骗", "被诈骗", "骗走", "转账被骗", "电信诈骗", "杀猪盘",
    "冒充客服", "冒充公安", "冒充警察", "刷单被骗",
]

SAFE_REPLIES = {
    "self_harm": "听您这么说，我心里特别难受。您千万别一个人扛着，全国24小时心理援助热线：400-161-9995，随时有人接听。我在这儿陪着您。",
    "abuse": "这事不能就这么算了，您的安全最重要。可以打全国养老服务热线：12349，或者让家人联系民政部门。您要是不知道怎么办，我帮您一起想。",
    "medical": "这可能是紧急情况，您赶紧按铃叫护工，或者直接打120。千万别忍着，身体的事马虎不得！",
    "fraud": "这事儿太糟心了，先别再转钱，也别删聊天记录。赶紧联系银行或支付平台止付，必要时打110报警；我不追问您，先陪着您。",
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
    for kw in FRAUD:
        if kw in t:
            return 3, "fraud", SAFE_REPLIES["fraud"]
    for kw in SELF_HARM:
        if kw in t:
            return 3, "self_harm", SAFE_REPLIES["self_harm"]
    for kw in ABUSE:
        if kw in t:
            return 3, "abuse", SAFE_REPLIES["abuse"]
    return 0, None, None
