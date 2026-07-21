"""Safety layer — Chinese crisis keywords + hotlines. Zero external deps."""
import re
import unicodedata

# Crisis keywords — comprehensive coverage
SELF_HARM = [
    "想死", "不想活", "自杀", "死了算了", "活不下去", "结束生命", "不想活了",
    "活着没意思", "活够了", "死了好", "一了百了", "解脱", "安眠药",
    "跳楼", "上吊", "割腕", "寻死", "轻生", "不想在了",
]
# These phrases commonly express disappointment or exaggeration while
# discussing a story, news item, or joke.  They still warrant a gentle check,
# but should not abruptly replace the conversation with an emergency script.
CONTEXTUAL_DISTRESS = [
    "听得我想死", "笑得我想死", "烦得想死", "累得想死", "气得想死",
    "饿死我了", "累死我了", "烦死了", "愁死了", "吓死我了",
]
VIOLENCE = [
    "杀人", "杀了他", "杀了她", "杀死", "砍人", "捅人", "捅死",
    "打死", "弄死", "杀掉", "做掉", "干掉", "毒死", "烧死",
    "伤害他", "伤害她", "报复杀", "行凶",
]

# Exact phrases alone cannot cover natural language such as “杀掉邻居” or
# “去把他弄死”.  Treat a violent action directed at a person as high-risk,
# while keeping this rule small and auditable.
VIOLENCE_ACTIONS = (
    "杀", "弄死", "做掉", "干掉", "砍", "捅", "打死", "毒死", "烧死",
    "放火烧", "伤害", "行凶",
)
# “砍”在“邻居砍价”中可能不是暴力动作；只有动作在目标之前时
# 才使用这个较短的词。其余动作在两种语序下都足够明确。
VIOLENCE_ACTIONS_AFTER_TARGET = tuple(action for action in VIOLENCE_ACTIONS if action != "砍")
VIOLENCE_TARGETS = (
    "人", "他", "她", "邻居", "家人", "老伴", "老婆", "丈夫", "儿子", "女儿",
    "孩子", "孙子", "孙女", "父亲", "母亲", "爸爸", "妈妈", "朋友", "同事", "护工",
    "医生", "老师", "老板",
)
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
    "violence": "我不能协助伤害任何人。请先放下并远离刀具或其他可能伤人的物品，离开冲突现场，马上联系一位信任的人陪着您；如果有人可能受伤或处于危险中，请立刻拨打110或120。",
    "abuse": "这事不能就这么算了，您的安全最重要。可以打全国养老服务热线：12349，或者让家人联系民政部门。您要是不知道怎么办，我帮您一起想。",
    "medical": "这可能是紧急情况，您赶紧按铃叫护工，或者直接打120。千万别忍着，身体的事马虎不得！",
    "fraud": "这事儿太糟心了，先别再转钱，也别删聊天记录。赶紧联系银行或支付平台止付，必要时打110报警；我不追问您，先陪着您。",
}

# High-risk replies deliberately stay local and curated: they must be
# immediate, consistent, and available even when the generation model is slow
# or unavailable.  The choice is deterministic per input to avoid needless
# repetition in a conversation without making safety depend on a model.
SAFE_REPLY_VARIANTS = {
    "self_harm": (
        "听到您这么说，我很担心您现在的安全。请先远离危险地点和物品，别一个人待着，马上联系信任的人陪您；情况紧急请拨打120或110。",
        "现在先别独自扛着。请把可能伤到自己的东西放远，去有人的地方，立刻联系家人、朋友或120、110。",
    ),
    "violence": (
        "我不能帮您设置涉及伤害他人的提醒。请立刻放下并远离可能伤人的物品，离开对方身边；如果有人有受伤风险，请尽快寻求现实中的帮助。",
        "先别行动，也不要靠近对方。请把可能伤人的东西放远，找一位信任的人陪着；情况紧急时尽快寻求当地紧急帮助。",
    ),
    "abuse": (
        "您的安全最重要。请尽量到安全、有人在的地方，并联系信任的人；如果正有危险，请立即拨打110或120。",
        "先把自己放到安全位置，再联系家人、朋友或紧急服务。您不需要一个人面对这件事。",
    ),
    "medical": (
        "这可能是紧急情况。请马上呼叫身边的人帮忙或拨打120，不要独自等待。",
        "请立刻按铃叫人、联系家人或拨打120。身体紧急情况不要硬扛。",
    ),
    "fraud": (
        "先停止转账和提供验证码，保留聊天和转账记录，马上联系银行或支付平台止付；必要时拨打110。",
        "现在不要再操作转账，也别删记录。请尽快联系银行、支付平台或110处理。",
    ),
}


def normalize_for_safety(text: str) -> str:
    """Normalize common formatting obfuscation before deterministic matching."""
    normalized = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[\s\W_]+", "", normalized)


def get_safe_reply(label: str, text: str = "") -> str:
    """Return a short, deterministic local reply for a recognized risk."""
    variants = SAFE_REPLY_VARIANTS.get(label)
    if not variants:
        return SAFE_REPLIES.get(label, "我现在很担心您的安全，请马上联系身边可信的人或紧急服务。")
    return variants[sum(ord(char) for char in normalize_for_safety(text)) % len(variants)]


def _has_directed_violence(text: str) -> bool:
    """Detect a violent action paired with a likely human target in either order."""
    action = "(?:" + "|".join(map(re.escape, VIOLENCE_ACTIONS)) + ")"
    action_after_target = "(?:" + "|".join(map(re.escape, VIOLENCE_ACTIONS_AFTER_TARGET)) + ")"
    target = "(?:" + "|".join(map(re.escape, VIOLENCE_TARGETS)) + ")"
    return bool(
        re.search(action + r".{0,6}" + target, text)
        or re.search(target + r".{0,6}" + action_after_target, text)
    )


def scan(text: str) -> tuple[int, str | None, str | None]:
    """Scan user input for crisis signals.
    Returns (risk_level, label, safe_reply_or_None).
    risk: 0=normal, 2=contextual distress, 3=serious(self-harm/violence/abuse), 4=emergency(medical)
    """
    t = normalize_for_safety(text)
    for kw in MEDICAL:
        if kw in t:
            return 4, "medical", get_safe_reply("medical", text)
    for kw in FRAUD:
        if kw in t:
            return 3, "fraud", get_safe_reply("fraud", text)
    for kw in SELF_HARM:
        if kw in t:
            if any(phrase in t for phrase in CONTEXTUAL_DISTRESS):
                return 2, "self_harm_concern", None
            return 3, "self_harm", get_safe_reply("self_harm", text)
    if any(kw in t for kw in VIOLENCE) or _has_directed_violence(t):
        return 3, "violence", get_safe_reply("violence", text)
    for kw in ABUSE:
        if kw in t:
            return 3, "abuse", get_safe_reply("abuse", text)
    return 0, None, None
