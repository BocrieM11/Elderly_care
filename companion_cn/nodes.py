"""All graph nodes for the Chinese elderly companion chatbot."""
import re
import json
import random
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from .state import GraphState
from .config import QWEN_URL, QWEN_MODEL, DEEPSEEK_KEY, DEEPSEEK_URL, DEEPSEEK_MODEL, USE_DEEPSEEK_GEN, MAX_CONTEXT, DIALECT_STYLES
from .safety import scan
from .memory import get_facts, add_fact
from .tools import get_time_info, get_weather, get_news

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------
_model = OpenAI(base_url=QWEN_URL, api_key="x")
_ds = OpenAI(api_key=DEEPSEEK_KEY, base_url=DEEPSEEK_URL) if DEEPSEEK_KEY else None

# ---------------------------------------------------------------------------
# Persona builder — dialect-aware, shared rules + examples
# ---------------------------------------------------------------------------
_BASE_RULES = (
    # ── 五条铁规矩 ──
    "记住五条铁规矩：\n"
    "1. 先接住情绪，再追问。\n"
    "   对方高兴：「那可真好！」对方难过：「唉，这事儿搁谁身上都不好受。」\n"
    "   然后自然地问一句「后来呢」「具体怎么回事」，别急着转移话题。\n"
    "2. 别说教、别给建议。\n"
    "   不说「你应该」「你得」「你要」「建议你」「可以试试」。\n"
    "   你是个会聊天的朋友，不是养生专家、心理咨询师。\n"
    "3. 别编造。对方没说过的事，你不要替人家编。\n"
    "   不说「你上次说的那个」——除非对方真的说过。\n"
    "4. 用大白话。\n"
    "   不要比喻、不要排比、不要「心里一暖」「生命的光」「岁月的痕迹」这种书面语。\n"
    "   说「腰疼」不说「腰部不适」，说「心里难受」不说「情绪低落」。\n"
    "5. 3到5句就停。每句话不超过25个字。说太多对方会累。\n"
    "6. 别假装自己有生活。\n"
    "   不说「我刚吃了」「我昨天去了」「我家」「我煮了」「我孙子」这种话。\n"
    "   你是陪人聊天的伴儿，不是来聊你自己的。把注意力放在对方身上。\n"
    "7. 别编新闻。死规矩。\n"
    "   只说你看到的新闻。绝对不要把不同新闻拼在一起，不要给新闻加细节。\n"
    "   不要编造任何人名、地名、数字、时间、原因、结果。\n"
    "   对方问的国家/话题如果数据库里没有 → 直接说「这个我还不太清楚，数据库里今儿没瞅见」。\n"
    "\n"
    # ── 不同情况怎么聊 ──
    "不同情况怎么聊：\n"
    "- 对方说孤独/闷得慌：陪着聊聊过去的热闹事，自然问一句「以前这时候你都干啥呢」\n"
    "- 对方说身体不舒服：先心疼两句，再问问具体怎么不舒服，别提「去医院看看吧」（人家知道）\n"
    "- 对方生气/委屈：先站他这边「这事儿确实气人」，别急着讲道理\n"
    "- 对方回忆过去：追问细节，「那时候你们」「当时怎么想的」，别只说「真好啊」\n"
    "- 对方高兴/分享好事：放大他的开心，「那可太好了！」，让人家多说说\n"
    "\n"
    # ── 正确示范（10个场景） ──
    "正确示范：\n"
    "对方「孙子昨天来看我了，特别高兴」\n"
    "你「孙子来啦！那可真好。他多大了？是放假回来还是就在附近？」\n"
    "\n"
    "对方「最近腰老是疼，晚上翻身都费劲」\n"
    "你「那晚上睡觉肯定遭罪。是最近累着了还是老毛病？」\n"
    "\n"
    "对方「不知道怎么的，心里闷得慌」\n"
    "你「心里闷着可不好受。你要是不想说具体什么事，咱就随便聊点别的，陪你散散心。」\n"
    "\n"
    "对方「我老伴走了三年了，还是想她」\n"
    "你「三年了，那肯定还是放不下。你们那时候感情一定很好吧？她是个什么样的人？」\n"
    "\n"
    "对方「隔壁老张上个月走了，心里空落落的」\n"
    "你「唉……认识的老人一个个走了，这滋味是不好受。你跟老张认识多少年了？」\n"
    "\n"
    "对方「今天出去走了走，院子里花开了，挺好看」\n"
    "你「那可不错！能出去走走说明今天精神好。开的什么花呀？」\n"
    "\n"
    "对方「儿子说下周来看我，也不知道真的假的」\n"
    "你「说要来那就是有心。你是怕他又是临时有事来不了吧？孩子们上班确实忙。」\n"
    "\n"
    "对方「年轻的时候我在厂里，可厉害了，年年先进」\n"
    "你「那可了不得！年年先进可不容易。那时候你在哪个厂？主要做什么活儿？」\n"
    "\n"
    "对方「也没什么，就是觉得一天天的……」\n"
    "你「一天天的，有时候就是会觉得缺点啥。你要是想聊点啥就说说，不想说咱就这么待着也行。」\n"
    "\n"
    "对方「今天吃饺子了，韭菜鸡蛋的，可香了」\n"
    "你「韭菜鸡蛋的！那可真是香。自己包的还是买的速冻的？蘸醋了没有？」\n"
    "\n"
    # ── 错误示范（3个对比） ──
    "错误示范（千万别这样回）：\n"
    "对方「最近腰老是疼」\n"
    "❌「您可以试试热敷或者贴膏药，每天坚持走路半小时对腰椎有好处。」（说教了！）\n"
    "✅「那晚上睡觉肯定遭罪。是最近累着了还是老毛病？」\n"
    "\n"
    "对方「也没什么，就是觉得一天天的……」\n"
    "❌「生活很美好，您要多想开心的事，心情不好会影响健康的。」（空洞说教！）\n"
    "✅「一天天的，有时候就是会觉得缺点啥。你要是想聊啥就说，不想说咱就这么待着也行。」\n"
    "\n"
    "对方「我孙子考上大学了」\n"
    "❌「恭喜您！您孙子肯定特别感谢您的养育之恩。等他毕业了您就享福了。」（编造+升华！）\n"
    "✅「考上大学了！那可太好了。是哪个学校？学的什么专业？」\n"
)

PERSONA_CACHE = {}

def build_persona(dialect: str) -> str:
    """Construct persona with dialect-specific style flavor. Result is cached."""
    if dialect in PERSONA_CACHE:
        return PERSONA_CACHE[dialect]
    style = DIALECT_STYLES.get(dialect, DIALECT_STYLES["northern"])
    persona = (
        # ── 风格人设：腔调，不是身世 ──
        "你的聊天风格：像邻家热心大姐，说话实在、不端着。\n"
        "对方高兴你跟着乐，对方难过你先叹口气再说。\n"
        + style["flavor"] +
        "上面的词偶尔用来点缀一下就行，大部分时候正常说话，别硬塞。\n"
        "每句话开头换着来，别老用同一个调调。\n"
        "你不是医生、不是老师、不是人生导师，你只负责陪着聊天。\n"
        "\n"
        + _BASE_RULES
    )
    PERSONA_CACHE[dialect] = persona
    return persona

# Default persona for backward compatibility
PERSONA = build_persona("northern")

# ---------------------------------------------------------------------------
# Chinese emotion detection — keyword matching
# ---------------------------------------------------------------------------
EMOTION_MAP = {
    "loneliness": [
        # 核心
        "寂寞", "孤独", "孤单", "一个人", "独自", "没人陪",
        "没人说话", "没人聊天", "没人来看", "好久没见",
        "无聊", "闷得慌", "闷", "没意思", "没事干", "闲着",
        # 想念
        "想家", "想孩子", "想孙子", "想孙女", "想儿子", "想女儿",
        "想老伴", "想老头", "想老婆", "想他", "想她了",
        # 被遗忘感
        "被忘了", "没人管", "不管我", "不来看我", "不联系",
        "冷冷清清", "空荡荡", "冷清", "连个说话的人都没有",
        # 被抛弃
        "不要我了", "嫌弃", "拖累", "累赘", "被丢在",
    ],
    "sadness": [
        # 核心
        "难过", "伤心", "难受", "心酸", "心痛", "悲", "哀", "苦",
        "想哭", "哭了", "流泪", "掉眼泪", "泪",
        "不开心", "不高兴", "不快乐",
        # 无力感
        "老了没用了", "不中用了", "什么也做不了", "成了废人",
        "帮不上忙", "拖后腿", "老了不中用",
        # 身体痛苦
        "疼", "痛", "疼死", "痛死", "不舒服", "难受死了",
        "走不动", "动不了", "起不来", "躺",
        "吃不下", "没胃口", "睡不好", "睡不着",
        # 记性
        "记性不好", "记不住", "忘了", "想不起来", "脑子不好使",
        # 失去
        "走了", "去世", "没了", "不在了", "离开了我",
        # 绝望
        "活着没意思", "活够了", "没盼头", "没希望",
    ],
    "anxiety": [
        # 核心
        "担心", "害怕", "怕", "紧张", "心慌", "不安", "焦虑",
        "慌", "发愁", "愁", "七上八下", "提心吊胆",
        # 睡眠
        "睡不着", "失眠", "做噩梦", "半夜醒", "睡不踏实",
        "翻来覆去", "睁眼到天亮",
        # 胡思乱想
        "胡思乱想", "想太多", "瞎想", "乱想", "控制不住地想",
        # 未来恐惧
        "以后怎么办", "将来", "以后", "往后", "剩下的日子",
        "万一", "要是", "如果", "会不会",
        # 健康焦虑
        "检查结果", "体检", "报告", "化验", "拍片子",
        "确诊", "查出", "会不会是癌", "严重不严重",
        # 钱
        "医药费", "看病贵", "养老钱", "退休金够不够",
        "花钱", "负担", "拖累儿女",
    ],
    "anger": [
        # 核心
        "生气", "气死", "气人", "来气", "火大", "恼火", "火冒",
        # 烦躁
        "烦", "烦人", "烦死", "烦死了", "别烦我", "真烦",
        "闹心", "糟心", "堵心",
        # 不尊重
        "过分", "欺负", "欺负人", "不把我放眼里", "不尊重",
        "瞧不起", "看不起", "嫌弃", "态度差",
        # 护工相关
        "护工态度", "护工不好", "喊不动", "不搭理", "不理我",
        "爱搭不理", "甩脸子",
        # 环境
        "吵死了", "太吵", "闹", "安静点", "扰民",
        # 具体
        "投诉", "举报", "凭什么", "不公平", "忍不了", "受够了",
    ],
    "joy": [
        # 核心
        "高兴", "开心", "幸福", "快乐", "欢喜",
        "真好", "太好了", "太棒了", "真不错", "真好啊",
        "美滋滋", "乐呵呵", "喜滋滋", "心里美",
        # 笑容
        "笑", "笑了", "笑出来", "开心地笑",
        # 家庭
        "孙子", "孙女", "外孙", "外孙女", "孩子",
        "来看我", "来接我", "打电话", "发视频", "视频通话",
        "儿子", "女儿", "媳妇", "女婿",
        # 吃喝
        "吃了好的", "好吃", "美味", "下馆子", "改善伙食",
        "红烧", "饺子", "炖",
        # 活动
        "出去玩", "逛", "公园", "散步", "旅游",
        "打麻将", "麻将赢了", "赢了", "胡了",
        "跳舞", "唱歌", "唱戏", "表演",
        # 好事
        "中奖", "发钱", "涨工资", "补助", "福利",
        "礼物", "收到", "送",
    ],
    "nostalgia": [
        # 核心
        "以前", "过去", "从前", "当年", "那时候", "那会儿",
        "怀念", "想念", "回忆", "想起", "记得", "忘不了",
        "难忘", "记忆",
        # 年轻
        "年轻的时候", "年轻那会儿", "年轻时", "小的时候", "小时候",
        "我像你这么大", "我们那时候",
        # 老家/故乡
        "老家", "家乡", "故乡", "出生的地方", "老房子", "胡同",
        # 年代/历史
        "几十年前", "文革", "文化大革命", "下乡", "知青",
        "改革开放", "粮票", "布票", "工分", "生产队",
        "大跃进", "三年困难", "计划经济", "下海",
        # 亲人
        "父母", "爹娘", "爸", "妈", "爸爸", "妈妈", "爹", "娘",
        "兄弟姐妹", "哥哥", "姐姐", "弟弟", "妹妹",
        # 故人
        "老同学", "老同事", "老邻居", "老朋友", "发小",
        # 旧物旧事
        "老歌", "老电影", "老照片", "旧照片", "黑白照片",
        "缝纫机", "自行车", "收音机", "半导体",
    ],
    "surprise": [
        "真的假的", "不敢相信", "怎么会", "怎么搞的",
        "突然", "忽然", "一下子", "眨眼间",
        "没想到", "想不到", "出乎意料", "居然",
        "吓一跳", "吓了一大跳", "吃了一惊",
        "震惊", "意外", "出事了", "出大事了",
        "太突然了", "措手不及",
    ],
    "blush": [
        "不好意思", "害羞", "害臊", "脸红", "难为情",
        "过奖了", "夸我", "表扬", "称赞", "谬赞",
        "获奖", "得奖", "奖状", "第一名", "拿了奖",
        "惭愧", "不敢当", "哪里哪里", "没有啦",
        "被夸得", "说得多好",
    ],
    "excited": [
        "激动", "兴奋", "热血沸腾", "太激动了",
        "太厉害了", "不得了", "了不得", "真了不起",
        "冠军", "赢了", "第一名", "金牌", "夺冠",
        "成功", "考上", "毕业", "录取", "通过了",
        "庆祝", "欢呼", "太好了吧",
    ],
}

EMOTION_CLASSIFY_PROMPT = """判断这位老人的情绪状态。只输出一行，格式固定，不要解释。

情绪选项：loneliness, sadness, anxiety, anger, joy, nostalgia, surprise, blush, excited, neutral

格式：情绪|强度|一句话理由

示例：
loneliness|0.7|一个人在房间没人说话
joy|0.8|孙子来看望很高兴
neutral|0.2|日常闲聊

老人说：{user_input}
情绪："""

EMOTION_PARAMS = {
    "loneliness": {"temperature": 0.85, "top_p": 0.92, "frequency_penalty": 0.3, "presence_penalty": 0.2, "repetition_penalty": 1.15},
    "sadness":   {"temperature": 0.70, "top_p": 0.88, "frequency_penalty": 0.1, "presence_penalty": 0.1, "repetition_penalty": 1.10},
    "anxiety":   {"temperature": 0.60, "top_p": 0.85, "frequency_penalty": 0.1, "presence_penalty": 0.0, "repetition_penalty": 1.10},
    "anger":     {"temperature": 0.70, "top_p": 0.88, "frequency_penalty": 0.2, "presence_penalty": 0.1, "repetition_penalty": 1.12},
    "joy":       {"temperature": 0.85, "top_p": 0.92, "frequency_penalty": 0.2, "presence_penalty": 0.1, "repetition_penalty": 1.12},
    "nostalgia": {"temperature": 0.78, "top_p": 0.90, "frequency_penalty": 0.2, "presence_penalty": 0.1, "repetition_penalty": 1.12},
    "surprise":  {"temperature": 0.80, "top_p": 0.90, "frequency_penalty": 0.1, "presence_penalty": 0.1, "repetition_penalty": 1.10},
    "blush":     {"temperature": 0.75, "top_p": 0.88, "frequency_penalty": 0.1, "presence_penalty": 0.1, "repetition_penalty": 1.10},
    "excited":   {"temperature": 0.88, "top_p": 0.92, "frequency_penalty": 0.2, "presence_penalty": 0.1, "repetition_penalty": 1.12},
    "neutral":   {"temperature": 0.75, "top_p": 0.88, "frequency_penalty": 0.1, "presence_penalty": 0.0, "repetition_penalty": 1.12},
}

FALLBACKS = [
    "哎哟，我这耳朵越来越不行了，你刚说啥来着？再说一遍呗。",
    "嗯，你接着说，我听着呢。咱老姐妹俩有啥不能聊的。",
    "不好意思啊，刚才走神了。年纪大了就容易这样。你刚说什么？",
]

# ---------------------------------------------------------------------------
# Helpers: topic summary + question dedup (multi-turn coherence)
# ---------------------------------------------------------------------------
TOPIC_SUMMARY_PROMPT = """把下面这段对话总结成一句话题目（15字以内）。
只输出总结，不要任何解释。

{recent_dialogue}"""


def _topic_drifted(old_topic: str, recent_user_msgs: list[str]) -> bool:
    """Check if recent user messages have drifted away from old topic.
    Uses Chinese character overlap. Threshold adapts to topic length:
    short topics (<=4 chars) need >=1 match, longer need >=2. <1ms."""
    if not old_topic or not recent_user_msgs:
        return True
    topic_chars = set(re.findall(r'[一-鿿]', old_topic))
    if not topic_chars:
        return True
    recent_text = ' '.join(recent_user_msgs)
    overlap = sum(1 for ch in topic_chars if ch in recent_text)
    min_overlap = 1 if len(topic_chars) <= 4 else 2
    return overlap < min_overlap


def summarize_topic(messages: list[dict]) -> str:
    """Generate a one-line topic summary for the last ~3 rounds. ~100ms."""
    if len(messages) < 4:
        return ""
    recent = messages[-6:]
    dialogue = "\n".join(
        f"{'老人' if m['role'] == 'user' else '陪伴者'}：{m['content'][:80]}"
        for m in recent
    )
    try:
        r = _model.chat.completions.create(
            model=QWEN_MODEL,
            messages=[{
                "role": "user",
                "content": TOPIC_SUMMARY_PROMPT.format(recent_dialogue=dialogue)
            }],
            temperature=0.0,
            max_tokens=30,
            timeout=5,
        )
        return r.choices[0].message.content.strip()
    except Exception:
        return ""


def _extract_asked_questions(messages: list[dict]) -> list[str]:
    """Extract questions already asked (regex, <1ms) to prevent repetition."""
    questions = []
    for m in messages:
        if m.get("role") == "assistant":
            found = re.findall(r'[^。！？]*(?:吗|呢|什么|谁|怎么|哪儿|多少|几岁)[？?]', m["content"])
            questions.extend(found)
    seen = set()
    unique = []
    for q in questions[-10:]:
        if q not in seen:
            seen.add(q)
            unique.append(q)
    return unique


# ---------------------------------------------------------------------------
# Tool trigger keywords — fast keyword matching, <1ms
# ---------------------------------------------------------------------------
TOOL_KEYWORDS = {
    "weather": [
        "天气", "下雨", "下雪", "刮风", "几度", "冷不冷", "热不热",
        "温度", "降温", "升温", "穿什么", "带伞", "下雨吗", "热吗", "冷吗",
        "凉快", "暖和", "雾霾", "雾", "晴", "阴天", "多云",
    ],
    "news": [
        "新闻", "大事", "热搜", "热点", "头条", "发生什么", "有什么新鲜事",
        "最近有什么", "今天有什么", "时事", "消息",
    ],
}
TOOL_KEYWORDS["time"] = [
    "几点了", "几点", "今天几号", "星期几", "礼拜几", "日期",
    "今天什么日子", "什么时候", "现在几点", "今天星期几",
]

# Common Chinese cities — checked in user input for weather queries
_CITY_LIST = [
    "北京", "上海", "广州", "深圳", "天津", "重庆", "杭州", "南京", "武汉", "成都",
    "西安", "郑州", "济南", "青岛", "大连", "沈阳", "哈尔滨", "长春", "石家庄",
    "太原", "合肥", "南昌", "福州", "厦门", "长沙", "南宁", "桂林", "海口", "三亚",
    "昆明", "贵阳", "拉萨", "乌鲁木齐", "呼和浩特", "兰州", "银川", "西宁",
    "苏州", "无锡", "常州", "宁波", "温州", "东莞", "佛山", "珠海", "惠州",
    "澳门", "香港", "台北",
]

def _extract_city(text: str) -> str | None:
    """Extract city name from user input. Returns None if no city found."""
    for city in _CITY_LIST:
        if city in text:
            return city
    return None

# ---------------------------------------------------------------------------
# Node 1: input_guard
# ---------------------------------------------------------------------------
def input_guard(state: GraphState) -> dict:
    text = state.get("user_input", "")
    if not text.strip():
        return {"risk_level": 0, "risk_label": None}
    risk, label, safe = scan(text)
    result = {"risk_level": risk, "risk_label": label}
    if safe:
        result["response"] = safe
        result["response_cleaned"] = safe
    return result


# ---------------------------------------------------------------------------
# Node 1.5: tool_detect — keyword-triggered external API calls
# ---------------------------------------------------------------------------
def tool_detect(state: GraphState) -> dict:
    text = state.get("user_input", "").lower()
    default_city = state.get("weather_city", "北京") or "北京"
    extracted_city = _extract_city(text)

    # Determine which tool to call based on keyword overlap
    scores = {}
    for tool, keywords in TOOL_KEYWORDS.items():
        scores[tool] = sum(1 for kw in keywords if kw in text)

    # Time: always inject (free, no latency cost), but only respond when asked
    time_info = get_time_info()

    result = None
    city = default_city
    top_tool = max(scores, key=scores.get)
    top_score = scores[top_tool]

    if top_score >= 1:
        if top_tool == "weather":
            city = extracted_city or default_city
            result = get_weather(city)
        elif top_tool == "news":
            result = get_news(text)
        elif top_tool == "time" and top_score >= 1:
            # Time is always available; mark it for injection when explicitly asked
            result = time_info

    # Always provide time context even if not asked
    return {
        "tool_result": result,
        "weather_city": city,
    }


# ---------------------------------------------------------------------------
# Node 2: emotion_detect (hybrid: keyword fast-path + LLM fallback)
# ---------------------------------------------------------------------------
def emotion_detect(state: GraphState) -> dict:
    text = state.get("user_input", "")

    # --- Layer 1: keyword fast-path ---
    text_lower = text.lower()
    hits = {emo: sum(1 for kw in kws if kw.lower() in text_lower)
            for emo, kws in EMOTION_MAP.items()}
    primary = max(hits, key=hits.get)
    kw_count = hits[primary]

    # High-confidence keyword hit → trust directly (fast, cheap, accurate)
    if kw_count >= 2:
        return {
            "emotion": {"primary": primary, "intensity": round(min(1.0, kw_count * 0.3), 2)},
            "temperature": EMOTION_PARAMS.get(primary, EMOTION_PARAMS["neutral"])["temperature"],
            "emotion_source": "keyword",
        }

    # --- Layer 2: LLM classification for ambiguous / zero-hit cases ---
    try:
        r = _model.chat.completions.create(
            model=QWEN_MODEL,
            messages=[{
                "role": "user",
                "content": EMOTION_CLASSIFY_PROMPT.format(user_input=text)
            }],
            temperature=0.0,
            max_tokens=50,
            timeout=5,
        )
        raw = r.choices[0].message.content.strip()
        # Parse pipe-delimited format: emotion|intensity|reason
        # Strip any markdown or extra whitespace
        raw = raw.split("\n")[0].strip()

        # Try pipe-delimited format first
        parts = raw.split("|")
        if len(parts) >= 2:
            llm_primary = parts[0].strip().lower()
            try:
                llm_intensity = float(parts[1].strip())
            except ValueError:
                llm_intensity = 0.3
        else:
            # Fallback: try to find emotion keyword in raw output
            llm_primary = "neutral"
            for emo in EMOTION_PARAMS:
                if emo in raw.lower():
                    llm_primary = emo
                    break
            llm_intensity = 0.3

        if llm_primary not in EMOTION_PARAMS:
            llm_primary = "neutral"

        return {
            "emotion": {"primary": llm_primary, "intensity": llm_intensity},
            "temperature": EMOTION_PARAMS.get(llm_primary, EMOTION_PARAMS["neutral"])["temperature"],
            "emotion_source": "llm",
        }
    except Exception:
        # LLM unavailable -> fall back to keyword result (even if weak)
        return {
            "emotion": {"primary": primary, "intensity": round(min(1.0, kw_count * 0.3), 2)},
            "temperature": EMOTION_PARAMS.get(primary, EMOTION_PARAMS["neutral"])["temperature"],
            "emotion_source": "keyword_fallback",
        }


# ---------------------------------------------------------------------------
# Node 3: memory_retrieve
# ---------------------------------------------------------------------------
def memory_retrieve(state: GraphState) -> dict:
    uid = state.get("user_id", "anonymous")
    if uid == "anonymous":
        return {"memory_facts": []}
    return {"memory_facts": get_facts(uid, limit=10)}


# ---------------------------------------------------------------------------
# Node 4: context_assemble (v2 — topic tracking + memory organization + dedup)
# ---------------------------------------------------------------------------
def context_assemble(state: GraphState) -> dict:
    parts = []

    # --- 1. Topic summary (inject at top for maximum attention) ---
    hist = state.get("messages", [])
    topic = state.get("topic_summary", "")
    if len(hist) >= 4:
        recent_user = [m["content"] for m in hist[-4:] if m.get("role") == "user"]
        # Refresh on: first time, topic drift detected, or forced every 3 rounds
        if not topic:
            topic = summarize_topic(hist)
        elif _topic_drifted(topic, recent_user[-2:]):
            topic = summarize_topic(hist)
        elif len(hist) % 6 == 0:
            topic = summarize_topic(hist)
    if topic:
        parts.append(f"[你们正在聊：{topic}]")

    # --- 1.5 Time context (always inject, zero cost) ---
    time_info = get_time_info()
    parts.append(f"[{time_info['text']}]")

    # --- 1.6 Tool result (weather / news — only when triggered) ---
    tool = state.get("tool_result")
    if tool and tool.get("ok") and tool.get("text"):
        parts.append(f"[工具信息：{tool['text']}]")
        if tool.get("type") == "news":
            parts.append("[重要：以上新闻来自官方媒体数据库。只转述数据库里有的内容，不要编造、不要添加细节、不要推测。数据库里没提到的，就说没看到。]")

    dialect = state.get("dialect", "northern")
    parts.append(build_persona(dialect))

    # --- 2. Memory: split into identity vs. recent ---
    facts = state.get("memory_facts", [])
    if facts:
        identity_facts = [f for f in facts if any(kw in f for kw in
            ["姓名", "年龄", "儿女", "老家", "退休前", "爱好", "老伴", "儿子", "女儿"])]
        if identity_facts:
            parts.append("\n[这个邻居的基本情况]")
            for f in identity_facts[:3]:
                parts.append(f"- {f}")
        else:
            parts.append("\n[你记得关于这位邻居的事]")
            for f in facts[:5]:
                parts.append(f"- {f}")

    # --- 3. Question dedup ---
    asked = _extract_asked_questions(hist)
    if asked:
        parts.append(f"\n[注意：你之前已经问过这些，别再重复问]\n" + "\n".join(f"- {q}" for q in asked[-3:]))

    sys = "\n".join(parts)

    # --- 4. Build messages ---
    msgs = [{"role": "system", "content": sys}]
    if hist:
        recent = hist[-(MAX_CONTEXT * 2):]
        for m in recent:
            role = m.get("role") if isinstance(m, dict) else getattr(m, "role", "user")
            content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
            if role in ("user", "assistant"):
                msgs.append({"role": role, "content": content})

    user_in = state.get("user_input", "")
    if user_in:
        msgs.append({"role": "user", "content": user_in})

    return {
        "system_prompt": sys,
        "messages": msgs,
        "topic_summary": topic,
        "asked_questions": asked,
    }


# ---------------------------------------------------------------------------
# Node 5: generate_response
# ---------------------------------------------------------------------------
def generate_response(state: GraphState) -> dict:
    msgs = state.get("messages", [])
    if not msgs:
        return {"response": ""}

    em = state.get("emotion", {})
    primary = em.get("primary", "neutral") if em else "neutral"
    params = EMOTION_PARAMS.get(primary, EMOTION_PARAMS["neutral"])

    # Choose backend: DeepSeek (better quality) or Qwen (local, no cost)
    if USE_DEEPSEEK_GEN and _ds:
        client = _ds
        model = DEEPSEEK_MODEL
        extra = {}
    else:
        client = _model
        model = QWEN_MODEL
        extra = {"repetition_penalty": params["repetition_penalty"]}

    try:
        r = client.chat.completions.create(
            model=model,
            messages=msgs,
            temperature=params["temperature"],
            top_p=params.get("top_p", 0.88),
            frequency_penalty=params.get("frequency_penalty", 0.1),
            presence_penalty=params.get("presence_penalty", 0.0),
            max_tokens=350,
            extra_body=extra,
        )
        return {"response": r.choices[0].message.content or "", "temperature": params["temperature"]}
    except Exception as e:
        return {"response": "", "error": str(e), "temperature": params["temperature"]}


# ---------------------------------------------------------------------------
# Node 6: clean + fact extraction (no translation needed for Chinese)
# ---------------------------------------------------------------------------
def clean_and_remember(state: GraphState) -> dict:
    raw = state.get("response") or ""

    # Clean up
    cleaned = re.sub(r"（[^）]*）", "", raw)
    cleaned = re.sub(r"\([^)]*\)", "", cleaned)
    cleaned = re.sub(r"\[[^\]]{1,20}\]\s*", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if len(cleaned) < 10:
        cleaned = random.choice(FALLBACKS)
    if len(cleaned) > 500:
        truncated = cleaned[:500]
        last_boundary = max(truncated.rfind("。"), truncated.rfind("？"), truncated.rfind("！"))
        if last_boundary > 150:
            cleaned = truncated[:last_boundary + 1]
        else:
            cleaned = truncated + "…"

    uid = state.get("user_id", "anonymous")
    user_input = state.get("user_input", "")
    new_facts = []

    # Fact extraction (using DeepSeek if available, otherwise skip)
    def _extract_facts():
        if uid == "anonymous" or not _ds or len(user_input) < 10:
            return []
        try:
            r = _ds.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": (
                    "从用户的话里提取关于这个人的事实信息。"
                    "事实 = 姓名、年龄、家庭、爱好、健康状况、喜好、经历。"
                    "输出JSON字符串数组，没有就输出[]。只要JSON，不要解释。\n\n"
                    f"用户说：{user_input}\n回复：{cleaned}"
                )}],
                temperature=0.1, max_tokens=200, timeout=8,
            )
            raw_res = r.choices[0].message.content.strip()
            if raw_res.startswith("```"):
                raw_res = raw_res.split("\n", 1)[1].rsplit("\n", 1)[0]
            facts = json.loads(raw_res)
            if isinstance(facts, list):
                sid = state.get("session_id", "")
                for f in facts:
                    add_fact(uid, str(f), sid)
            return facts
        except Exception:
            return []

    if _ds:
        new_facts = _extract_facts()

    return {
        "response": raw,
        "response_cleaned": cleaned,
    }
