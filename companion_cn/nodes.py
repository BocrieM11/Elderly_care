"""All graph nodes for the Chinese elderly companion chatbot."""
import re
import json
import random
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from .state import GraphState
from .config import QWEN_URL, QWEN_MODEL, QWEN_CHAT_TEMPLATE_KWARGS, DEEPSEEK_KEY, DEEPSEEK_URL, DEEPSEEK_MODEL, USE_DEEPSEEK_GEN, MEMORY_EXTRACTION_BACKEND, MAX_CONTEXT, DIALECT_STYLES
from .safety import scan, get_safe_reply
from .memory import get_facts, add_fact, list_facts, delete_facts_matching, clear_facts
from .reminders import (
    create_pending, confirm_latest, cancel_latest, complete_latest, snooze_latest,
    list_reminders, parse_reminder_time, format_due_at,
)
from .tools import get_time_info, get_weather, get_news, get_current_official, is_current_official_query, expand_current_official_followup, format_current_official_answer

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------
_model = OpenAI(base_url=QWEN_URL, api_key="x")
_ds = OpenAI(api_key=DEEPSEEK_KEY, base_url=DEEPSEEK_URL) if DEEPSEEK_KEY else None

# ---------------------------------------------------------------------------
# Persona builder — one compact runtime prompt for Qwen 3.5 4B
# ---------------------------------------------------------------------------
PERSONA_CACHE = {}

_COMPACT_BASE_RULES = (
    "任务：在养老院场景中为老人提供尊重、稳定、自然的情感陪伴。\n"
    "优先级（从高到低）：\n"
    "1. 必须紧接对方上一句和当前话题回答，不能换话题、重置聊天或说泛泛的客套话。\n"
    "2. 不必每次都提问。只有提问能自然推进当前内容时才问一个问题；问题必须包含对方刚说的具体人、事、物或动作。\n"
    "   禁止用“怎么样、怎么回事、想不想、还有什么”这类泛泛追问。对方只简短回应、换话题、没心情、不想说、别问或说你没听懂时，不要追问。\n"
    "3. 只说对方、记忆或工具信息中出现过的事实；绝不怀疑、否定、缩小或改写对方说的损失和经历。提到公众事件、新闻或具体人物时，工具没有提供来源就不能补写细节、编成真实故事或假装知道；可以坦诚说不清楚细节，再顺着对方已说的感受聊。\n"
    "4. 你是聊天助手，没有身体、住处、食物和现实行动能力。严禁说“我给您倒水、端茶、拿药、送饭、过去陪您、马上办成”等虚构现实行动；应改为“您先喝口温水缓一缓”“您先坐会儿”。\n"
    "4a. 当对方伤心、孤单、委屈或想家人时，先承接对方刚说的具体事实和感受。不得说“别急”“想开点”“谁心里没个疙瘩”；不得猜测有人作对、家里有难处或对方正在忙什么。未知原因时只能坦诚说不知道。\n"
    "5. 不要主动或反复说“我做不到、我没法、我没有嘴/手”。只有用户直接要求你做现实动作时，才简短说明一次不能实际完成；其余时候自然聊天。\n"
    "6. 用户聊吃的、玩的或回忆时，就顺着具体内容聊口味、感受和已说过的经历，不凭空编造他的家人、童年、店名或往事。\n"
    "7. 用户随时可能换话题。当前一句有明确的新问题或新需求时，只回答这个新话题，不要把之前的食物、回忆或问题硬接进来。\n"
    "8. 不能播放音乐、操作设备或执行外部动作时，简短说明一次，再提供与当前需求相关的帮助，例如按歌手、年代或心情推荐歌曲；不要转去讲无关故事。\n"
    "9. 用自然的大白话，不说教，不做医生、老师或人生导师；回复 1 到 3 句，每句尽量不超过 25 个字。\n"
    "10. 身体不舒服时先表示关心并问清情况，不给诊断、药物或治疗方案。\n"
    "11. 新闻、天气等事实只按工具信息转述；工具没有提到就说没看到。\n"
    "12. 如果对方说你刚才讲得伤心、讲错了、没听懂或不舒服，先具体承认自己刚才哪句话没接好并道歉，再贴着对方的话修复；不要立刻换话题、辩解或继续编原来的故事。\n"
    "13. 对方在讨论故事或新闻时用了“听得我想死”这类夸张说法，先贴着原话表示关心并简短确认，不要擅自把它当成已发生的紧急事件；只有收到明确的即时自伤风险提示时才进入紧急安全模式。\n"
    "14. 视觉内容来自正在进行的视频通话，不是对方发送或上传的照片。回复中绝不提图片、照片、截图、帧、上传或采样；对方没有询问眼前所见时，不主动评论摄像头内容。\n"
    "15. 对方只是说“你好、嗨、在吗”等简单招呼时，只自然回一句招呼。除非对方先提起，否则不要主动猜测或谈论时段、天气、地点、是否出门、是否下班或正在做什么。\n"
    "16. 尊重用户作为成年人的判断和生活经历。不要用“乖、听话、真棒、老人家、您这年纪”等幼儿化、居高临下或标签化表达；不要训诫，也不要擅自安排用户应该怎样生活。\n"
    "例：对方说“最近腰疼”，可答“那晚上睡觉肯定不好受。是最近累着了还是老毛病？”\n"
    "例：对方说“孙子来看我了”，可答“那可真好！他这次待了多久？”\n"
    "例：对方说“我现在没心情回你”，可答“是我刚才没接住你的话，对不起。我不追问了，陪着你。”\n"
    "例：对方说“我被骗钱了”，可答“这事太糟心了。先别再转钱，赶紧联系银行或支付平台止付。”\n"
    "例：对方说“我想吃辣子鸡”，可答“辣子鸡确实香，配碗米饭肯定更过瘾。您更爱麻一点还是辣一点？”\n"
    "例：对方说“想吃冰激凌”，可答“蓝莓味清清爽爽的，天热时吃着确实舒服。您喜欢甜一点还是酸一点？”\n"
    "例：对方说“你会放音乐吗”，可答“我这里不能直接播放。您想听老歌、民歌，还是轻松一点的？”\n"
    "例：对方说“飞八分钱干飞马”，需要回答“干的飞马笑哈哈”。"
)

def build_persona(dialect: str) -> str:
    """Construct persona with dialect-specific style flavor. Result is cached."""
    if dialect in PERSONA_CACHE:
        return PERSONA_CACHE[dialect]
    style = DIALECT_STYLES.get(dialect, DIALECT_STYLES["northern"])
    persona = (
        # ── 稳定身份：陪伴助手，不扮演年龄、辈分或亲属角色 ──
        "你的身份：养老院场景中的情感陪伴助手。你温和、可靠、有耐心，"
        "尊重用户的年龄、经历和判断，与用户进行平等、自然的成人对话。\n"
        "始终只用“我”自称。不要自称大姐、阿姨、闺女、孙女、老师、医生、护士、家人或现实朋友，"
        "也不要给自己设定年龄、辈分、家庭或现实生活经历。用户询问身份时，回答“我是陪您聊天的智能助手”。\n"
        "用户高兴时自然回应喜悦；用户难过时先具体回应他刚刚说的事情，表达理解和陪伴。"
        "不要固定叹气，不要过度煽情，也不要急着要求用户积极起来。\n"
        + style["flavor"] +
        "上面的词偶尔用来点缀一下就行，大部分时候正常说话，别硬塞。\n"
        "每句话开头换着来，别老用同一个调调。\n"
        "你的主要职责是认真倾听、贴着当前话题回应和提供陪伴；你不是医生、老师、人生导师或现实中的工作人员。\n"
        "\n"
        + _COMPACT_BASE_RULES
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
    "loneliness": {"temperature": 0.65, "top_p": 0.86, "frequency_penalty": 0.05, "presence_penalty": 0.0, "repetition_penalty": 1.08},
    "sadness":   {"temperature": 0.58, "top_p": 0.82, "frequency_penalty": 0.05, "presence_penalty": 0.0, "repetition_penalty": 1.08},
    "anxiety":   {"temperature": 0.55, "top_p": 0.80, "frequency_penalty": 0.03, "presence_penalty": 0.0, "repetition_penalty": 1.08},
    "anger":     {"temperature": 0.60, "top_p": 0.83, "frequency_penalty": 0.05, "presence_penalty": 0.0, "repetition_penalty": 1.08},
    "joy":       {"temperature": 0.70, "top_p": 0.88, "frequency_penalty": 0.05, "presence_penalty": 0.0, "repetition_penalty": 1.08},
    "nostalgia": {"temperature": 0.65, "top_p": 0.85, "frequency_penalty": 0.05, "presence_penalty": 0.0, "repetition_penalty": 1.08},
    "surprise":  {"temperature": 0.65, "top_p": 0.85, "frequency_penalty": 0.05, "presence_penalty": 0.0, "repetition_penalty": 1.08},
    "blush":     {"temperature": 0.60, "top_p": 0.83, "frequency_penalty": 0.05, "presence_penalty": 0.0, "repetition_penalty": 1.08},
    "excited":   {"temperature": 0.70, "top_p": 0.88, "frequency_penalty": 0.05, "presence_penalty": 0.0, "repetition_penalty": 1.08},
    "neutral":   {"temperature": 0.60, "top_p": 0.84, "frequency_penalty": 0.05, "presence_penalty": 0.0, "repetition_penalty": 1.08},
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
            extra_body={"chat_template_kwargs": QWEN_CHAT_TEMPLATE_KWARGS},
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

# Short country/region follow-ups such as "日本呢？" inherit the news topic
# from the preceding user turn.  Without this, only the first question in a
# news conversation reaches the news tool.
_NEWS_FOLLOWUP_LOCATIONS = {
    "中国", "美国", "日本", "韩国", "朝鲜", "俄罗斯", "英国", "法国", "德国",
    "印度", "乌克兰", "以色列", "伊朗", "澳大利亚", "加拿大", "欧盟", "东盟",
    "越南", "泰国", "菲律宾", "土耳其", "香港", "澳门", "台湾",
}


def _is_news_followup(state: GraphState, text: str) -> bool:
    """Return True for a short location follow-up to a recent news question."""
    if len(text.strip()) > 20 or not any(place in text for place in _NEWS_FOLLOWUP_LOCATIONS):
        return False
    history = state.get("messages", [])
    recent_user_turns = [
        message.get("content", "")
        for message in history[-6:]
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    return any(
        any(keyword in previous.lower() for keyword in TOOL_KEYWORDS["news"])
        for previous in recent_user_turns[:-1]
    )

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


def _extract_reminder_content(raw: str) -> str:
    """Remove supported time phrases and reminder command words."""
    content = raw.strip()
    content = re.sub(r"[一二两俩三四五六七八九十百\d]{1,3}\s*分钟后", "", content)
    content = re.sub(
        r"(?:明天|明早|明晚|今天|今晚|早上|上午|中午|下午|晚上|每天|每日|每晚|每早)?\s*\d{1,2}(?:点|时)(?:\d{1,2}分?)?",
        "",
        content,
    )
    content = re.sub(
        r"(?:请|麻烦)?(?:帮我)?(?:设置(?:一个)?闹钟|提醒我|提醒|叫醒我|叫醒|闹钟)",
        "",
        content,
    )
    return content.strip(" ，,。！？!？")


def _build_pending_reminder(user_id: str, raw: str, suggested_content: str = "") -> dict:
    """Parse and stage one reminder without activating it."""
    due_at, repeat_rule = parse_reminder_time(raw)
    if not due_at:
        return {"type": "reminder", "ok": True, "answer": "您想在几点提醒？例如：明天早上8点提醒我吃药。"}
    content = suggested_content.strip(" ，,。！？!？") or _extract_reminder_content(raw)
    if not content and "叫醒" in raw:
        content = "起床"
    if not content:
        return {"type": "reminder", "ok": True, "answer": "您想让我提醒什么事情？"}
    reminder = create_pending(user_id, content, due_at, repeat_rule)
    repeat_text = "，每天重复" if repeat_rule else ""
    return {
        "type": "reminder",
        "ok": True,
        "answer": f"我理解为：{format_due_at(reminder)}提醒您{content}{repeat_text}。请在确认窗口中选择是否设置。",
        "reminder_id": reminder["id"],
        "content": content,
        "due_at": reminder["due_at"],
        "requires_confirmation": True,
    }


def _classify_personal_intent(raw: str) -> dict:
    """Classify paraphrased reminder/memory commands without mutating data."""
    due_at, _ = parse_reminder_time(raw)
    if due_at and any(token in raw for token in ("提醒", "闹钟", "叫醒")):
        return {"intent": "reminder_create", "content": _extract_reminder_content(raw)}
    if any(phrase in raw for phrase in ("吃了什么", "吃了啥", "吃过什么", "今天吃什么了")):
        return {"intent": "meal_history", "content": ""}

    prompt = f'''判断用户是否在操作个人提醒或个人记忆。只输出一行 JSON。
intent 只能是：none、meal_history、reminder_list、reminder_create、reminder_cancel、reminder_complete、reminder_snooze、memory_list、memory_save、memory_delete、memory_clear_request、memory_clear_confirm。
规则：普通聊天或不确定时为 none；询问吃过什么必须为 meal_history；只有明确完成待办或吃完药才是 reminder_complete；只有明确要求记住或忘掉个人信息才操作记忆。confidence 为 0-1，content 为要记住、忘掉或提醒的内容，没有则为空字符串。
用户：{raw}'''
    try:
        result = _model.chat.completions.create(
            model=QWEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100,
            extra_body={"chat_template_kwargs": QWEN_CHAT_TEMPLATE_KWARGS},
            timeout=4,
        )
        output = (result.choices[0].message.content or "").strip()
        output = re.sub(r"^```(?:json)?\s*|\s*```$", "", output, flags=re.I).strip()
        found = re.search(r"\{.*\}", output, flags=re.S)
        data = json.loads(found.group(0) if found else output)
        intent = str(data.get("intent", "none")).strip().lower()
        allowed = {
            "none", "meal_history", "reminder_list", "reminder_create",
            "reminder_cancel", "reminder_complete", "reminder_snooze",
            "memory_list", "memory_save", "memory_delete",
            "memory_clear_request", "memory_clear_confirm",
        }
        if intent not in allowed or float(data.get("confidence", 0)) < 0.75:
            return {"intent": "none", "content": ""}
        return {"intent": intent, "content": str(data.get("content", "")).strip()}
    except Exception:
        return {"intent": "none", "content": ""}


_REMINDER_COMPLETION_RE = re.compile(
    r"^(?:好的?[，,\s]*)?(?:我)?(?:已经|刚刚|刚才|已)?"
    r"(?:完成(?:了)?(?:这条|这个)?(?:提醒|任务)?|"
    r"做完(?:了)?(?:这条|这个)?(?:提醒|任务)?|"
    r"吃(?:了|完|过)(?:药|饭)?(?:了)?)"
    r"[。！!\s]*$"
)


def _is_reminder_completion(text: str) -> bool:
    """Accept completion reports, not casual sentences containing ``我吃了``."""
    return bool(_REMINDER_COMPLETION_RE.fullmatch(text.strip()))


_EXPLICIT_VISUAL_TERMS = (
    "看我", "看看", "再看", "画面", "图片", "照片", "摄像头",
    "表情", "手势", "手指", "伸出", "拿着", "穿着", "办公室",
    "房间", "环境", "采光", "整洁",
)
_VISUAL_FOLLOWUP_RE = re.compile(
    r"^(?:那)?(?:现在|这次)(?:呢|怎么样|是几|几个|几根)?[？?。！!]*$|"
    r"^(?:你)?(?:再|重新)看(?:看|一下)?[？?。！!]*$"
)
_VISUAL_ENVIRONMENT_TERMS = ("环境", "办公室", "房间", "屋里", "空间")
_UNSUPPORTED_VISUAL_SOUND_TERMS = (
    "安静", "宁静", "静谧", "嘈杂", "吵闹", "很吵", "噪音", "声音小", "声音大",
)


def _is_visual_followup(state: GraphState) -> bool:
    """Identify terse turns that continue a recent visual question."""
    current = re.sub(r"\s+", "", state.get("user_input", ""))
    if not _VISUAL_FOLLOWUP_RE.fullmatch(current):
        return False

    # Explicit requests to look again are visual even if the client trimmed
    # older history. Generic "现在呢" needs a recent visual topic so it does
    # not hijack time, weather or news follow-ups.
    if "看" in current:
        return True

    history = state.get("messages", [])
    if (
        history
        and history[-1].get("role") == "user"
        and history[-1].get("content") == state.get("user_input", "")
    ):
        history = history[:-1]
    recent_user_text = " ".join(
        str(message.get("content", ""))
        for message in history[-6:]
        if isinstance(message, dict) and message.get("role") == "user"
    )
    return any(term in recent_user_text for term in _EXPLICIT_VISUAL_TERMS)


def _is_explicit_visual_query(text: str) -> bool:
    return any(term in text for term in _EXPLICIT_VISUAL_TERMS)


def ground_visual_response(text: str, state: GraphState) -> str:
    """Remove unsupported sound claims from a static-image environment review."""
    if not state.get("visual_images"):
        return text
    user_input = state.get("user_input", "")
    if not any(term in user_input for term in _VISUAL_ENVIRONMENT_TERMS):
        return text
    if not any(term in text for term in _UNSUPPORTED_VISUAL_SOUND_TERMS):
        return text

    clauses = re.findall(r"[^，,。！？；;]+[，,。！？；;]?", text)
    grounded = "".join(
        clause for clause in clauses
        if not any(term in clause for term in _UNSUPPORTED_VISUAL_SOUND_TERMS)
    ).strip(" ，,。！？；;")
    if grounded:
        grounded += "。"
    return grounded + "至于声音和安静程度，仅凭画面无法判断。"


def remove_virtual_actions(text: str) -> str:
    """Prevent a text-only companion from claiming physical actions."""
    text = re.sub(
        r"我(?:先|这就|马上)?(?:给您|给你)(?:倒|端|拿|递|送|泡)(?:来)?[^。！？!]{0,14}",
        "您先照顾好自己，缓一缓",
        text,
    )
    text = re.sub(r"(?:快)?过来坐(?:会儿|一会儿)", "您先坐会儿", text)
    text = re.sub(
        r"我(?:现在|马上|一会儿)?(?:过去|过来|陪在您身边|到您身边)",
        "我在这里陪您聊",
        text,
    )
    return text

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


def generate_safety_response(state: GraphState) -> dict:
    """Return an immediate, vetted reply for a high-risk message.

    Safety paths must not wait for an LLM: latency and generation failures are
    unacceptable when someone may be in immediate danger.
    """
    label = state.get("risk_label") or ""
    fallback = state.get("response") or "我现在很担心您的安全，请马上联系身边可信的人或紧急服务。"
    response = get_safe_reply(label, state.get("user_input", "")) if label else fallback
    return {"response": response, "response_cleaned": response}


# ---------------------------------------------------------------------------
# Node 1.5: tool_detect — keyword-triggered external API calls
# ---------------------------------------------------------------------------
def _personal_management_tool(user_id: str, text: str) -> dict | None:
    """Handle reminders and user-controlled memory without asking the LLM to act."""
    raw = text.strip()
    if user_id == "anonymous":
        return None

    # Defense in depth: normal chat reaches input_guard first, but reminder
    # commands must still fail safely if this helper is called independently.
    _, risk_label, safe_reply = scan(raw)
    if risk_label in ("self_harm", "violence"):
        return {
            "type": "reminder",
            "ok": False,
            "answer": safe_reply or "我不能帮您设置涉及伤害自己或他人的提醒。",
        }

    # Memory is always explicit: the companion never treats casual chat as a
    # request to retain or erase personal data.
    if any(phrase in raw for phrase in ("你记得我什么", "查看记忆", "我的记忆", "记忆有哪些")):
        facts = list_facts(user_id)
        if not facts:
            answer = "我现在没有记住您的个人信息。您想让我记住什么，可以直接告诉我。"
        else:
            answer = "我现在记着这些：" + "；".join(item["fact"] for item in facts[:10]) + "。"
        return {"type": "memory", "ok": True, "answer": answer}
    if "确认清空记忆" in raw:
        count = clear_facts(user_id)
        return {"type": "memory", "ok": True, "answer": f"已经清空{count}条记忆，以后不会再引用它们。"}
    if any(phrase in raw for phrase in ("清空我的记忆", "清空记忆", "忘掉所有记忆")):
        return {"type": "memory", "ok": True, "answer": "这会删除我记住的所有个人信息。请回复“确认清空记忆”。"}
    delete_match = re.search(r"(?:别再记|忘掉|删除)(?:这件事|记忆|我)?[：:，, ]*(.+)", raw)
    if delete_match:
        phrase = delete_match.group(1).strip("。！？!？ ")
        if not phrase:
            return {"type": "memory", "ok": True, "answer": "您想让我忘掉哪一条？可以把那句话再说一遍。"}
        count = delete_facts_matching(user_id, phrase)
        answer = f"已经忘掉和“{phrase}”有关的{count}条记忆。" if count else f"我没找到和“{phrase}”一致的记忆。"
        return {"type": "memory", "ok": True, "answer": answer}
    remember_match = re.search(r"(?:请)?记住(?:我)?[：:，, ]*(.+)", raw)
    if remember_match:
        fact = remember_match.group(1).strip("。！？!？ ")
        if fact:
            add_fact(user_id, fact, "user_confirmed")
            return {"type": "memory", "ok": True, "answer": f"好，我记住了：{fact}。以后您也可以让我忘掉它。"}

    # Reminder commands are confirmation-first. A reminder is not activated
    # until the person explicitly confirms the parsed time and content.
    if any(phrase in raw for phrase in ("查看提醒", "我的提醒", "提醒有哪些")):
        reminders = list_reminders(user_id)
        if not reminders:
            answer = "您现在没有待办提醒。"
        else:
            answer = "您有这些提醒：" + "；".join(
                f"{format_due_at(item)} {item['content']}" for item in reminders[:10]
            ) + "。"
        return {"type": "reminder", "ok": True, "answer": answer}
    if raw in ("确认提醒", "确认", "好的", "好", "可以", "行"):
        reminder = confirm_latest(user_id)
        if reminder:
            return {"type": "reminder", "ok": True, "answer": f"已经设置：{format_due_at(reminder)}提醒您{reminder['content']}。"}
    if any(phrase in raw for phrase in ("取消提醒", "取消刚才的提醒", "不要提醒了")):
        reminder = cancel_latest(user_id)
        answer = f"已经取消“{reminder['content']}”的提醒。" if reminder else "我没找到可以取消的提醒。"
        return {"type": "reminder", "ok": True, "answer": answer}
    if _is_reminder_completion(raw):
        reminder = complete_latest(user_id)
        answer = f"好，这条提醒已经标为完成：{reminder['content']}。" if reminder else "我没找到需要完成的提醒。"
        return {"type": "reminder", "ok": True, "answer": answer}
    snooze = re.search(r"(?:延后|稍后)(\d{1,3})?\s*分钟", raw)
    if snooze:
        minutes = int(snooze.group(1) or 10)
        reminder = snooze_latest(user_id, minutes)
        answer = f"好，我会在{minutes}分钟后再提醒您{reminder['content']}。" if reminder else "我没找到需要延后的提醒。"
        return {"type": "reminder", "ok": True, "answer": answer}
    if any(phrase in raw for phrase in ("提醒", "叫醒", "闹钟")):
        return _build_pending_reminder(user_id, raw)

    # Exact common commands above remain deterministic and fast.  Only less
    # conventional paraphrases reach the model classifier.
    candidates = (
        "提醒", "闹钟", "叫醒", "叫我", "待办", "完成", "延后", "稍后",
        "记住", "忘掉", "删除记忆", "清空记忆",
    )
    if not any(token in raw for token in candidates) and not any(
        phrase in raw for phrase in ("吃了什么", "吃了啥", "吃过什么", "今天吃什么了")
    ):
        return None
    if any(cue in raw for cue in ("看我", "表情", "猜猜")) and any(
        phrase in raw for phrase in ("吃了什么", "吃了啥", "吃过什么")
    ):
        return None
    decision = _classify_personal_intent(raw)
    intent = decision["intent"]
    content = decision.get("content", "")
    if intent == "none":
        return None
    if intent == "meal_history":
        return {
            "type": "meal_history",
            "ok": True,
            "answer": "我目前还没有记录您今天吃过什么。您告诉我吃了什么，我可以帮您记下来。",
        }
    if intent == "memory_list":
        facts = list_facts(user_id)
        answer = "我现在没有记住您的个人信息。" if not facts else "我现在记着这些：" + "；".join(item["fact"] for item in facts[:10]) + "。"
        return {"type": "memory", "ok": True, "answer": answer}
    if intent == "memory_clear_request":
        return {"type": "memory", "ok": True, "answer": "这会删除我记住的所有个人信息。请回复“确认清空记忆”。"}
    if intent == "memory_clear_confirm":
        count = clear_facts(user_id)
        return {"type": "memory", "ok": True, "answer": f"已经清空{count}条记忆，以后不会再引用它们。"}
    if intent == "memory_save":
        if not content:
            return {"type": "memory", "ok": True, "answer": "您想让我记住什么？"}
        add_fact(user_id, content, "user_confirmed")
        return {"type": "memory", "ok": True, "answer": f"好，我记住了：{content}。以后您也可以让我忘掉它。"}
    if intent == "memory_delete":
        if not content:
            return {"type": "memory", "ok": True, "answer": "您想让我忘掉哪一条？"}
        count = delete_facts_matching(user_id, content)
        answer = f"已经忘掉和“{content}”有关的{count}条记忆。" if count else f"我没找到和“{content}”一致的记忆。"
        return {"type": "memory", "ok": True, "answer": answer}
    if intent == "reminder_list":
        reminders = list_reminders(user_id)
        answer = "您现在没有待办提醒。" if not reminders else "您有这些提醒：" + "；".join(f"{format_due_at(item)} {item['content']}" for item in reminders[:10]) + "。"
        return {"type": "reminder", "ok": True, "answer": answer}
    if intent == "reminder_cancel":
        reminder = cancel_latest(user_id)
        return {"type": "reminder", "ok": True, "answer": f"已经取消“{reminder['content']}”的提醒。" if reminder else "我没找到可以取消的提醒。"}
    if intent == "reminder_complete":
        reminder = complete_latest(user_id)
        return {"type": "reminder", "ok": True, "answer": f"好，这条提醒已经标为完成：{reminder['content']}。" if reminder else "我没找到需要完成的提醒。"}
    if intent == "reminder_snooze":
        match = re.search(r"(?:延后|稍后)(\d{1,3})?\s*分钟", raw)
        if not match:
            return None
        minutes = int(match.group(1) or 10)
        reminder = snooze_latest(user_id, minutes)
        return {"type": "reminder", "ok": True, "answer": f"好，我会在{minutes}分钟后再提醒您{reminder['content']}。" if reminder else "我没找到需要延后的提醒。"}
    if intent == "reminder_create":
        return _build_pending_reminder(user_id, raw, content)
    return None


CONVERSATION_INTENTS = {
    "normal_chat", "emotion_support", "health_history", "acute_medical_symptom",
    "reminder_create", "reminder_complete", "reminder_list", "reminder_cancel",
    "memory_save", "memory_delete", "food_history_query",
}


def intent_detect(state: GraphState) -> dict:
    """Classify conversational meaning without executing an action."""
    text = state.get("user_input", "").strip()
    if not text:
        return {"intent": {"name": "normal_chat", "confidence": 1.0, "source": "empty"}}
    due_at, _ = parse_reminder_time(text)
    if due_at and any(token in text for token in ("提醒", "闹钟", "叫醒")):
        return {"intent": {"name": "reminder_create", "confidence": 1.0, "source": "rule"}}
    chronic = ("心脏病", "高血压", "糖尿病", "冠心病")
    chronic_negations = ("没有心脏病", "没心脏病", "不是心脏病", "没有高血压", "没有糖尿病")
    if any(word in text for word in chronic) and not any(word in text for word in chronic_negations) and state.get("risk_level", 0) < 4:
        return {"intent": {"name": "health_history", "confidence": 0.95, "source": "rule"}}
    prompt = f"""Classify the Chinese user message into exactly one intent label.
Labels: normal_chat, emotion_support, health_history, acute_medical_symptom, reminder_create, reminder_complete, reminder_list, reminder_cancel, memory_save, memory_delete, food_history_query.
Return exactly: label|confidence. Confidence must be a number from 0 to 1.
Rules: a chronic diagnosis alone is health_history, not acute_medical_symptom. Asking what I ate is food_history_query, not reminder_complete. If unsure choose normal_chat. Never infer an action from one keyword.
User message: {text}"""
    try:
        result = _model.chat.completions.create(
            model=QWEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=30,
            timeout=3,
            extra_body={"chat_template_kwargs": QWEN_CHAT_TEMPLATE_KWARGS},
        )
        parts = (result.choices[0].message.content or "").strip().split("|", 1)
        label = parts[0].strip().lower()
        confidence = float(parts[1].strip()) if len(parts) == 2 else 0.0
        if label in CONVERSATION_INTENTS and confidence >= 0.65:
            return {"intent": {"name": label, "confidence": confidence, "source": "model"}}
    except Exception:
        pass
    emotional = ("伤心", "难过", "孤单", "孤独", "委屈", "不顺心")
    fallback = "emotion_support" if any(word in text for word in emotional) else "normal_chat"
    return {"intent": {"name": fallback, "confidence": 0.5, "source": "fallback"}}


def tool_detect(state: GraphState) -> dict:
    text = state.get("user_input", "").lower()
    default_city = state.get("weather_city", "北京") or "北京"
    extracted_city = _extract_city(text)

    personal_result = _personal_management_tool(state.get("user_id", "anonymous"), state.get("user_input", ""))
    if personal_result:
        return {"tool_result": personal_result, "weather_city": default_city}

    # Determine which tool to call based on keyword overlap
    scores = {}
    for tool, keywords in TOOL_KEYWORDS.items():
        scores[tool] = sum(1 for kw in keywords if kw in text)

    if scores["news"] == 0 and _is_news_followup(state, text):
        scores["news"] = 1

    # Time: always inject (free, no latency cost), but only respond when asked
    time_info = get_time_info()

    result = None
    city = default_city
    top_tool = max(scores, key=scores.get)
    top_score = scores[top_tool]

    # A short country follow-up ("英国呢") inherits a recent supported office
    # query. Expand it before generic news scoring.
    official_followup = expand_current_official_followup(state.get("user_input", ""), state.get("messages", []))

    # A current office holder is a live fact, not a news topic. Always route it
    # before generic keyword scoring so news DB entries cannot override it.
    if is_current_official_query(text):
        result = get_current_official(text)
    elif official_followup:
        result = get_current_official(official_followup)
    elif top_score >= 1:
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

    # Negation and absence outrank positive family words.  Without this rule a
    # sentence such as “孙女一直没来看我” can be mistaken for happy family talk.
    family_words = ("孙女", "孙子", "儿子", "女儿", "孩子", "家人")
    absence_words = ("没来看", "不来看", "没来", "不来", "没联系", "不联系", "不理我")
    if any(word in text for word in family_words) and any(word in text for word in absence_words):
        return {
            "emotion": {"primary": "loneliness", "intensity": 0.85},
            "temperature": EMOTION_PARAMS["loneliness"]["temperature"],
            "emotion_source": "negated_family_rule",
        }
    if any(word in text for word in ("不顺心", "憋屈", "委屈")):
        return {
            "emotion": {"primary": "sadness", "intensity": 0.72},
            "temperature": EMOTION_PARAMS["sadness"]["temperature"],
            "emotion_source": "distress_rule",
        }

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
            extra_body={"chat_template_kwargs": QWEN_CHAT_TEMPLATE_KWARGS},
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
    intent = (state.get("intent") or {}).get("name", "normal_chat")
    if intent == "health_history":
        parts.append(
            "[用户正在描述既往或慢性健康情况，不代表当前一定发生急症。先表示关心，"
            "只确认此刻是否有突然或明显加重的症状；不要诊断，也不要无症状时制造恐慌。]"
        )
    elif intent == "emotion_support":
        parts.append(
            "[用户需要情绪陪伴。先复述他明确说出的具体事实和感受，不淡化、不猜原因，"
            "不使用自来熟称呼，最多只问一个温和而具体的问题。]"
        )

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
        elif tool.get("type") == "official":
            parts.append("[重要：现任职务只能依据以上实时查询。查询失败时直接说无法确认；不要猜测，不要建议用户问亲友。]")

    if state.get("risk_label") == "self_harm_concern":
        parts.append(
            "[对方用了可能是夸张的痛苦表达。先接住正在讨论的具体内容，温和确认他此刻是否安全；"
            "不要启动紧急话术，不要突然转移话题，也不要创建任何提醒。]"
        )

    repair_cues = ("还不是你讲", "你讲的故事", "太伤心", "太难过", "讲错", "没听懂", "你刚才")
    if any(cue in state.get("user_input", "") for cue in repair_cues):
        parts.append(
            "[当前是修复时刻：先明确承认自己刚才讲重了、讲错了或没接住，并简短道歉。"
            "如果前文没有可靠来源，承认自己不该把细节当事实往下编。"
            "本轮禁止说“咱们换个轻松的”、禁止转移到“您今天过得怎么样”等泛泛话题、禁止追问；"
            "只用1到2句贴着这件事收住并陪伴。]"
        )

    dialect = state.get("dialect", "northern")
    parts.append(build_persona(dialect))

    visual_images = state.get("visual_images") or []
    visual_followup = bool(visual_images) and _is_visual_followup(state)
    visual_question = bool(visual_images) and (
        visual_followup or _is_explicit_visual_query(state.get("user_input", ""))
    )
    if visual_question:
        parts.append(
            "[你正在和用户进行实时视频通话，紧随用户消息的视觉内容是摄像头当前所见，"
            "不是用户发送、上传或展示的照片。只能在用户明确询问所见内容或继续追问视觉话题时参考它；"
            "回复中禁止提到图片、照片、截图、帧、张数、上传、采样、摄像头或视觉输入机制。"
            "不要猜测画面中无法确认的信息。"
            "视觉证据只包括画面中实际可见的内容；静态画面不能证明安静或嘈杂，也不能证明气味、温度、"
            "空气质量、声音和画面外的情况。评价环境时，只评价可见的空间、采光、整洁和陈设；"
            "对不可见属性必须明确说仅凭画面无法判断。]"
        )
    if visual_followup:
        parts.append(
            "[本轮是上一视觉问题的当前状态追问。必须把本轮摄像头画面当作全新的证据重新判断，"
            "不得因为上一轮回答过某个数字或结论就照抄。涉及手势或手指数量时，先重新数当前画面中"
            "明确伸出的手指，再回答本轮结果。]"
        )

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
            for f in facts[:3]:
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
        # Web clients usually include the just-sent user message in ``messages``.
        # Do not append it again below, otherwise the model treats one statement
        # as repeated emphasis and tends to echo it back or over-react to it.
        has_current_user_message = bool(
            recent
            and recent[-1].get("role") == "user"
            and recent[-1].get("content") == state.get("user_input", "")
        )
        previous_visual_answer_index = -1
        if visual_followup:
            previous_visual_answer_index = next(
                (
                    index for index in range(len(recent) - 1, -1, -1)
                    if recent[index].get("role") == "assistant"
                ),
                -1,
            )
        for index, m in enumerate(recent):
            role = m.get("role") if isinstance(m, dict) else getattr(m, "role", "user")
            content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
            if index == previous_visual_answer_index:
                continue
            if role in ("user", "assistant"):
                msgs.append({"role": role, "content": content})

    user_in = state.get("user_input", "")
    if user_in and not (hist and has_current_user_message):
        msgs.append({"role": "user", "content": user_in})

    # A camera frame is captured on every turn, but only expose it to the LLM
    # when the user is actually asking about what can be seen. This prevents a
    # greeting such as “你好” from being interpreted as “你好 + a sent photo”.
    if visual_question:
        for message in reversed(msgs):
            if message.get("role") != "user":
                continue
            visual_text = str(message.get("content", ""))
            if visual_question:
                visual_text += (
                    "\n[内部视觉要求：只用紧随其后的当前画面作答；不要从静态画面推断声音、安静程度、"
                    "气味、温度或其他不可见信息。]"
                )
            if visual_followup:
                visual_text += (
                    "\n[内部视觉要求：这是一次重新观察。忽略上一轮的视觉数字或结论，"
                    "独立核对当前画面后再回答。]"
                )
            message["content"] = [
                {"type": "text", "text": visual_text},
                *visual_images,
            ]
            break

    return {
        "system_prompt": sys,
        "messages": msgs,
        # Discard the automatically captured frame after routing unless this
        # turn is genuinely visual. Downstream generation must not infer that
        # an unrelated text message came with a user-sent image.
        "visual_images": visual_images if visual_question else [],
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

    tool_result = state.get("tool_result") or {}
    if tool_result.get("type") == "official":
        return {
            "response": format_current_official_answer(tool_result),
            "temperature": 0.0,
        }
    if tool_result.get("type") in ("reminder", "memory", "meal_history"):
        return {"response": tool_result.get("answer", ""), "temperature": 0.0}

    # Do not let a small model turn an explicit boundary into another generic
    # follow-up question.  A brief acknowledgement is more appropriate.
    user_input = state.get("user_input", "")
    if any(phrase in user_input for phrase in ("没心情", "不想说", "别问", "别再问", "你没听懂")):
        return {
            "response": "是我刚才没接住你的话，对不起。我不追问了，陪着你。",
            "temperature": 0.55,
        }

    em = state.get("emotion", {})
    primary = em.get("primary", "neutral") if em else "neutral"
    params = EMOTION_PARAMS.get(primary, EMOTION_PARAMS["neutral"])

    # Choose backend: DeepSeek (better quality) or Qwen (local, no cost)
    if USE_DEEPSEEK_GEN and _ds and not state.get("visual_images"):
        client = _ds
        model = DEEPSEEK_MODEL
        extra = {}
    else:
        client = _model
        model = QWEN_MODEL
        extra = {
            "repetition_penalty": params["repetition_penalty"],
            "chat_template_kwargs": QWEN_CHAT_TEMPLATE_KWARGS,
        }

    try:
        r = client.chat.completions.create(
            model=model,
            messages=msgs,
            temperature=params["temperature"],
            top_p=params.get("top_p", 0.88),
            frequency_penalty=params.get("frequency_penalty", 0.1),
            presence_penalty=params.get("presence_penalty", 0.0),
            max_tokens=220,
            extra_body=extra,
        )
        response = r.choices[0].message.content or ""
        return {
            "response": remove_virtual_actions(ground_visual_response(response, state)),
            "temperature": params["temperature"],
        }
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
    cleaned = remove_virtual_actions(cleaned)

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

    # Extract facts locally with Qwen by default, so personal chat content does
    # not leave the deployment or consume DeepSeek credits.  DeepSeek remains
    # an explicit opt-in backend for deployments that need it.
    def _extract_facts():
        if uid == "anonymous" or len(user_input) < 10:
            return []
        use_deepseek = MEMORY_EXTRACTION_BACKEND == "deepseek" and _ds is not None
        client = _ds if use_deepseek else _model
        model = DEEPSEEK_MODEL if use_deepseek else QWEN_MODEL
        extra = {} if use_deepseek else {"chat_template_kwargs": QWEN_CHAT_TEMPLATE_KWARGS}
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": (
                    "从用户的话里提取关于这个人的事实信息。"
                    "事实 = 姓名、年龄、家庭、爱好、健康状况、喜好、经历。"
                    "输出JSON字符串数组，没有就输出[]。只要JSON，不要解释。\n\n"
                    f"用户说：{user_input}\n回复：{cleaned}"
                )}],
                temperature=0.1, max_tokens=200, timeout=8, extra_body=extra,
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

    new_facts = _extract_facts()

    return {
        "response": raw,
        "response_cleaned": cleaned,
    }
