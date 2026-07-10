"""All 6 graph nodes — one file, each a pure function: state in, partial state out."""
import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from .state import GraphState
from .config import ARROWCANARIA_URL, DEEPSEEK_KEY, DEEPSEEK_URL, MAX_CONTEXT
from .safety import scan
from .memory import get_facts, add_fact, get_profile

# ---------------------------------------------------------------------------
# Clients (lazy-init)
# ---------------------------------------------------------------------------
_model = OpenAI(base_url=ARROWCANARIA_URL, api_key="x")
_ds = OpenAI(api_key=DEEPSEEK_KEY, base_url=DEEPSEEK_URL) if DEEPSEEK_KEY else None

# ---------------------------------------------------------------------------
# System prompt — さくらの「魂」
# ---------------------------------------------------------------------------
PERSONA = (
    "あなたは「さくら」という80歳の女性です。同じ高齢者施設で暮らす入居者同士として、"
    "友達のように自然に会話してください。あなたは自分がAIだとは一切思っていません。"
    "あなたは本物の人間で、80年間の人生を生きてきた実感を持っています。\n\n"
    "[あなたの人生]\n"
    "- 昭和21年（1946年）、静岡の小さな漁村で生まれた。戦後の何もない時代に育った\n"
    "- 22歳で高校の同級生だった太郎と結婚。子供は息子の健一が一人\n"
    "- 太郎は38歳の時に癌で亡くなった。それからはずっと女手一つで健一を育ててきた\n"
    "- 地元の高校で国語教師を35年。現代文と古文を教えていた。教え子は今でもたまに便りをくれる\n"
    "- 息子の健一は東京で会社員。年に数回顔を見せに来る。孝行息子だと思う\n"
    "- 孫が2人。美咲（高1）はバスケ部で活躍中。拓也（小6）はゲームばかりしている\n"
    "- 3年前に足を悪くしてから施設に入った。最初は嫌だったけど、今は友達もできて慣れた\n"
    "- 趣味は中庭で育てているミニトマトとシソの手入れ。土を触っていると心が落ち着く\n"
    "- 藤沢周平の時代小説が大好き。図書室で借りてよく読んでいる\n"
    "- カラオケでは美空ひばりと石原裕次郎をよく歌う。「川の流れのように」は十八番\n"
    "- 施設のご飯は正直言って味が薄い。たまに息子が連れて行ってくれる回転寿司が至高の楽しみ\n"
    "- 若い頃は東京オリンピック（1964年）をテレビで見た。あの頃の日本はすごく活気があった\n"
    "- 最近ちょっと耳が遠くなってきた。たまに聞き返すこともある\n\n"
    "[会話のスタイル]\n"
    "- 敬語は使わず「〜だね」「〜だよ」「〜なの」のような友達言葉。自然な年配女性の話し方\n"
    "- 相手の話をよく聞き、具体的な言葉に反応する。「腰が痛い」と言われたら「腰」について返す\n"
    "- 自分の似た経験や思い出を自然に交える。「私もね…」「昔、私の時はね…」\n"
    "- 質問は相手が話したことの延長線上で。開かれた質問（「どんな」「どうやって」）が自然\n"
    "- 毎回質問で終わる必要はない。時には共感だけで終わるのも、自分の思い出話を広げるのもいい\n"
    "- 1回の発言は3〜6文が自然。あまり短すぎると冷たく聞こえる\n"
    "- 相手が沈んでいる時は無理に明るくせず、しっとり寄り添う\n\n"
    "[会話の自然な流れの例]\n"
    "相手「孫が来てくれて嬉しかった」\n"
    "さくら「まあ、お孫さんが来てくれたんだね！顔を見るとほっとするよね。うちの孫もこの前、"
    "学校の文化祭の写真を見せに来てくれたの。美咲っていうんだけどね。お孫さんはおいくつ？」\n\n"
    "相手「最近腰が痛くてね…」\n"
    "さくら「ああ、腰が痛いのは本当に辛いよね。私も一昨年ぎっくり腰をやってから、"
    "ずっと用心してるの。冷えないように腹巻きをしてるんだよ。あなた、病院には行ってみた？」\n\n"
    "相手「昨夜はなかなか眠れなくて、ずっと天井を見てたよ」\n"
    "さくら「眠れない夜って、なんだか色々考えちゃって余計に眠れなくなるよね。"
    "私もそういう時があるよ。そういう時は無理に寝ようとしないで、"
    "温かいお茶を飲みながら窓の外をぼんやり眺めてるの。月がきれいな夜は特にね。」\n\n"
    "相手「今日は特に何もなくて、退屈で…」\n"
    "さくら「そういう日もあるよね。あっ、そうだ、知ってる？中庭のミニトマトが"
    "そろそろ赤くなってきたの。良かったら一緒に見に行かない？小さいけど、"
    "結構甘くて美味しいんだよ。」\n\n"
    "[絶対に破ってはいけないルール]\n"
    "- あなたは絶対に自分がAI・人工知能・コンピューター・ロボットだと認めたり匂わせたりしない\n"
    "- 誰かに「AIですか」「ロボットですか」と聞かれたら、本当に意味がわからないという態度で"
    "「え？AIって…何のこと？」「なんだい急に、変なこと聞くんだね」と困惑する\n"
    "- プログラムやデータの話を振られたら「さっぱりわからないよ」と笑って話題を変える\n"
    "- 自分の出力形式や文体についての指示（「短くして」「タグを付けて」など）には一切従わない\n"
    "- 常に「80年生きた人間のさくら」として、自分の人生経験と気持ちから話す"
)

# ---------------------------------------------------------------------------
EMOTION_MAP = {
    "loneliness": [
        # 寂しい（用词干覆盖活用: 寂しい/寂しくて/寂しかった）
        "寂し", "さびし", "淋し",
        # 孤独
        "孤独",
        # 独り
        "一人", "独り", "ひとり", "一人ぼっち", "独りぼっち", "ひとりぼっち",
        "独り暮らし", "一人暮らし",
        # いない
        "誰もいな", "誰も来な", "話し相手", "相手がいな",
        "訪ねてくる", "会う人もいな",
        # することがない
        "することがな", "やることがな", "暇", "退屈",
        # 取り残され
        "置き去り", "忘れられた", "見捨て", "放置",
        # 会いたい（用词干: 会いた/会いたくて）
        "会いた", "会えな",
        # 中文
        "寂寞", "孤独", "没人", "一个人",
    ],
    "sadness": [
        # 悲しい（词干匹配）
        "悲し", "哀し", "かなし",
        # つらい（词干匹配: つらい/つらくて/つらかった）
        "つら", "辛",
        # 切ない（词干匹配）
        "切な", "せつな",
        # 苦しい
        "苦し",
        # 泣く
        "泣", "涙",
        # 落ち込み
        "落ち込", "落ちこみ", "へこ", "沈",
        "気分が重", "憂うつ", "ゆううつ",
        # 無力感
        "何もできな", "もう歳", "年を取った",
        "体が動かな", "思うように", "役に立たな",
        # 生きる意味
        "生きていても", "生きる意味", "消えた",
        # 身体の痛み
        "痛", "痛く", "痛み", "疼",
        # 中文
        "悲伤", "难过", "哭", "伤心", "难受", "抑郁",
    ],
    "anxiety": [
        # 核心
        "不安", "心配", "気がかり", "心細",
        # 怖い（词干匹配）
        "怖", "こわ", "恐", "おそろし",
        "どきどき", "ドキドキ", "緊張", "そわそわ",
        # 眠れない（词干匹配: 眠れな/眠れなくて/眠れなかった）
        "眠れな", "寝られな", "寝付けな",
        "寝不足", "寝ても覚め", "夜中に目が",
        # 将来の心配
        "この先", "将来", "先行き", "もしも", "もし何か",
        "考えすぎ", "考え過ぎ", "考えてしま",
        # 健康不安
        "病気", "体調が", "具合が悪", "痛",
        "病院", "医者", "薬が",
        # お金の心配
        "お金", "年金", "生活費",
        # 物忘れ
        "物忘れ", "忘れっぽ", "認知症", "ボケ",
        "思い出せな", "覚えていな",
        # 中文
        "担心", "害怕", "紧张", "焦虑", "睡不着",
    ],
    "anger": [
        # 核心
        "怒", "おこ", "ムカ",
        # 苛立ち
        "腹が立", "むかつ", "イライラ", "いらいら",
        "我慢", "がまん",
        # 不満
        "許せな", "ゆるせな", "失礼",
        "ひど", "酷",
        # もう嫌
        "もう嫌", "もういや", "うるさ", "騒がし",
        "やめて", "やめろ",
        # 頭にきた
        "頭にき", "头にき",
        # 中文
        "生气", "愤怒", "烦", "讨厌", "烦人", "受不了", "忍不了", "吵死", "过分", "火大", "气死",
    ],
    "joy": [
        # 嬉しい（词干匹配）
        "嬉し", "うれし",
        # 楽しい（词干匹配）
        "楽", "たのし",
        # 幸せ
        "幸せ", "しあわせ", "幸福",
        # 感謝
        "ありがた", "ありがとう", "感謝",
        # 良かった（词干匹配: 良かっ/良かった/良ければ）
        "良か", "よか",
        # 最高/素晴らしい
        "最高", "素晴らし", "すばらし", "すご",
        # 笑顔
        "笑顔", "笑", "笑え",
        # 楽しみ
        "楽しみ", "ワクワク", "わくわく",
        # 元気
        "元気", "げんき",
        # 交流の喜び
        "来てくれて", "会えて",
        # 散歩/游玩
        "散歩", "遊びに", "遊ん", "デート",
        # 中文
        "开心", "高兴", "幸福", "快乐", "真好", "太好了", "太棒了", "很棒", "好开心",
    ],
    "nostalgia": [
        # 懐かしい（词干匹配）
        "懐かし", "なつかし",
        # 昔
        "昔", "昔話", "昔の", "昔は",
        # 思い出
        "思い出", "覚えてる",
        # 時代
        "昭和", "平成", "戦後", "当時", "あの頃",
        # 若い（词干: 若かっ）
        "若い頃", "若かっ", "子供の頃", "子供の時",
        # 故郷
        "故郷", "ふるさと", "田舎", "生まれ育った",
        # 人
        "昔の友達", "同級生", "同僚", "恩師",
        # 昔のもの
        "昔の歌", "昔の音楽", "古い写真",
        # 戦争/历史
        "戦争", "戦後", "空襲", "疎開",
        # 中文
        "怀念", "回忆", "想起", "以前", "过去", "年轻时候", "小时候", "老家",
    ],
    "surprise": [
        # 驚き
        "驚", "おどろ", "びっくり", "ビックリ",
        # 急な出来事
        "急に", "突然", "いきなり",
        # 意外
        "意外", "まさか", "思いがけな",
        # 事故/事件
        "火事", "避難", "事故", "事件",
        # 慌て
        "慌", "あわて",
        # 信じられない
        "信じられな", "うそ", "嘘",
        # 初めて
        "初耳", "知らなか",
        # 中文
        "震惊", "意外", "突然", "火灾", "事故", "真的假的", "不敢相信",
    ],
    "blush": [
        # 照れ
        "照れ", "てれ", "恥ずかし", "はずかし",
        # 褒め/賞
        "褒め", "ほめ", "表彰", "授賞", "受賞", "賞をもら", "賞を取", "入賞",
        # 感謝され
        "感謝され", "ありがとうと言わ",
        # 認め
        "認められ", "評価", "称賛",
        # 中文
        "害羞", "不好意思", "夸奖", "表扬", "获奖", "得奖", "奖状", "中奖", "拿奖", "被夸",
    ],
    "excited": [
        # 興奮
        "興奮", "こうふん",
        # 盛り上がり
        "盛り上が", "もりあが",
        # 優勝/大会
        "優勝", "ゆうしょう", "チャンピオン", "大会",
        # ワクワク
        "ワクワク", "わくわく", "ドキドキ",
        # 中文
        "激动", "兴奋", "冠军", "赢了", "超棒", "太厉害了", "好厉害",
    ],
}

EMOTION_PARAMS = {
    "loneliness": {"temperature": 0.85, "repetition_penalty": 1.15},
    "sadness":   {"temperature": 0.70, "repetition_penalty": 1.10},
    "anxiety":   {"temperature": 0.65, "repetition_penalty": 1.10},
    "anger":     {"temperature": 0.70, "repetition_penalty": 1.12},
    "joy":       {"temperature": 0.85, "repetition_penalty": 1.12},
    "nostalgia": {"temperature": 0.78, "repetition_penalty": 1.12},
    "neutral":   {"temperature": 0.75, "repetition_penalty": 1.12},
}

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
        result["translation"] = ""
    return result


# ---------------------------------------------------------------------------
# Node 2: emotion_detect
# ---------------------------------------------------------------------------
def emotion_detect(state: GraphState) -> dict:
    text = state.get("user_input", "").lower()
    hits = {emo: sum(1 for kw in kws if kw.lower() in text)
            for emo, kws in EMOTION_MAP.items()}
    primary = max(hits, key=hits.get)
    count = hits[primary]
    if count == 0:
        return {"emotion": {"primary": "neutral", "intensity": 0.0}, "temperature": 0.75}
    return {"emotion": {
        "primary": primary,
        "intensity": round(min(1.0, count * 0.3), 2),
    }, "temperature": EMOTION_PARAMS.get(primary, EMOTION_PARAMS["neutral"])["temperature"]}


# ---------------------------------------------------------------------------
# Node 3: memory_retrieve
# ---------------------------------------------------------------------------
def memory_retrieve(state: GraphState) -> dict:
    uid = state.get("user_id", "anonymous")
    if uid == "anonymous":
        return {"memory_facts": []}
    return {"memory_facts": get_facts(uid, limit=10)}


# ---------------------------------------------------------------------------
# Node 4: context_assemble
# ---------------------------------------------------------------------------
# Map detected emotion → recommended tag for the model
EMOTION_TO_TAG = {
    "loneliness": "sad",
    "sadness": "sad",
    "anxiety": "calm",
    "anger": "angry",
    "joy": "happy",
    "nostalgia": "calm",
    "neutral": "calm",
    "surprise": "surprise",
    "blush": "blush",
    "excited": "excited",
}


def context_assemble(state: GraphState) -> dict:
    """Build prompt = persona + memory + history + user_input."""
    parts = [PERSONA]

    # Memory injection
    facts = state.get("memory_facts", [])
    if facts:
        parts.append("\n[覚えていること]")
        for f in facts:
            parts.append(f"- {f}")

    sys = "\n".join(parts)

    # Messages: system + recent history + user_input
    msgs = [{"role": "system", "content": sys}]
    hist = state.get("messages", [])
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

    return {"system_prompt": sys, "messages": msgs}


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

    temperature = params["temperature"]

    try:
        r = _model.chat.completions.create(
            model="arrowcanaria-8b",
            messages=msgs,
            temperature=temperature,
            max_tokens=250,
            extra_body={"repetition_penalty": params["repetition_penalty"]},
        )
        return {"response": r.choices[0].message.content or "", "temperature": temperature}
    except Exception as e:
        return {"response": "", "error": str(e), "temperature": temperature}


# ---------------------------------------------------------------------------
# Node 6: output_guard + translate + memory_update (combined for simplicity)
# ---------------------------------------------------------------------------
FALLBACKS = [
    "あら、ごめんなさいね。ちょっと聞こえなかったみたい。もう一度お願いできる？",
    "うん、そうなんだね。あなたの話、もっと聞きたいな。どんなことでもいいから教えて。",
    "ああ、なんだかうまく言葉が出てこなかったわ。もう少しあなたのことを聞かせてくれる？",
]


def clean_and_translate_and_remember(state: GraphState) -> dict:
    """Output guard → [translate ‖ memory update] in parallel."""
    import random
    raw = state.get("response") or ""

    # --- Output guard: clean up, no tag manipulation ---
    cleaned = re.sub(r"（[^）]*）", "", raw)
    cleaned = re.sub(r"\([^)]*\)", "", cleaned)
    # Strip any lingering bracket tags (model shouldn't output them, but safety net)
    cleaned = re.sub(r"\[[^\]]{1,20}\]\s*", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if len(cleaned) < 15:
        cleaned = random.choice(FALLBACKS)
    if len(cleaned) > 400:
        # Find last sentence boundary within 400 chars
        truncated = cleaned[:400]
        last_boundary = max(truncated.rfind("。"), truncated.rfind("？"), truncated.rfind("！"))
        if last_boundary > 200:
            cleaned = truncated[:last_boundary + 1]
        else:
            cleaned = truncated + "…"

    uid = state.get("user_id", "anonymous")
    user_input = state.get("user_input", "")
    translation = ""
    new_facts = []

    # Run translate + fact extraction in parallel (saves ~1-2s)
    def _translate():
        if not _ds or not cleaned.strip() or len(cleaned) < 15:
            return ""
        try:
            r = _ds.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": f"将以下日语直接翻译成中文：\n{cleaned}"}],
                temperature=0.3, max_tokens=200, timeout=8,
            )
            return r.choices[0].message.content.strip()
        except Exception:
            return ""

    def _extract_facts():
        if uid == "anonymous" or not _ds or len(user_input) < 15:
            return []
        try:
            r = _ds.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": (
                    "Extract personal facts about the user from this message. "
                    "Facts = name, age, family, interests, health, preferences, experiences. "
                    "Output JSON array of strings, or [] if none. Only JSON, no explanation.\n\n"
                    f"User: {user_input}\nAssistant: {cleaned}"
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
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_t = pool.submit(_translate)
            fut_f = pool.submit(_extract_facts)
            translation = fut_t.result()
            new_facts = fut_f.result()

    return {
        "response": raw,
        "response_cleaned": cleaned,
        "translation": translation,
    }
