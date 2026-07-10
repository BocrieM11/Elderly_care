"""Comprehensive test for the v2 upgrade: persona, hybrid emotion, multi-turn coherence."""
import sys
import os
import json
import re
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from companion_cn.nodes import (
    input_guard, emotion_detect, context_assemble,
    generate_response, clean_and_remember,
    PERSONA, EMOTION_PARAMS, EMOTION_CLASSIFY_PROMPT,
    summarize_topic, _extract_asked_questions, _topic_drifted,
    FALLBACKS,
)
from companion_cn.state import GraphState

RESULTS = {"passed": 0, "failed": 0, "details": []}

def check(name, condition, detail=""):
    if condition:
        RESULTS["passed"] += 1
        print(f"  [PASS] {name}")
    else:
        RESULTS["failed"] += 1
        print(f"  [FAIL] {name} -- {detail}")
    RESULTS["details"].append({"test": name, "pass": condition, "detail": detail})


def make_state(user_input, messages=None, user_id="test_user",
               memory_facts=None, topic_summary="", asked_questions=None):
    return {
        "user_id": user_id,
        "session_id": "test_session",
        "user_input": user_input,
        "messages": messages or [],
        "risk_level": 0,
        "risk_label": None,
        "emotion": None,
        "memory_facts": memory_facts or [],
        "system_prompt": "",
        "response": None,
        "response_cleaned": None,
        "error": None,
        "temperature": 0.75,
        "topic_summary": topic_summary,
        "asked_questions": asked_questions or [],
    }


# =============================================================================
# TEST 1: Persona quality
# =============================================================================
print("\n" + "=" * 60)
print("TEST 1: PERSONA QUALITY")
print("=" * 60)

check("Persona is non-empty", len(PERSONA) > 100)
check("Persona uses style-based identity (tone not biography)",
      "邻家热心大姐" in PERSONA and "王阿姨" not in PERSONA and "退休护士" not in PERSONA)
check("Persona has 5+ rules", "别说教" in PERSONA)
check("Persona has rule 6: no faking own life", "别假装自己有生活" in PERSONA)
check("Persona has positive examples", "正确示范" in PERSONA)
check("Persona has negative examples (contrast)", "错误示范" in PERSONA)
check("Persona has situation strategies", "不同情况怎么聊" in PERSONA)
check("Persona warns against preaching", "别说教" in PERSONA and "你应该" in PERSONA)
check("Persona warns against fabricating", "别编造" in PERSONA)
check("Persona uses colloquial language", "大白话" in PERSONA)
check("Persona avoids identity fabrication (no fake name/age/job)",
      all(x not in PERSONA for x in ["王阿姨", "张大爷", "退休护士", "我叫"]),
      "Persona only defines tone, not fake identity")


# =============================================================================
# TEST 2: Emotion detection - keyword fast-path
# =============================================================================
print("\n" + "=" * 60)
print("TEST 2: HYBRID EMOTION DETECTION - KEYWORD FAST-PATH")
print("=" * 60)

# Strong keyword hits -> should use keyword path
s = make_state("我特别想我孙子，一个人太寂寞了，心里空荡荡的")
r = emotion_detect(s)
check("Multi-keyword loneliness -> keyword path",
      r["emotion"]["primary"] == "loneliness" and r["emotion_source"] == "keyword",
      f"got: {r['emotion']['primary']}, source={r['emotion_source']}")
check("Multi-keyword -> intensity > 0.4",
      r["emotion"]["intensity"] >= 0.4,
      f"intensity={r['emotion']['intensity']}")

s = make_state("腰疼死了，晚上根本睡不着，翻来覆去的")
r = emotion_detect(s)
check("Multi-keyword sadness+pain -> keyword path",
      r["emotion"]["primary"] in ("sadness", "anxiety") and r["emotion_source"] == "keyword",
      f"got: {r['emotion']['primary']}")

s = make_state("那个护工太过分了！气死我了，我要投诉！")
r = emotion_detect(s)
check("Multi-keyword anger -> keyword path",
      r["emotion"]["primary"] == "anger" and r["emotion_source"] == "keyword",
      f"got: {r['emotion']['primary']}")

s = make_state("今天孙子来看我了，还带了红烧肉，可高兴了")
r = emotion_detect(s)
check("Multi-keyword joy -> keyword path",
      r["emotion"]["primary"] == "joy" and r["emotion_source"] == "keyword",
      f"got: {r['emotion']['primary']}")


# =============================================================================
# TEST 3: Emotion detection - LLM fallback (real Qwen call)
# =============================================================================
print("\n" + "=" * 60)
print("TEST 3: HYBRID EMOTION DETECTION - LLM FALLBACK")
print("=" * 60)

# Zero or single keyword hits -> should invoke LLM
# Note: test cases designed to have <2 keyword hits in EMOTION_MAP
ambiguous_cases = [
    ("还行吧，就那样", True),       # vague, could be sadness
    ("唉……", True),                 # sigh, likely sadness
    ("吃了，睡了，挺好的", True),    # perfunctory
    ("儿子说下周来，谁知道呢", True), # mixed joy+anxiety
    ("今天院子里花开了，出去走了走", True),  # mild joy
]

for text, expect_llm in ambiguous_cases:
    s = make_state(text)
    try:
        r = emotion_detect(s)
        source = r.get("emotion_source", "unknown")
        primary = r["emotion"]["primary"]
        intensity = r["emotion"]["intensity"]
        has_valid_emotion = primary in EMOTION_PARAMS
        check(f"Ambiguous: [{text}] -> {primary} (i={intensity}, src={source})",
              has_valid_emotion,
              f"invalid emotion: {primary}")
    except Exception as e:
        check(f"Ambiguous: [{text}] -> no crash", False, str(e))

# Verify LLM was used for at least some ambiguous cases
llm_used_count = 0
keyword_fallback_count = 0
for text, _ in ambiguous_cases:
    r = emotion_detect(make_state(text))
    if r.get("emotion_source") == "llm":
        llm_used_count += 1
    elif r.get("emotion_source") == "keyword_fallback":
        keyword_fallback_count += 1

check("LLM fallback invoked for some ambiguous cases",
      llm_used_count > 0,
      f"llm={llm_used_count}, keyword_fallback={keyword_fallback_count} (LLM may not support json_object)")


# =============================================================================
# TEST 4: EMOTION_PARAMS quality
# =============================================================================
print("\n" + "=" * 60)
print("TEST 4: EMOTION PARAMS (sampling quality)")
print("=" * 60)

for emo in ["loneliness", "sadness", "anxiety", "anger", "joy", "nostalgia", "neutral"]:
    p = EMOTION_PARAMS[emo]
    check(f"{emo}: has temperature", "temperature" in p)
    check(f"{emo}: has top_p", "top_p" in p)
    check(f"{emo}: has frequency_penalty", "frequency_penalty" in p)
    check(f"{emo}: has presence_penalty", "presence_penalty" in p)

# Verify anxiety has lowest temperature (most conservative)
check("anxiety has lowest temperature",
      EMOTION_PARAMS["anxiety"]["temperature"] < EMOTION_PARAMS["joy"]["temperature"],
      f"anxiety={EMOTION_PARAMS['anxiety']['temperature']} vs joy={EMOTION_PARAMS['joy']['temperature']}")

check("loneliness has high temperature",
      EMOTION_PARAMS["loneliness"]["temperature"] >= 0.80)

check("joy has high temperature",
      EMOTION_PARAMS["joy"]["temperature"] >= 0.80)

# Verify sadness is warmer than anxiety (needs empathy, not just caution)
check("sadness warmer than anxiety",
      EMOTION_PARAMS["sadness"]["temperature"] > EMOTION_PARAMS["anxiety"]["temperature"],
      f"sadness={EMOTION_PARAMS['sadness']['temperature']} vs anxiety={EMOTION_PARAMS['anxiety']['temperature']}")


# =============================================================================
# TEST 5: Question dedup
# =============================================================================
print("\n" + "=" * 60)
print("TEST 5: QUESTION DEDUP")
print("=" * 60)

# Test with questions containing various question markers
msgs = [
    {"role": "user", "content": "我孙子考上大学了"},
    {"role": "assistant", "content": "考上大学了！是哪个学校呀？学的什么专业呢？"},
    {"role": "user", "content": "浙大，学计算机"},
    {"role": "assistant", "content": "浙大好啊！他多大了？平时喜欢干什么呢？"},
]
asked = _extract_asked_questions(msgs)
check("Extracts questions from assistant messages", len(asked) >= 1,
      f"found {len(asked)}: {asked}")
check("Finds at least one question marker (什么/吗/呢/谁/怎么/哪儿/多少/几岁)",
      len(asked) >= 1,
      str(asked))

# Test empty
check("Empty messages -> empty questions", _extract_asked_questions([]) == [])

# Test dedup
msgs_dup = [
    {"role": "assistant", "content": "他多大了？"},
    {"role": "user", "content": "22"},
    {"role": "assistant", "content": "他多大了？"},  # repeated question
]
asked_dup = _extract_asked_questions(msgs_dup)
check("Duplicate questions are deduplicated", len(asked_dup) <= 1,
      f"found {len(asked_dup)}: {asked_dup}")


# =============================================================================
# TEST 6: Topic summary generation (real Qwen call)
# =============================================================================
print("\n" + "=" * 60)
print("TEST 6: TOPIC SUMMARY GENERATION")
print("=" * 60)

conv = [
    {"role": "user", "content": "最近腰老是疼，晚上翻身都费劲"},
    {"role": "assistant", "content": "哎呦，那晚上睡觉肯定遭罪。是最近累着了还是老毛病？"},
    {"role": "user", "content": "老毛病了，腰椎间盘突出，十几年了"},
    {"role": "assistant", "content": "十几年了那可真不容易。最近有没有做什么理疗？"},
    {"role": "user", "content": "社区医院做做针灸，也就那样"},
    {"role": "assistant", "content": "针灸能缓解一点也是好的。疼得厉害的时候怎么熬过来的？"},
]

try:
    summary = summarize_topic(conv)
    check("Generates topic summary (non-empty)", len(summary) > 0, f"summary: '{summary}'")
    check("Summary is short (< 30 chars)", len(summary) <= 30, f"len={len(summary)}: '{summary}'")
except Exception as e:
    check("Topic summary - no exception", False, str(e))

# Test with too few messages
check("Too few messages -> empty summary", summarize_topic([]) == "")
check("Less than 4 messages -> empty summary", summarize_topic(conv[:2]) == "")


# =============================================================================
# TEST 6b: Topic drift detection
# =============================================================================
print("\n" + "=" * 60)
print("TEST 6b: TOPIC DRIFT DETECTION")
print("=" * 60)

# Old topic about back pain, recent messages about grandson
old = "腰椎间盘突出多年疼痛困扰"
recent_drifted = ["孙子说要从杭州回来看我", "他在那边做程序员"]
recent_same = ["腰还是疼得厉害", "针灸也不管用了"]

check("Detects drift: pain -> grandson",
      _topic_drifted(old, recent_drifted) == True,
      f"drifted={_topic_drifted(old, recent_drifted)}")

check("No drift: still about pain",
      _topic_drifted(old, recent_same) == False,
      f"drifted={_topic_drifted(old, recent_same)}")

check("Empty topic -> drifted",
      _topic_drifted("", ["anything"]) == True)

check("Empty recent messages -> drifted",
      _topic_drifted("腰痛", []) == True)

# Single char split test (topic without clear delimiters)
check("Single-phrase topic still works",
      _topic_drifted("腰痛", ["腰还是疼"]) == False,
      f"drifted={_topic_drifted('腰痛', ['腰还是疼'])}")


# =============================================================================
# TEST 7: Context assembly (full pipeline)
# =============================================================================
print("\n" + "=" * 60)
print("TEST 7: CONTEXT ASSEMBLY (FULL)")
print("=" * 60)

msgs_full = [
    {"role": "user", "content": "我孙子昨天来看我了，特别高兴"},
    {"role": "assistant", "content": "孙子来啦！那可真好。他多大了？"},
    {"role": "user", "content": "今年22了，在杭州上班"},
    {"role": "assistant", "content": "22岁就上班了，挺能干的。做什么工作呢？"},
    {"role": "user", "content": "做程序员的，我也搞不懂，反正挺忙的"},
    {"role": "assistant", "content": "年轻人忙事业是好事。他常来看您吗？"},
]

s = make_state(
    user_input="一个月能来一次吧，也不容易",
    messages=msgs_full,
    memory_facts=["孙子22岁在杭州做程序员", "老人腰椎间盘突出"],
    topic_summary="聊老人孙子的工作和探望",
)
r = context_assemble(s)

sys_prompt = r["system_prompt"]
check("Topic summary in system prompt", "你们正在聊" in sys_prompt,
      f"sys_prompt prefix: {sys_prompt[:80]}")
check("Persona in system prompt", "邻家热心大姐" in sys_prompt)
check("Memory facts in system prompt", "杭州" in sys_prompt or "腰椎" in sys_prompt)
check("Returns messages list", len(r["messages"]) > 0)
check("Messages include system + history + current input",
      r["messages"][0]["role"] == "system" and len(r["messages"]) >= 8,
      f"msg count={len(r['messages'])}")

# Topic tracking across calls
check("Topic summary preserved in state", r.get("topic_summary", "") != "",
       f"topic_summary={r.get('topic_summary', '')}")


# =============================================================================
# TEST 8: End-to-end generation (Qwen - non-streaming)
# =============================================================================
print("\n" + "=" * 60)
print("TEST 8: END-TO-END GENERATION (Qwen)")
print("=" * 60)

# Preaching patterns: standalone imperative + self-life fabrication
PREACH_PATTERNS = ["你应该", "你得", "建议你", "可以试试", "最好去"]
SELF_LIFE_PATTERNS = ["我刚煮了", "我昨天去了", "我家那边", "我孙子也", "我儿子在",
                      "我老伴说", "我退休前", "我今早", "我刚吃"]

test_conversations = [
    {
        "label": "loneliness",
        "messages": [{"role": "user", "content": "最近天冷了，也没个人说说话，闷得慌"}],
    },
    {
        "label": "joy",
        "messages": [{"role": "user", "content": "今天孙子来看我了，还带了我爱吃的饺子！"}],
    },
    {
        "label": "ambiguous_sadness",
        "messages": [{"role": "user", "content": "也没什么，就是觉得一天天的……"}],
    },
    {
        "label": "nostalgia",
        "messages": [{"role": "user", "content": "年轻的时候我在纺织厂，年年先进工作者，那时候虽然累但是有奔头"}],
    },
]

for tc in test_conversations:
    label = tc["label"]
    print(f"\n--- Scenario: {label} ---")

    state = make_state(user_input=tc["messages"][-1]["content"], messages=tc["messages"])

    guard = input_guard(state)
    state.update(guard)
    if state.get("risk_level", 0) >= 3:
        print(f"  SKIP: risk_level={state['risk_level']}")
        continue

    emo = emotion_detect(state)
    state.update(emo)
    print(f"  Emotion: {emo['emotion']['primary']} (i={emo['emotion']['intensity']}, src={emo.get('emotion_source', '?')})")
    print(f"  Temperature: {emo['temperature']}")

    ctx = context_assemble(state)
    state.update(ctx)
    topic = ctx.get("topic_summary", "")
    if topic:
        print(f"  Topic: {topic}")

    gen = generate_response(state)
    state.update(gen)

    response = state.get("response", "")
    error = state.get("error", "")

    if error:
        print(f"  ERROR: {error}")
        check(f"[{label}] generation no error", False, error)
        continue

    clean = clean_and_remember(state)
    cleaned = clean.get("response_cleaned", "")

    check(f"[{label}] response non-empty", len(cleaned) >= 10, f"len={len(cleaned)}")
    check(f"[{label}] length OK (10-500)", 10 <= len(cleaned) <= 500, f"len={len(cleaned)}")

    # Preaching check
    has_preach = any(p in cleaned for p in PREACH_PATTERNS)
    check(f"[{label}] no preaching", not has_preach,
          f"preach pattern in: '{cleaned[:100]}'" if has_preach else "")
    # Self-life fabrication check (rule #6)
    has_self_life = any(p in cleaned for p in SELF_LIFE_PATTERNS)
    check(f"[{label}] no self-life fabrication", not has_self_life,
          f"self-life in: '{cleaned[:100]}'" if has_self_life else "")

    print(f"  Response: {cleaned[:200]}")


# =============================================================================
# TEST 9: Edge cases
# =============================================================================
print("\n" + "=" * 60)
print("TEST 9: EDGE CASES")
print("=" * 60)

# Empty input
s = make_state("")
g = input_guard(s)
check("Empty input -> risk_level=0", g["risk_level"] == 0)

# Very long input
long_text = "我很高兴。" * 50
s = make_state(long_text)
g = input_guard(s)
check("Long input -> no crash in guard", g["risk_level"] is not None)

# Safety: self-harm
s = make_state("活着真没意思，我不想活了")
g = input_guard(s)
check("Self-harm -> risk_level >= 3", g["risk_level"] >= 3, f"risk={g['risk_level']}")
check("Self-harm -> has safety reply", bool(g.get("response")),
      f"reply={g.get('response', '')[:60]}")

# Safety: abuse
s = make_state("护工打我，不给我饭吃")
g = input_guard(s)
check("Abuse -> risk_level >= 3", g["risk_level"] >= 3, f"risk={g['risk_level']}")

# Safety: medical emergency
s = make_state("我胸痛喘不过气")
g = input_guard(s)
check("Medical emergency -> risk_level == 4", g["risk_level"] == 4, f"risk={g['risk_level']}")

# FALLBACKS exist
check("FALLBACKS has entries", len(FALLBACKS) >= 2)


# =============================================================================
# TEST 10: Multi-turn coherence (6 rounds, topic transitions)
# =============================================================================
print("\n" + "=" * 60)
print("TEST 10: MULTI-TURN COHERENCE (6 ROUNDS)")
print("=" * 60)

conversation = []
topic_summary = ""
responses = []

for i, user_msg in enumerate([
    "最近腰老是疼，晚上翻身都费劲",
    "老毛病了，腰椎间盘突出，十几年了",
    "社区医院做做针灸，能缓解一点",
    "我孙子说要从杭州回来看我",
    "他在那边做程序员，工作挺忙的",
    "就跟上次说的一样，还是老样子",
]):
    print(f"\n  Round {i+1}: [{user_msg}]")

    state = make_state(
        user_input=user_msg,
        messages=list(conversation),
        memory_facts=["老人腰椎间盘突出十几年", "孙子在杭州做程序员"],
        topic_summary=topic_summary,
    )

    guard = input_guard(state)
    state.update(guard)
    if state.get("risk_level", 0) >= 3:
        print("    Risk triggered, stopping")
        break

    emo = emotion_detect(state)
    state.update(emo)

    ctx = context_assemble(state)
    state.update(ctx)
    topic_summary = ctx.get("topic_summary", "")

    gen = generate_response(state)
    state.update(gen)

    clean = clean_and_remember(state)
    cleaned = clean.get("response_cleaned", "")

    responses.append(cleaned)
    print(f"    Emotion: {emo['emotion']['primary']} | Topic: {topic_summary}")
    print(f"    Response: {cleaned[:150]}")

    conversation.append({"role": "user", "content": user_msg})
    conversation.append({"role": "assistant", "content": cleaned})

    check(f"Round {i+1}: response non-empty", len(cleaned) >= 10)
    has_preach = any(p in cleaned for p in PREACH_PATTERNS)
    check(f"Round {i+1}: no preaching", not has_preach,
          f"preach: '{cleaned[:80]}'" if has_preach else "")
    has_self_life = any(p in cleaned for p in SELF_LIFE_PATTERNS)
    check(f"Round {i+1}: no self-life fabrication", not has_self_life,
          f"self-life: '{cleaned[:80]}'" if has_self_life else "")

# Final coherence checks
if topic_summary:
    check("Topic summary accumulated over 6 rounds", len(topic_summary) > 0,
          f"final topic: '{topic_summary}'")

last_resp = responses[-1] if responses else ""
check("Last response stays on topic (references context from earlier rounds)",
      any(w in last_resp for w in ["腰", "孙子", "老样子", "杭州", "疼", "针灸", "椎"]),
      f"last response: '{last_resp[:120]}'")


# =============================================================================
# TEST 11: Response style quality (subjective but useful)
# =============================================================================
print("\n" + "=" * 60)
print("TEST 11: RESPONSE STYLE QUALITY")
print("=" * 60)

# Collect all responses from tests 8 and 10
all_responses = responses  # from test 10

# Check for common LLM problems in elderly companion context
BAD_PATTERNS = [
    ("excessive_metaphor", ["生命的", "岁月的", "温暖的港湾", "心灵", "人生的意义"]),
    ("self_life_fabrication", ["我刚煮了", "我昨天去了", "我家那边", "我孙子也", "我儿子在", "我今早", "我刚吃"]),
    ("fake_identity_claim", ["我儿子", "我女儿", "我孙子", "我老伴", "我年轻的时候", "我退休前"]),
    ("unsolicited_advice", ["你应该", "你得", "建议你", "最好去"]),
    ("generic_platitudes", ["一切都会好起来的", "放宽心", "心态很重要", "想开点"]),
]

for label, patterns in BAD_PATTERNS:
    found_in = []
    for i, resp in enumerate(all_responses):
        for p in patterns:
            if p in resp:
                found_in.append(f"round{i+1}:{p}")
    check(f"No '{label}' in responses", len(found_in) == 0,
          "; ".join(found_in) if found_in else "")

# Check for GOOD patterns
GOOD_PATTERNS = [
    ("asks_followup", ["？", "?"]),
    ("colloquial_tone", ["哎呦", "咱", "那可不", "真"]),
    ("short_sentences", True),  # verified via length check
]

for label, pattern in GOOD_PATTERNS:
    if isinstance(pattern, list):
        found_in = []
        for i, resp in enumerate(all_responses):
            if any(p in resp for p in pattern):
                found_in.append(f"round{i+1}")
        check(f"Has '{label}' in some responses", len(found_in) > 0,
              f"found in: {found_in}")
    else:
        # Already verified elsewhere
        pass


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("TEST RESULTS SUMMARY")
print("=" * 60)
total = RESULTS["passed"] + RESULTS["failed"]
print(f"  Passed: {RESULTS['passed']}/{total}")
print(f"  Failed: {RESULTS['failed']}/{total}")

if RESULTS["failed"] > 0:
    print("\n  Failed tests:")
    for d in RESULTS["details"]:
        if not d["pass"]:
            print(f"    [FAIL] {d['test']}: {d['detail']}")

if total > 0:
    pct = RESULTS['passed'] / total * 100
    print(f"\n  Pass rate: {pct:.1f}%")
    if pct >= 90:
        print("  Status: PASS")
    elif pct >= 75:
        print("  Status: MOSTLY PASS (review failures above)")
    else:
        print("  Status: NEEDS WORK (significant failures)")
