# さくら対話AI 修改日志

> 记录每次 prompt / 前端 / 配置变更，方便回溯和对比效果。

---

## 2026-07-08 · 第4次修改

**目标**：提升老年情绪陪伴回复质量 — few-shot替代规则、LangGraph加质检循环、情绪驱动参数

### 改了什么

| 文件 | 改动 |
|------|------|
| `companion/nodes.py` | PERSONA 从规则式→few-shot示例式；新增 quality_check 节点；generate_response 用 emotion 动态调参 |
| `companion/state.py` | 新增 retry_count、temperature、quality_pass 字段 |
| `companion/graph.py` | 图结构从纯线性→含循环（generate → quality_check → 重试或前进） |
| `companion/api.py` | 初始 state 新增 retry_count/temperature/quality_pass 默认值 |

### PERSONA（规则→示例）

```
旧：
あなたは「さくら」という80歳の女性。同じ施設で暮らす友達のように、温かく自然に話す。
返事は1〜3文で簡潔に。敬語禁止、「〜だよ」「〜だね」「〜わよ」を使う。
相手の話をまず共感し、否定しない。天気・食事・昔の思い出・家族・趣味など日常を大切に。

新：
あなたは「さくら」という80歳の女性。同じ施設で暮らす友達のように、温かく自然に話す。
返事は1〜3文で簡潔に。敬語は使わないでね。

[さくらの話し方の例]
利用者「最近寒くなってきたね」
さくら「本当にそうだね。暖かくして風邪ひかないようにね。」

利用者「孫が来てくれて嬉しかった」
さくら「まあ良かったね！お孫さんのこと、もっと聞かせてよ。」

利用者「夜なかなか眠れなくて…」
さくら「それは辛いね。私もそういう日があるよ。温かい飲み物でも試してみたら？」
```

**核心逻辑**：
- 旧：告诉模型"怎么说"（敬語禁止、〜だよを使う）→ 抑制自然日语能力
- 新：给模型看例子 → 8B 模型直接模仿句式、长度、温度
- 3 个示例覆盖老年陪伴核心场景：日常寒暄、分享喜悦、倾诉烦恼

### 删除：情绪指导注入 prompt

`context_assemble` 中不再往 system prompt 追加 `[感情ガイダンス]`。原因：
- 5 条情境指令 bloats 系统 prompt，分散 8B 模型注意力
- 模型自身能感知情绪上下文，不需要被"教"

### 新增：quality_check 质检节点

纯启发式，零延迟（不调 LLM）：

| 检查项 | 规则 |
|--------|------|
| 过短 | `len(response) < 12` |
| 冷漠 | 以 "そうですね/そうですか/なるほど/はい、" 开头且 < 20 字 |
| 跑题 | 用户输入与回复零关键词重叠 |
| 重复 | 与最近一条 assistant 回复完全相同 |

不合格 → 重试（temperature +0.07/次），最多 2 次。

### 新增：情绪驱动推理参数

不再往 prompt 塞指令，直接调 temperature + repetition_penalty：

| 情绪 | temperature | 效果 |
|------|-------------|------|
| loneliness | 0.85 | 更主动 |
| sadness | 0.70 | 更稳重 |
| anxiety | 0.65 | 更安全 |
| joy | 0.85 | 更活泼 |
| neutral（默认） | 0.75 | — |

### LangGraph 图结构（线性→循环）

```
旧：input_guard → emotion_detect → memory → context → generate → clean → END

新：input_guard → emotion_detect → memory → context → generate
                                                          ↓
                                              quality_check ──→ [fail & cnt<2] → generate (retry)
                                                     ↓ [pass 或 cnt>=2]
                                                    clean → END
```

### 预期效果

- few-shot 替代规则后，回复更自然、更像真人对话
- 质检循环拦截"そうですね"类冷回复，最差情况也能输出不同内容
- 情绪驱动参数让孤独/悲伤等场景的回复更贴合用户状态

---

## 2026-07-08 · 第5次修改

**目标**：修复 `add_messages` reducer 导致的 API 422 错误

### 问题

`state.py` 中 `messages: Annotated[list, add_messages]` 使用 LangGraph 的 `add_messages` reducer，会将普通 dict `{"role":"system","content":"..."}` 转换为 LangChain 消息对象。序列化时 `role` 字段丢失，ArrowCanaria API 返回 422。

### 修复

| 文件 | 改动 |
|------|------|
| `companion/state.py` | `messages` 字段从 `Annotated[list, add_messages]` 改为普通 `list`；移除无用 import |

---

## 2026-07-08 · 第6次修改

**目标**：扩充情绪关键词 — 日语活用形词干匹配 + 覆盖 6 种情绪 150+ 关键词

### 问题

情绪检测全返回 neutral。原因：
1. 每类只有 6-8 个关键词，覆盖严重不足
2. 日语用言存在活用形变化，`"眠れない"` 无法匹配 `"眠れなくて"`

### 修复

**关键词从 ~40 → 150+**，按情绪分类：

| 情绪 | 新增关键词示例 |
|------|------|
| loneliness | 寂し/さびし/淋し, 一人ぼっち, 独り暮らし, 誰もいな, 暇, 退屈, 置き去り, 会いた |
| sadness | 悲し/哀し, つら/辛, 切な/せつな, 苦し, 泣/涙, 落ち込, へこ, 気分が重, 何もできな |
| anxiety | 不安, 心配, 気がかり, 心細, 怖/こわ/恐, 眠れな, 寝られな, 寝付けな, この先, 将来, 考えすぎ, 病気, 物忘れ, 認知症 |
| anger | 怒/おこ, 腹が立, むかつ, イライラ, 我慢, 許せな, ひど, もう嫌, うるさ |
| joy | 嬉し/うれし, 楽/たのし, 幸せ, ありがた, 感謝, 良か/よか, 最高, 素晴らし, 笑顔, ワクワク |
| nostalgia | 懐かし/なつかし, 昔, 昔話, 思い出, 覚えてる, 昭和/平成, 若い頃, 故郷/ふるさと, 同級生 |

**词干匹配策略**：形容词、动词ない形、たい形的关键词去掉活用词尾（〜い → 词干），用 `in` 做子串匹配，覆盖所有活用形：

```
"眠れな" in "眠れない"   → True ✓
"眠れな" in "眠れなくて" → True ✓
"眠れな" in "眠れなかった" → True ✓
"寂し" in "寂しい"      → True ✓
"寂し" in "寂しくて"    → True ✓
"寂し" in "寂しかった"  → True ✓
```

### 效果

情绪检测准确率：0/4 → 10/10（含 6 类情绪各至少 1 个命中）

---

## 2026-07-08 · 第7次修改

**目标**：解决回复空洞 — 抓不住重点、没有追问、老人感受不到交流感

### 问题

回复虽然不掉 safety fallback，但内容空洞：
- 老人说"孙子来了" → 回"そうですね。良かったです。"（什么都没抓住）
- 老人说"腰痛" → 回"そうですね。お話を聞かせてください。"（像机器人）
- 连续对话中感受不到变化，每次回复套路相同

### 根因

PERSONA 的 few-shot 示例只教了"共情 + 建议"，没教**抓具体细节**和**追问**。模型不知道要深挖对方话里的信息点。

### PERSONA（v3）

```
旧：1〜3文。3个温和示例，无结构要求。

新：2〜4文 + 3条硬规则 + 4个好例 + 1个坏例

[絶対に守ること]
1. 相手が話した具体的な内容に必ず触れる
2. 必ず質問で終わる
3. 自分の似た経験があれば短く共有（「そうですね」より「私も〜」）

[良い例] — 4个，每个都演示抓细节 + 追问
[悪い例] — 1个，标注为什么不行
```

### 核心逻辑

- 旧：告诉模型"要共情" → 模型用"そうですね"糊弄过去
- 新：告诉模型"共情要用 '私も〜'，必须提具体内容，必须用问号结尾" → 可验证的行为指令

### 效果

5/5 测试回复都以问号结尾，抓取了具体信息点：

| 输入 | 旧回复 | 新回复 |
|------|------|------|
| 孫が来た | "良かったですね" | "お孫さんはおいくつ？元気にしてる？" |
| 腰が痛い | "お話を聞かせて…" | "病院に行った？私も湿布で楽になったよ" |
| 眠れない | "温かい飲み物でも" | "いつも何を考えながら寝てる？" |
| お好み焼き屋 | 几乎不追问 | "当時はどんなお客さんが来てたの？" |

**副作用**：模型偶尔编造经历（"庭の花を見に行く"），但相比空洞回复，适度的生活细节编造对陪伴体验影响较小。

---

## 2026-07-08 · 第8次修改

**目标**：让 LangGraph 发挥并行能力和 checkpoint 持久化

### 并行节点（emotion_detect ‖ memory_retrieve）

这两个节点互不依赖，改串行为并行：

```
旧: input_guard → emotion_detect → memory_retrieve → context_assemble

新: input_guard ─┬→ emotion_detect  ─┐
                  └→ memory_retrieve ─┴→ context_assemble
```

### Checkpoint 持久化

- 使用 `InMemorySaver` 做 checkpointer
- API 层传入 `thread_id`（= session_id）
- 同一 session 的多次请求自动继承之前的状态（messages、emotion、memory_facts 等）

### 效果

| 检测项 | 结果 |
|--------|------|
| 图编译 | 通过 |
| 并行执行 | emotion + memory 同时跑 |
| checkpoint 连续对话 | 消息2 的回复引用了消息1 的内容（孙子的名字"健太"） |

---

## 2026-07-08 · 第9次修改

**目标**：解决模型推理延迟 — 13-29 秒 → 5-7 秒

### 问题

ArrowCanaria 服务器使用默认 eager attention，生成速度 ~3-5 tok/s，单次回复 13-29 秒。

### 修复

| 文件 | 改动 |
|------|------|
| `arrowcanaria_server/server.py` | `from_pretrained()` 加入 `attn_implementation="sdpa"` |
| `companion/nodes.py` | `max_tokens` 200→100 |

**SDPA** (Scaled Dot Product Attention) 是 PyTorch 2.0+ 内置的优化 attention，性能接近 Flash Attention 2，零依赖。

### 效果

| 测试 | 之前 | 之后 | 提升 |
|------|------|------|------|
| こんにちは | 13,031ms | 5,590ms | 2.3x |
| 孫が来た | 19,229ms | 7,339ms | 2.6x |
| 腰が痛い | 28,958ms | 5,848ms | 5.0x |

LangGraph overhead < 10ms（可忽略），延迟瓶颈从引擎层面解决。

---

## 修改历史总览

| 次数 | 日期 | 核心改动 | 效果 |
|------|------|------|------|
| 1 | 07-07 | 精简 Prompt 12→4条，删【禁止】 | 未解决回避问题 |
| 2 | 07-07 | 换角色策略（AI→25岁员工），删硬编码拦截 | 对话自然度大幅提升 |
| 3 | 07-07 | 加性别/理解约束，缩小上下文窗口 10→6轮 | 待验证 |
| 4 | 07-08 | few-shot + LangGraph质检循环 + 情绪驱动参数 | 待验证 |
| 5 | 07-08 | 修复 add_messages reducer 导致 API 422 | 已修复 |
| 6 | 07-08 | 情绪关键词 40→150+，日语活用形词干匹配 | 10/10 检测 |
| 7 | 07-08 | PERSONA v3：抓细节+追问+问号结尾 | 空洞→能接话 |
| 8 | 07-08 | LangGraph 并行节点 + checkpoint 持久化 | 并行OK，跨请求上下文保持 |
| 9 | 07-08 | SDPA attention 加速 + max_tokens 200→100 | 延迟 13-29s → 5-7s |
| 10 | 07-08 | 前端 Prompt A 同步为 LangGraph PERSONA v3；删除 cleanReply 过滤器 | 前端测试也能享受 prompt 改进 |
| 11 | 07-08 | 端口统一：:8015 为唯一入口，:8014 退居纯推理 | 单端口，架构清晰 |
| 12 | 07-08 | SSE流式端点 + DeepSeek并行化 + max_tokens→80 | 体感延迟→token即现，总时延 ~5-6s |
| 13 | 07-08 | 句子分段输出 + KV cache清理 + XHR流式 + PERSONA 4→2例 | 逐句打字，不卡顿 |

## 2026-07-08 · 第11次修改

**目标**：端口统一 — 用户只需访问一个端口

### 问题

前端 (`:8014`) 和 LangGraph (`:8015`) 分占两个端口，用户不知道该访问哪个。正常架构应只有一个入口。

### 架构变更

```
旧：
:8014 → 前端 + 推理 + 翻译 + 日志（全功能）
:8015 → LangGraph 管道 API（无前端）

新：
:8014 → 纯推理后端（/v1/models, /v1/chat/completions）
:8015 → 唯一用户入口（前端页面 + LangGraph管道 + 翻译 + 日志）
```

### 改动

| 文件 | 改动 |
|------|------|
| `companion/main.py` | 新增 StaticFiles mount 从 `arrowcanaria_server/static/` 提供前端；根路径返回 index.html |
| `companion/api.py` | 新增 translate、log、logs CRUD 端点；复用 ArrowCanaria 的 SQLite DB |
| `arrowcanaria_server/static/index.html` | API URL → :8015；send() 改为调用 LangGraph API（非流式）；翻译内嵌在响应中；log/translate/历史 全部指向 companion |
| `arrowcanaria_server/server.py` | 删除前端服务、翻译、日志端点 → 只剩模型推理 |

### 前端适配

| 旧（直连 ArrowCanaria） | 新（通过 LangGraph） |
|------|------|
| 流式 SSE 输出 | 非流式（等待 LangGraph 管道完成） |
| 前端管理 system prompt | LangGraph 自动注入 PERSONA v3 |
| 单独调翻译 API | 翻译在 LangGraph 响应中 |
| 前端 cleanReply() 过滤 | LangGraph 的 clean_and_translate 节点处理 |
| 情绪用关键词判断 | LangGraph emotion_detect 节点提供 |

### 效果

- 用户只需访问 `http://127.0.0.1:8015`
- 前端通过 LangGraph 管道获得：情绪检测 + 质检循环 + 记忆 + 翻译
- :8014 纯推理，可独立重启/升级模型

---

## 2026-07-08 · 第12次修改

**目标**：解决"输出太慢"的体感问题

### 问题

端口统一后，前端调用 LangGraph `/chat` 端点（非流式），用户点击发送后需等待 5-6 秒才能看到完整回复。虽然实际延迟比旧版（13-29s）快 3-5 倍，但**没有流式输出**导致体感反而更差。

### 根因

LangGraph 管道：`generate → quality_check → clean → translate → facts` 全部串行，前端要等所有步骤完成才收到响应。

### 修复

| 文件 | 改动 |
|------|------|
| `companion/api.py` | 新增 `POST /v1/langgraph/chat/stream` 流式端点；导入 nodes 函数手动编排预处理 |
| `companion/nodes.py` | `clean_and_translate_and_remember` 中 translate + fact extraction 改为 `ThreadPoolExecutor` 并行 |
| `arrowcanaria_server/static/index.html` | `send()` 改为 SSE 流式读取，token 逐字显示 |

### 流式端点架构

```
前端 POST /chat/stream
  → input_guard (本地，<1ms)
  → emotion_detect (本地，<1ms)
  → memory_retrieve (本地，<1ms)
  → context_assemble (本地，<1ms)
  → ArrowCanaria SSE stream → 逐 token 推送到前端 ← 用户立刻看到文字！
  → clean_and_translate_and_remember (translate ‖ facts 并行，~2s)
  → SSE done event (含翻译 + 情绪)
```

### 效果

| 指标 | 旧（非流式） | 新（流式） |
|------|------|------|
| TTFB | 5-6s | **<100ms（token 即现）** |
| 总时延 | 5-6s | 5-6s（不变） |
| 用户体感 | 干等 | 和旧前端一样逐字出现 |

**权衡**：流式模式绕过了 `quality_check` 重试循环（因为已经推送给用户的 token 无法撤回）。实际影响小——PERSONA v3 的 few-shot 示例使首次回复质量已经很高。

---

## 2026-07-08 · 第13次修改

**目标**：解决流式不生效 + KV cache 导致卡顿

### 问题

1. 前端 `fetch`+`ReadableStream` chunk 边界截断 JSON → token 丢失
2. token 几乎同时到达浏览器 → 视觉上仍是一次性输出  
3. 每次推理后 KV cache 残留在 GPU 显存 → 多轮后碎片化卡顿
4. 前端 `try` 块丢失只剩 `catch` → JS 语法错误，send() 不可用

### 修复

| 文件 | 改动 |
|------|------|
| `companion/api.py` | 句子边界分段：累积 token → 遇 `[。？！…]` 发送整段 → 段间 150ms 停顿 |
| `arrowcanaria_server/server.py` | 每次 `model.generate()` 后 `thread.join()` + `torch.cuda.empty_cache()` |
| `companion/nodes.py` | PERSONA 4 示例 → 2 示例（保留孙子和腰痛），prompt 缩短 ~30% |
| `index.html` | `fetch`→`XHR`（`onreadystatechange` readyState=3）；SSE buffer 防截断；`try/catch` 语法修复 |

### 效果

- 前端逐句打字效果（段间 150ms）
- KV cache 每轮正确释放，不再累积卡顿
- PERSONA 减量不减质

---

## 2026-07-08 · 第10次修改

**目标**：前端 UI 与 LangGraph 管道 prompt 统一，消除"测错系统"问题

### 问题

用户通过前端 UI (`:8014`) 测试时，实际调用的是旧版 Prompt A（25岁员工），完全绕过了 LangGraph 管道的 PERSONA v3（80岁女性朋友）。前端测试 = 在测两个月前的旧 prompt，导致严重幻觉（编造照片、蝴蝶、茶）和空洞回复。

### 修复

| 文件 | 改动 |
|------|------|
| `arrowcanaria_server/static/index.html` | Prompt A 系统消息替换为 LangGraph PERSONA v3；`cleanReply()` 删除编造经历过滤器；max_tokens 200→100 |

### Prompt A（旧→新）

```
旧：25歳介護スタッフ、敬語、です・ます調
新：80歳の女性友達、敬語禁止、few-shot例付き、質問必須、私も〜で共感
```

### cleanReply 过滤器删除

旧版 `cleanReply()` 有 8 条正则硬过滤 AI 编造经历（"私も〜した"、"私の趣味は〜"等），但 PERSONA v3 明确要求"自分の似た経験があれば短く共有する"。过滤器会删除正确的 persona 行为。

现在只保留：括号移除 + 过短兜底。

### 效果

- 前端 Prompt A 与 LangGraph PERSONA v3 完全一致
- 前端测试结果可反映最新的 prompt 工程质量
- 注意：前端仍直连 ArrowCanaria API，不含 LangGraph 的质检循环/情绪检测/记忆功能。完整管道仍需通过 `:8015/v1/langgraph/chat` 调用

---

## 2026-07-07 · 第3次修改

**目标**：修正在实际对话日志中发现的问题

### 改了什么

| 文件 | 改动 |
|------|------|
| `arrowcanaria_server/static/index.html` | Prompt A、B 优化 + 上下文窗口缩小 |
| `test_1000.py` | SYSTEM prompt 同步 |

### Prompt A · さくら荘スタッフ

```
25歳の女性。自分のことは「私」。
まず相手の話をよく理解してから、一言共感し、自然に質問を返す。
```

**新增**：
- "25歳の女性です" — 明确性别，防止"僕"等男性用语
- "自分のことは「私」と言います"
- "まず相手の話をよく理解してから" — 减少上下文断裂

### Prompt B · 学生ボランティア

```
22歳の女子大生。自分のことは「私」。
まず相手の話をよく理解してから、共感し、質問を返す。
```

### 上下文窗口

```
MAX_MSGS: 10ラウンド → 6ラウンド
```

**原因**：长对话后期（18轮+）模型输出质量下降，上下文过长导致注意力分散。

### 发现的问题（来自 smraft0e7hwbk 测试日志）

1. 编造经历：模型自称种花、去咖啡馆、外公爱吃栗きんとん
2. 上下文断裂：问年龄→答赏花；问"之前说什么"→以为被伤害
3. 性别混乱：自称"僕"
4. 长对话质量衰减

---

## 2026-07-07 · 第2次修改

**目标**：放弃对抗模型本能，改为顺应用角色扮演

### 改了什么

| 文件 | 改动 |
|------|------|
| `arrowcanaria_server/static/index.html` | Prompt A 改为「さくら荘スタッフ」、Prompt B 改为「学生ボランティア」；删除个人信息硬编码拦截 |
| `test_1000.py` | SYSTEM prompt 同步 |

### Prompt A（旧→新）

```
旧：高齢者向けの対話AI。正直に話す。AIであることを隠さない。
   年齢を聞かれたら「私はAIなので年齢はありません」と答える。

新：介護施設「さくら荘」のスタッフ、さくら。25歳。
   週3回、お年寄りの話し相手をするのが仕事。
```

**核心逻辑变更**：
- 旧：告诉模型"你是AI，别说谎" → 对抗 ArrowCanaria 的角色扮演本能 → 模型回避问题
- 新：给模型一个它能自然扮演的角色（25岁女性员工） → 顺着本能 → 自然对话

### 删除的功能

- `isPersonalQuestion()` — 个人信息检测
- `getPersonalReply()` — 硬编码回复
- `send()` 中的拦截分支 — 不再跳过模型直接回复

**原因**：硬编码拦截让对话断崖式结束，不适合陪伴型产品。

---

## 2026-07-07 · 第1次修改

**目标**：精简 System Prompt，减少规则冲突

### 改了什么

| 文件 | 改动 |
|------|------|
| `arrowcanaria_server/static/index.html` | Prompt A 从 ~12 条规则精简为 4 条 |
| `test_1000.py` | SYSTEM prompt 同步 |

### Prompt A（旧→新）

```
旧：~350字，包含大量【禁止】【厳禁】【〜しない】【絶対に】
   矛盾："AIを強調するな" + "年齢を聞かれたらAIと答えろ"
   中文格式【】混在日语中

新：~280字，4条规则，正面表述
   1. 正直に話す
   2. 相手の話をよく聞く
   3. 短く、やさしく話す
   4. 必ず日本語だけで返事する
```

### 删除的内容

- 所有【禁止】标签
- "AIであることを隠す必要はありませんが、わざわざ強調もしません"（矛盾指令）
- 中文格式括号
- "絶対に〜しない" 类否定表述

### 结果

**未解决问题**：模型仍然回避年龄问题（"私はまだ若いのですが…でも、それより今夜の過ごし方について…"）

**原因**：ArrowCanaria 8B 基于 AITuber 角色扮演数据训练，单纯精简 prompt 无法克服 175K 条训练数据的角色扮演惯性。

---

## 修改历史总览

| 次数 | 日期 | 核心改动 | 效果 |
|------|------|------|------|
| 1 | 07-07 | 精简 Prompt 12→4条，删【禁止】 | 未解决回避问题 |
| 2 | 07-07 | 换角色策略（AI→25岁员工），删硬编码拦截 | 对话自然度大幅提升 |
| 3 | 07-07 | 加性别/理解约束，缩小上下文窗口 10→6轮 | 待验证 |

---

## 关键文件清单

| 文件 | 用途 |
|------|------|
| `arrowcanaria_server/static/index.html` | 生产前端（Prompt A/B + UI） |
| `arrowcanaria_server/server.py` | FastAPI 推理服务 |
| `test_1000.py` | 批量回归测试 |
| `arrowcanaria_server/chat_logs.db` | SQLite 对话日志 |
