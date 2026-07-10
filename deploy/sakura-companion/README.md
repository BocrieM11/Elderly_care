# さくら（Sakura）— 日语老年情绪陪伴 AI

> 80岁的AI朋友「さくら」，温暖自然地和养老院老人聊天。  
> 基于 ArrowCanaria 8B 本地推理 + LangGraph 状态图编排 + DeepSeek 翻译/记忆提取。

---

## 架构总览

```
浏览器 (:8015)
    │
    ▼
┌─────────────────────────────────────────────────┐
│  companion (FastAPI :8015)                       │
│  ├─ /                                              │
│  ├─ /v1/langgraph/chat        非流式               │
│  ├─ /v1/langgraph/chat/stream 流式 (SSE)           │
│  ├─ /v1/langgraph/translate   翻译 (DeepSeek)      │
│  ├─ /v1/langgraph/log         对话日志 CRUD         │
│  └─ /v1/langgraph/profile/:id 用户画像             │
│                                                    │
│  LangGraph StateGraph:                             │
│  input_guard → emotion_detect ‖ memory_retrieve    │
│       → context_assemble → generate_response       │
│       → clean + tag_correct + translate + remember               │
└────────────────────┬────────────────────────────┘
                     │ OpenAI-compatible API
                     ▼
┌─────────────────────────────────────────────────┐
│  arrowcanaria_server (FastAPI :8014)              │
│  ├─ /v1/chat/completions   推理 (stream/non)      │
│  └─ /v1/models              模型列表              │
│                                                    │
│  ArrowCanaria-Llama-8B-SFT-v0.1                   │
│  4-bit BitsAndBytes量化, SDPA attention            │
│  RTX 4060, ~4.5GB VRAM                            │
└─────────────────────────────────────────────────┘
```

两个服务分离：:8014 纯模型推理（无业务逻辑），:8015 负责所有编排、翻译、日志和前端。

---

## 快速启动

### 1. 环境变量

```bash
# .env
DEEPSEEK_API_KEY=sk-xxx   # 翻译 + 事实提取（可选，不设则跳过）
```

### 2. 安装依赖

```bash
pip install fastapi uvicorn openai langgraph pydantic python-dotenv
# ArrowCanaria 推理服务还需要:
pip install torch transformers accelerate bitsandbytes
```

### 3. 启动推理服务

```bash
cd arrowcanaria_server
python server.py   # → http://0.0.0.0:8014
```

### 4. 启动陪伴服务

```bash
python -m companion.main   # → http://0.0.0.0:8015
```

浏览器打开 `http://127.0.0.1:8015` 即可使用。

---

## LangGraph 节点说明

| 节点 | 功能 | 耗时 |
|------|------|------|
| `input_guard` | 安全扫描：自杀/虐待/急病关键词检测 → 风险≥3直接返回安全回复 | <1ms |
| `emotion_detect` | 关键词匹配8种情绪（loneliness/sadness/anxiety/anger/joy/nostalgia/surprise/blush/excited）→ 选推理参数+标签引导 | <1ms |
| `memory_retrieve` | SQLite 读取用户历史事实（最近10条） | <5ms |
| `context_assemble` | 组装 PERSONA + 记忆 + 历史消息 → 最终 prompt | <1ms |
| `generate_response` | 调用 ArrowCanaria 8B 生成回复 | 1-3s |
| `quality_check` | 启发式质检：过短/冷漠/跑题/重复 → 触发重试（最多2次） | <1ms |
| `clean_and_translate_and_remember` | 输出清洗 + 并行调用 DeepSeek 翻译 + 事实提取存储 | 1-2s |

### 图结构（含并行 + 循环）

```
                    ┌─────────────────┐
                    │  input_guard    │
                    └───────┬─────────┘
                            │ risk<3
              ┌─────────────┼─────────────┐
              ▼             │             ▼
     emotion_detect    memory_retrieve    END (risk≥3)
              │             │
              └──────┬──────┘
                     ▼
            context_assemble
                     │
                     ▼
            generate_response ◄──────────┐
                     │                    │
                     ▼                    │ retry
             quality_check ──────────────┘
                     │ pass / max retries
                     ▼
        clean_and_translate_and_remember
                     │
                     ▼
                    END
```

---

## 情绪 × 推理参数

不同情绪使用不同的 temperature 和 repetition_penalty，不往 prompt 塞指令：

| 情绪 | temperature | repetition_penalty | 策略 |
|------|------------|-------------------|------|
| loneliness | 0.85 | 1.15 | 更主动、更多样 |
| sadness | 0.70 | 1.10 | 更稳、更安全 |
| anxiety | 0.65 | 1.10 | 最稳、避免失言 |
| anger | 0.70 | 1.12 | 稳中带安抚 |
| joy | 0.85 | 1.12 | 活泼回应 |
| nostalgia | 0.78 | 1.12 | 温暖共鸣 |
| neutral | 0.75 | 1.12 | 默认 |

---

## PERSONA 设计

さくら的人设通过 few-shot 示例 + 三条核心规则定义（`companion/nodes.py:20-36`）：

- 必须触及对方说的具体内容
- 必须用提问结尾，让对方继续说话
- 用「私も〜」代替「そうですね」表达共情

附带好坏对比示例，8B 模型直接模仿示例的句式和温度。

---

## API 端点

### POST /v1/langgraph/chat
非流式对话，返回完整 ChatResponse。

```json
{
  "user_id": "user001",
  "session_id": "",
  "message": "最近腰が痛くてね…"
}
```

### POST /v1/langgraph/chat/stream
流式对话（SSE），按句子边界分段发送，150ms 间隔。

```
data: {"token": "腰が痛いのは辛いね。", "done": false}
data: {"token": "病院には行った？", "done": false}
data: {"token": "", "done": true, "translation": "...", "emotion": {...}}
```

### GET /v1/langgraph/profile/{user_id}
获取用户画像和历史事实。

### POST /v1/langgraph/translate
日语→中文翻译（DeepSeek）。

### POST/GET/DELETE /v1/langgraph/log[s]
对话日志 CRUD。

---

## 项目结构

```
养老agent/
├── companion/                    # 陪伴服务 (:8015)
│   ├── main.py                   # FastAPI 入口 + 前端托管
│   ├── api.py                    # REST 端点 (chat/stream/translate/log/profile)
│   ├── graph.py                  # LangGraph 图定义 (并行fan-out + 质检循环)
│   ├── state.py                  # GraphState TypedDict
│   ├── nodes.py                  # 6个节点 + 情绪检测 + PERSONA
│   ├── safety.py                 # 安全扫描 (自杀/虐待/急病)
│   ├── memory.py                 # SQLite 用户画像 & 事实存储
│   └── config.py                 # 常量 (端口/URL/API key)
│
├── arrowcanaria_server/          # 推理服务 (:8014)
│   ├── server.py                 # ArrowCanaria 8B 模型加载 + OpenAI兼容API
│   └── static/index.html         # 前端聊天界面
│
├── training_data/                # 训练数据 (ShareGPT JSONL)
│   ├── train.jsonl               # 16,125条
│   ├── val.jsonl                 # 2,016条
│   └── test.jsonl                # 2,019条
│
├── training_data_cn/             # 中文训练数据
├── dialogs/                      # 手工对话设计稿 (V1-V4, 多语言)
└── docs/                         # 调研报告 & 修改日志
```

---

## 训练数据集

用于 Qwen2.5-7B-Instruct LoRA 微调的日语老年关怀对话数据。

| 指标 | 数值 |
|------|------|
| 总条数 | 20,160 |
| 训练/验证/测试 | 16,125 / 2,016 / 2,019 |
| 平均轮次 | 5.8 轮/对话 |
| 真实数据占比 | 95.7% |
| 话题数 | 14 |
| 意图类型 | 8 |
| AI去重回复 | 20,745 种 |

数据来源：YouTube 老年采访的 GPU-Whisper 转写 + 专业 ASR（manifest.csv），覆盖年金、健康、家庭、孤独、怀旧等14个话题。

### LLaMA-Factory 微调

```bash
llamafactory-cli train \
  --dataset training_data/train.jsonl \
  --val_dataset training_data/val.jsonl \
  --format sharegpt \
  --model_name_or_path Qwen2.5-7B-Instruct \
  --lora_rank 16 \
  --lora_alpha 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir ./output/elderly-care-lora
```

详细数据说明见 [CHANGELOG.md](CHANGELOG.md) 和 [training_data/DATACARD.md](training_data/DATACARD.md)。

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 推理模型 | ArrowCanaria-Llama-8B-SFT-v0.1 (4-bit QLoRA) |
| 推理引擎 | PyTorch + Transformers + SDPA |
| 编排框架 | LangGraph (StateGraph + checkpoint) |
| Web 框架 | FastAPI + Uvicorn |
| 翻译/提取 | DeepSeek API (并行 ThreadPoolExecutor) |
| 存储 | SQLite (对话日志 + 用户画像 + 事实记忆) |
| 前端 | 原生 HTML/JS (SSE via XHR readyState=3) |
| 训练数据格式 | ShareGPT JSONL |
| 微调框架 | LLaMA-Factory (LoRA) |

---

> 2026年7月 · [BocrieM11/Query-data](https://github.com/BocrieM11/Query-data)
