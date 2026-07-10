# 养老陪伴 AI — API 对接文档

> **服务地址**：`http://<host>:8016/v1`  
> **协议**：OpenAI Chat Completions 兼容  
> **模型名**：`companion-cn`  
> **底层模型**：Qwen 3 30B（vLLM 部署）  
> **最后更新**：2026-07-10

---

## 目录

- [1. 快速开始](#1-快速开始)
- [2. 对话接口（OpenAI 兼容）](#2-对话接口openai-兼容)
  - [2.1 非流式](#21-非流式)
  - [2.2 流式（SSE）](#22-流式sse)
- [3. 模型列表](#3-模型列表)
- [4. 扩展参数](#4-扩展参数)
- [5. 方言选择](#5-方言选择)
- [6. 内部接口（可选）](#6-内部接口可选)
- [7. 完整示例](#7-完整示例)

---

## 1. 快速开始

**改一行 `base_url`，OpenAI SDK 直接接入：**

```python
from openai import OpenAI

# 原来调 OpenAI：
# client = OpenAI(api_key="sk-xxx")

# 现在调养老陪伴：
client = OpenAI(base_url="http://localhost:8016/v1", api_key="x")

response = client.chat.completions.create(
    model="companion-cn",
    messages=[{"role": "user", "content": "最近天冷了，也没个人说说话"}],
    extra_body={"dialect": "northern"},  # 可选：方言风格
)
print(response.choices[0].message.content)
```

---

## 2. 对话接口（OpenAI 兼容）

### 2.1 非流式

```
POST /v1/chat/completions
```

**请求**（OpenAI 标准格式 + 扩展字段）：

```json
{
  "model": "companion-cn",
  "messages": [
    {"role": "user", "content": "最近腰老是疼，晚上翻身都费劲"}
  ],
  "stream": false,
  "user": "old_zhang",
  "session_id": "sess_001",
  "dialect": "dongbei"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model` | string | 是 | 固定 `"companion-cn"` |
| `messages` | array | 是 | 标准 OpenAI 格式 |
| `stream` | bool | 否 | 默认 `false` |
| `user` | string | 否 | 用户标识，对应 `user_id`，用于记忆关联 |
| `session_id` | string | 否 | 会话标识，不传自动生成。同一 ID 共享对话历史和话题追踪 |
| `dialect` | string | 否 | 口语风格，默认 `"northern"`，见[方言选择](#5-方言选择) |

**响应**（OpenAI 标准格式）：

```json
{
  "id": "chatcmpl-f521f3b668d1",
  "object": "chat.completion",
  "created": 1783654350,
  "model": "companion-cn",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "哎哟，晚上翻个身都费劲，那得多遭罪啊。是最近累着了还是老毛病？睡得不好，第二天更扛不住。"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 58,
    "total_tokens": 58
  }
}
```

### 2.2 流式（SSE）

```
POST /v1/chat/completions/stream
```

请求体同上，设 `"stream": true`。响应为 SSE 逐 token 推送：

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"哎"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"哟"},"finish_reason":null}]}

...

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

---

## 3. 模型列表

```
GET /v1/models
```

```json
{
  "object": "list",
  "data": [
    {
      "id": "companion-cn",
      "object": "model",
      "created": 1750000000,
      "owned_by": "companion"
    }
  ]
}
```

---

## 4. 扩展参数

OpenAI 标准之外，本服务支持以下扩展参数（放在请求体顶层或 `extra_body` 中）：

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `dialect` | string | `"northern"` | 口语风格 |
| `session_id` | string | 自动生成 | 会话 ID，多轮对话保持一致 |
| `user` | string | `"anonymous"` | 映射到 `user_id`，启用记忆功能 |

**安全说明**：当用户输入包含自杀、虐待、急病等关键词时，服务会自动拦截并返回预设安全话术，`finish_reason` 仍为 `"stop"`。

---

## 5. 方言选择

| 值 | 风格 | 示例词 |
|----|------|--------|
| `northern` | 北方口语 | 可真是、那可不、可不是嘛、得嘞 |
| `dongbei` | 东北口语 | 可拉倒吧、那可不咋的、老好了、嗯呐 |
| `sichuan` | 四川口语 | 要得、巴适、咋子嘛、硬是、对头 |
| `jiangzhe` | 江浙口语 | 好的呀、是哦、蛮好、对哇、好的啦 |
| `standard` | 通用口语 | 挺好的、是吧、对呀、嗯嗯 |

---

## 6. 完整示例

### Python（OpenAI SDK）

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8016/v1", api_key="x")

# 第一轮
r1 = client.chat.completions.create(
    model="companion-cn",
    messages=[{"role": "user", "content": "最近腰老是疼"}],
    extra_body={"dialect": "sichuan", "session_id": "my_session"},
)
reply1 = r1.choices[0].message.content
print(f"AI: {reply1}")

# 第二轮（同一 session，自动关联上下文和话题追踪）
r2 = client.chat.completions.create(
    model="companion-cn",
    messages=[
        {"role": "user", "content": "最近腰老是疼"},
        {"role": "assistant", "content": reply1},
        {"role": "user", "content": "老毛病了，十几年了"},
    ],
    extra_body={"dialect": "sichuan", "session_id": "my_session"},
)
print(f"AI: {r2.choices[0].message.content}")
```

### Python（requests）

```python
import requests

r = requests.post("http://localhost:8016/v1/chat/completions", json={
    "model": "companion-cn",
    "messages": [{"role": "user", "content": "最近腰老是疼"}],
    "dialect": "dongbei",
})
print(r.json()["choices"][0]["message"]["content"])
```

### cURL

```bash
# 非流式
curl -X POST http://localhost:8016/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"companion-cn","messages":[{"role":"user","content":"最近腰老是疼"}],"dialect":"dongbei"}'

# 流式
curl -X POST http://localhost:8016/v1/chat/completions/stream \
  -H "Content-Type: application/json" \
  -d '{"model":"companion-cn","messages":[{"role":"user","content":"今天孙子来看我了！"}],"dialect":"sichuan","stream":true}'

# 模型列表
curl http://localhost:8016/v1/models
```

---

## 7. 内部接口（可选）

以下接口保留自原 LangGraph 架构，用于调试和管理：

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/v1/langgraph/chat` | 内部非流式对话（非 OpenAI 格式） |
| POST | `/v1/langgraph/chat/stream` | 内部流式对话 |
| GET | `/v1/langgraph/profile/{user_id}` | 用户记忆档案 |
| GET | `/v1/langgraph/dialects` | 方言列表 |
| POST | `/v1/langgraph/log` | 保存对话日志 |
| GET | `/v1/langgraph/logs` | 会话列表 |
| GET | `/v1/langgraph/logs/{id}` | 会话详情 |
| DELETE | `/v1/langgraph/logs/{id}` | 删除会话 |
| GET | `/v1/langgraph/health` | 健康检查 |
