# 代码审查问题清单

> 审查日期：2026-07-08
> 范围：arrowcanaria_server + companion + index.html

---

## 问题 1：Prompt B 选了不生效

**位置**：[index.html:370-388](../arrowcanaria_server/static/index.html#L370) + [companion/api.py:129-220](../companion/api.py#L129-L220)

**现象**：前端下拉框可以选 A/B，用户以为切换了角色，但后端永远用 Prompt A。

**原因**：
- 前端 `/chat/stream` 请求体里**没有传** `prompt_type` 字段
- 后端 `ChatRequest` 里**没有** `prompt_type` 字段
- `context_assemble` 节点永远注入 `PERSONA`（hardcoded Prompt A 的 80 岁版本）
- 前端只在写日志时传了 `prompt_type`，但那已经晚了

**影响**：整个 Prompt B 功能形同虚设。

**修法**：
1. `ChatRequest` 加 `prompt_type: str = "A"`
2. `/chat/stream` 前端请求体加 `prompt_type: currentPrompt`
3. `context_assemble` 读取 `state["prompt_type"]`，如果是 B 就用 Prompt B 的 persona

---

## 问题 2：Prompt B 也是旧版，质量落后 A 很多

**位置**：[index.html:373-387](../arrowcanaria_server/static/index.html#L373-L387)

**现象**：Prompt B 没有 few-shot 示例，没有好/坏对比，没有 3 条具体规则。

**对比**：

| | Prompt A | Prompt B |
|---|---|---|
| Persona | 80 岁女性朋友 | 22 岁女大学生 |
| Few-shot 示例 | 4 组 | 0 |
| 好/坏对比 | 1 组 | 0 |
| 具体规则 | 3 条 | 无（只有模糊描述） |
| 禁止定型文 | ✅ | ✅ |

**影响**：切换 B 后的回复质量（如果修了问题 1 的话）会比 A 差很多。

**修法**：给 B 也加 2-3 组 few-shot + 好/坏对比，规则对齐 A。

---

## 问题 3：few-shot 里的"利用者"说话不够真实

**位置**：[index.html:357-363](../arrowcanaria_server/static/index.html#L357-L363)

**现象**：当前示例里的老人说话太干净，不符合 80 岁日本老人真实的说话方式。

```
当前：「孫が来てくれて嬉しかった」
真实：「孫が来てくれて…嬉しゅうて、涙出たわ」

当前：「最近腰が痛くてね…」
真实：「腰が…あの、痛うて。年のせいか、あちこちガタがきてなあ」
```

**参考数据**：`dialogs/V4` 已按 50,704 条真实日语语料校准（22% 填充词密度、平均 20 字句长、30% 相槌率）。

**修法**：用 V4 的老年说话风格替换 few-shot 示例中的"利用者"侧文字。さくら的回复可以不换。

---

## 问题 4：server.py 删掉的翻译和日志接口，companion 已接盘

**确认状态**：✅ 已正确迁移，无断链。

| 旧位置 (server.py 已删) | 新位置 (companion/api.py) |
|---|---|
| `POST /v1/translate` | `POST /v1/langgraph/translate` |
| `POST /v1/log` | `POST /v1/langgraph/log` |
| `GET /v1/logs` | `GET /v1/langgraph/logs` |
| `GET /v1/logs/{id}` | `GET /v1/langgraph/logs/{id}` |
| `DELETE /v1/logs/{id}` | `DELETE /v1/langgraph/logs/{id}` |

前端用了 `${API_BASE}/log` 即 `http://127.0.0.1:8015/v1/langgraph/log` ✅

---

## 问题 5：companion 流式 pre-processing 不完整

**位置**：[companion/api.py:153-167](../companion/api.py#L153-L167)

**现象**：`/chat/stream` 里手动调了 `input_guard`、`emotion_detect`、`memory_retrieve`、`context_assemble`，**跳过了 LangGraph 的 `graph.ainvoke()`**。这意味着：
- 不走 `quality_check` 重试循环
- 图的状态管理（checkpointer、thread_id）被绕过
- 非流式 `/chat`（line 75）走的是 `graph.ainvoke()`，流式和不流式的逻辑路径不一致

**影响**：流式模式下没有质检重试，可能输出不完整的回复。

**修法**：考虑在流式管道里也加简易 post-check，或者让 graph 的 quality_check 节点也支持流式场景。也可以保持现状（追求速度不 retry）但要在注释里标明。

---

## 问题 6：无意义输入防御过弱

**位置**：[companion/nodes.py:40-50](../companion/nodes.py#L40-L50) + [companion/safety.py](../companion/safety.py)

**现象**：输入「には」→ 模型编造一段回复（之前已确认）。`input_guard` 只检测空输入和安全问题，没检测"无意义输入"。

**已在 Prompt A 加了规则**：
> 相手のメッセージが極端に短いまたは意味をなさない場合は「もう一度おっしゃっていただけますか？」

但 Prompt 层面的约束不如代码层可靠。

**修法（可选）**：在 `input_guard` 加一个 `len(text.strip()) <= 3` 的判断，直接返回追问回退。

---

## 问题 7：索引页 "/" 返回的是什么

**位置**：[companion/main.py:16-18](../companion/main.py#L16-L18)

**当前**：`@app.get("/")` 返回 `{"service": "Sakura Companion", "docs": "/docs"}`

**但是**：`deploy/README.md` 说"浏览器打开 http://127.0.0.1:8015 即可使用"，如果 8015 的 "/" 不返回 `index.html`，用户会看到一个 JSON 而不是聊天界面。

**检查 deploy 版本的 companion/main.py**：deploy 目录下的 main.py 可能加了静态文件挂载。

**修法**：确认 deploy 版的 main.py 是否挂载了 `index.html`，如果没有需要补上。或者在 start.bat 里说明用户应该打开哪个 URL。

---

## 建议的修改优先级

| 优先级 | 问题 | 工作量 |
|:--:|------|:--:|
| 🔴 高 | 问题 1：Prompt B 不生效 | 30min |
| 🔴 高 | 问题 7：首页不返回前端 | 10min |
| 🟡 中 | 问题 2：Prompt B 质量落后 | 30min |
| 🟡 中 | 问题 6：无意义输入防御 | 10min |
| 🟢 低 | 问题 3：few-shot 语言风格 | 20min |
| 🟢 低 | 问题 5：流式质检 | 1-2h |
