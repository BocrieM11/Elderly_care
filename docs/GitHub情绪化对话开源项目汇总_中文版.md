# 🧠 情绪化对话开源项目汇总

> **对象**：日本老年人 | **语言**：日语（中英文为参考） | **更新**：2026-07

---

## 📋 一分钟速览（TL;DR）

```
推荐技术栈（面向日本老年人）:

语音      →  J-Moshi        （名古屋大学·全双工·随声附和优化）
共情      →  KokoroChat      （电通大·ACL 2025·日语心理咨询）
声音克隆  →  Fun-CosyVoice3  （阿里巴巴·3秒克隆声音·家人声音）
记忆      →  Xiaoyou Core    （13种情感＋长期记忆＋生物节律）
评估      →  ESC-Judge       （EMNLP 2025·E-I-A 9维度评估）
```

| 成熟度 | 说明 |
|:---:|---|
| 🟢 | 商用/大规模验证 |
| 🟡 | 研究原型（有开源代码） |
| 🔴 | 仅论文/概念阶段 |

---

## 目录

| # | 章节 | 重要度 |
|:---:|---|:---:|
| 1 | [旗舰级硬件/生态（参考）](#一小智ai-硬件生态参考) | 📎 参考 |
| **2** | [**日本养老对话核心项目**](#二日本养老对话核心项目) | ⭐ **核心** |
| 3 | [学术 ESC情感支持对话](#三学术论文配套项目esc情感支持对话) | 📎 参考 |
| 4 | [学术 共情回复生成](#四学术论文配套项目共情回复生成) | 📎 参考 |
| 5 | [学术 语音/多模态](#五学术论文配套项目语音多模态) | 📎 参考 |
| 6 | [中文心理共情大模型](#六中文心理共情大模型) | 📎 参考 |
| 7 | [深度拟人情绪系统](#七深度拟人情绪系统) | 🔧 组件 |
| 8 | [轻量级/快速上手](#八轻量级快速上手项目) | 📎 参考 |
| **9** | [**养老场景选型建议**](#九养老场景选型建议) | ⭐ **核心** |

---

## 一、小智AI（硬件生态参考）

> 🟢 成熟度：商用 | 📎 硬件·架构参考用。不支持日语模型。

<details>
<summary><b>展开详情</b></summary>

### 小智AI — [78/xiaozhi-esp32](https://github.com/78/xiaozhi-esp32) ⭐ 25,000+

| 项目 | 内容 |
|:---|:---|
| 开发商 | 深圳十方融海科技 |
| 许可证 | MIT（可商用） |
| 设备数量 | 超120万台 |
| 日对话量 | 900万次 / 2700亿 Tokens |

**主要技术：**
- 多模态情感模型：2600万样本、26维人格体系
- 7维社交坐标系（亲密度·信任度·情感共鸣·记忆深度等）
- 500ms内识别26种情感
- 声纹克隆：4句话即可复制声音（3D Speaker技术）
- 记忆图谱＋情感计算的动态交互策略调整

| 相关仓库 | 用途 |
|:---|:---|
| [78/xiaozhi-esp32](https://github.com/78/xiaozhi-esp32) | ESP32固件 |
| [xiaozhi-server-go](https://github.com/AnimeAIChat/xiaozhi-server-go) | Go后端 |
| [xiaozhi-flutter-client](https://github.com/xfzen/xiaozhi-flutter-client) | Flutter客户端 |

> ⚠️ 硬件与语言无关，但AI模型以中文为主。必须替换为日语模型。

</details>

---

## 二、日本养老对话核心项目

### 🎯 整体架构

```
┌──────────┐    ┌──────────────┐    ┌─────────────────┐    ┌──────────┐
│  语音S2S  │ →  │   共情文本    │ →  │   老年人专用系统   │ →  │  评估层   │
│  J-Moshi │    │  KokoroChat  │    │  Mei-chan/BOCCO │    │ ESC-Judge│
│  🟡      │    │  🟡         │    │  🟢/🟢          │    │ 🔴      │
└──────────┘    └──────────────┘    └─────────────────┘    └──────────┘
```

---

### 2.1 🎙️ 语音对话 S2S

#### J-Moshi ⭐ 最优先

> 🟡 研究原型（已开源） | 名古屋大学 | Interspeech 2025

| 关键项 | 内容 |
|:---|:---|
| 基础模型 | Kyutai Moshi（开源全双工模型） |
| 训练数据 | J-CHAT（≈67,000小时）＋实验室录制数据 |
| 最大优势 | **日语随声附和（相槌）专用优化** — "原来如此""是这样啊""嗯" |
| 全双工 | 同时听＋说（无等待时间） |

> ✅ **最适合老年护理的理由**：全双工实现自然的对话节奏。没有翻译机式的不自然感。面向医疗对话的设计。
>
> ⚠️ **风险**：训练数据以年轻人为主。对老年人缓慢、不清晰发音的鲁棒性需要验证。

<br>

#### 其他 S2S 选项

<details>
<summary><b>📎 Fun-Audio-Chat-8B / Fun-CosyVoice3（点击展开）</b></summary>

| | Fun-Audio-Chat-8B | Fun-CosyVoice3 |
|:---|:---|:---|
| 开发商 | 阿里巴巴 | 阿里巴巴 |
| 成熟度 | 🟡 | 🟡 |
| 方式 | End-to-End S2S | TTS / S2S翻译 |
| 日语 | 多语言之一 | 9语言＋18方言（日语4,600h） |
| 情感 | 从语音识别＋共情回复 | 9种情感精细控制 |
| 声纹 | 无 | **3秒克隆** |
| 用途 | 情感识别＋共情对话 | **用家人的声音说话** |
| 链接 | [GitHub](https://github.com/FunAudioLLM/Fun-Audio-Chat) | — |

</details>

---

### 2.2 💬 日语共情对话模型

#### KokoroChat ⭐ 推荐

> 🟡 研究原型 | 电气通信大学 稻叶研究室 | **ACL 2025 Main**

| 关键项 | 内容 |
|:---|:---|
| 基础模型 | Llama-3.1-Swallow-8B-Instruct-v0.3 |
| 数据 | 由训练有素的咨询师通过角色扮演录制的 **6,589条日语心理咨询对话** |
| 适用领域 | 抑郁·家庭问题·人际关系·职业等 |
| 公开 | [GitHub](https://github.com/UEC-InabaLab/KokoroChat) ＋ [HF模型](https://huggingface.co/UEC-InabaLab/Llama-3.1-KokoroChat-Full) |

> ✅ 目前**最接近老年人心理咨询的日语共情数据集＋模型**。

<br>

<details>
<summary><b>📎 EmplifAI — 医疗共情专精（点击展开）</b></summary>

| 关键项 | 内容 |
|:---|:---|
| 来源 | IJCNLP/AACL 2025 |
| 数据 | 4,125条2轮对话、280个医疗场景 |
| 情感标签 | **28种细分情感**（GoEmotions日语适配版） |
| 模型 | LLM-jp-3.1-13b-instruct 微调 |

> 面向慢性病患者。适用于与慢性病老年人的对话。

</details>

---

### 2.3 👴 老年人专用对话系统

#### BOCCO emo —  Yukai工学 ⭐ 唯一有RCT证据的产品

> 🟢 **商用产品＋临床试验验证**

| 关键项 | 内容 |
|:---|:---|
| 临床试验 | **独居老年人73名（平均82.3岁）×4周 RCT** |
| 效果 | ✅ 孤独感 显著减少（-3.1, 95%CI -5.9~-0.4） |
| | ✅ 心理健康改善（+1.9, 95%CI 0.1~3.7） |
| 方式 | 人类操作员的共情回复通过机器人语音化输出 |

> 💡 **启示**：老年人×共情机器人的效果已被临床证实。通过AI化可以降低成本。

<br>

<details>
<summary><b>📎 其他5个老年人专精项目（点击展开）</b></summary>

#### Mei-chan — 神户大学 🟡

| 关键项 | 内容 |
|:---|:---|
| 方式 | FAISS＋LLM对话历史→自动构建用户画像→个性化共情回复 |
| 情感分析 | 日语BERT（ML分类＋人工定性解释的混合方法） |
| 实绩 | **长期运行中**，有真实世界数据 |

#### HarmonyLink — 庆应义塾大学＋Insight Edge 🟡

| 关键项 | 内容 |
|:---|:---|
| 特征 | 对话日志LLM摘要→**与家人共享→认知功能变化的早期检测** |
| 联动 | 语音＋LINE文本 |
| ACP | 促进预先护理计划（人生会议） |
| 专业性 | 基于认知症护理专科护士访谈的对话设计 |

#### 香川大学 — 倾听（验证疗法）模型 🟡

| 关键项 | 内容 |
|:---|:---|
| 来源 | JSAI 2025 |
| 方式 | 验证疗法技术（倾听·尊重·共情）→通过微调控制回复类型 |
| 目的 | 缓解护理人员短缺 |

#### 东北大学 — 机器人共情对话实验 🔴

| 关键项 | 内容 |
|:---|:---|
| 对象 | 65~78岁老年人8名×20分钟 |
| 方式 | 机器人＋日语BERT情感识别＋GPT-3.5回复 |
| 结果 | 老年人评价机器人为"好的倾听者" |

#### 老年医疗专用SLM — Zima 🟡

| 关键项 | 内容 |
|:---|:---|
| 模型 | Qwen 2.5 1.5B 微调 |
| 对象 | 孤独感·服药管理·慢性疾病等 |
| 限制 | 不支持日语（英语/中文）→ 架构参考 |
| 链接 | [HF](https://huggingface.co/YsK-dev/zima-qwen-geriatric-1.5b) |

</details>

---

### 2.4 📦 重要数据集

| 数据集 | 规模 | 特征 | 老年人适配度 |
|:---|:---|:---|:---:|
| **Magic Data** 🟡 | 日语全双工会话 | 精细标注随声附和·填充词·打断。**面向老年人看护** | ⭐⭐⭐ |
| KokoroChat 🟡 | 6,589对话 | 专业咨询师心理咨询角色扮演 | ⭐⭐⭐ |
| EmplifAI 🟡 | 4,125对话 | 面向慢性病患者28情感标签 | ⭐⭐ |
| J-CHAT 🟡 | ≈67,000小时 | 播客/YouTube（年轻人为主） | ⭐ |

---

### 2.5 🏆 日语模型选型总结

<table>
<tr><th>层级</th><th>🥇 最优先</th><th>🥈 替代方案</th><th>理由</th></tr>
<tr><td>语音 S2S</td><td><b>J-Moshi</b></td><td>Fun-Audio-Chat-8B</td><td>日语全双工＋随声附和 vs 多语言之一</td></tr>
<tr><td>共情文本</td><td><b>KokoroChat</b></td><td>EmplifAI</td><td>心理咨询 vs 医疗专精</td></tr>
<tr><td>声纹克隆</td><td><b>Fun-CosyVoice3</b></td><td>—</td><td>3秒＋日语4,600h训练</td></tr>
<tr><td>老年人专用</td><td><b>Mei-chan</b>（长期验证）</td><td>HarmonyLink</td><td>运行实绩 vs 认知症专精</td></tr>
<tr><td>临床证据</td><td><b>BOCCO emo</b>（RCT）</td><td>—</td><td>唯一的随机对照试验</td></tr>
</table>

---

## 三、学术论文配套项目（ESC情感支持对话）

<details>
<summary><b>📎 5个ESC项目（点击展开）</b></summary>

| 项目 | 来源 | 核心技术 | 日语适用 |
|:---|:---|:---|:---|
| [AffectiveFlow](https://github.com/chz2025/AffectiveFlow) 🟡 | — | MCTS蒸馏＋情感流优先优化（AFPO） | ✅ 模型无关 |
| [IntentionESC](https://github.com/43zxj/IntentionESC_ICECoT) 🟡 | ACL 2025 | ICECoT：情感分析→意图推理→策略选择→回复 | ✅ 推理链与语言无关 |
| [ESC-Judge](https://anonymous.4open.science/r/ESC-Judge-A508) 🔴 | EMNLP 2025 | E-I-A咨询模型9维度自动评估 | ✅ 评估标准语言通用 |
| [NLPCC 2025 Task8](https://github.com/Jin-zd/NLPCC-2025-Shared-Task8) 🟡 | NLPCC 2025 | 个性化情感支持对话（PESC） | ⚠️ 中文数据 → 设计参考 |
| [ESC-Skills/Qwen DianJin](https://github.com/aliyun/qwen-dianjin) 🟡 | — | Intervention Units (IUs) 技能中心架构 | ✅ 设计思想与语言无关 |

</details>

---

## 四、学术论文配套项目（共情回复生成）

<details>
<summary><b>📎 5个共情回复项目（点击展开）</b></summary>

| 项目 | 来源 | 核心技术 |
|:---|:---|:---|
| [NEC-EmpChat](https://github.com/huangfu170/NEC-empchat) 🟡 | COLING 2025 | 非情感中心化共情对话：对比学习＋上下文敏感实体 |
| [StyEmp](https://github.com/fuyahuii/StyEmp) 🟡 | SIGDIAL 2024 | 多粒度前缀编码器＋人格增强对比学习 |
| [EGRET](https://github.com/Jithendra-k/EGRET) 🟡 | — | 情感图谱追踪＋因果检测＋优先优化。**带Streamlit UI** |
| [ECC](https://github.com/Yuan-23/ECC) 🔴 | EMNLP 2025 | 情感-因果对话数据集自动生成（2.4K对话） |
| [AFEC](https://github.com/yuboxie/afec) 🔴 | — | 从Reddit自动构建的知识图谱（134K说话者+666K倾听者） |

</details>

---

## 五、学术论文配套项目（语音/多模态）

<details>
<summary><b>📎 3个语音/多模态项目（点击展开）</b></summary>

| 项目 | 来源 | 核心技术 | 日语 |
|:---|:---|:---|:---:|
| [OpenS2S](https://casia-lm.github.io/OpenS2S) 🟡 | CASIA 2025 | 首个全开源端到端共情语音对话大模型 | ❌ 仅中英文 |
| [Chain-Talker](https://github.com/AI-S2-Lab/Chain-Talker) 🟡 | ACL 2025 | 情感理解→语义理解→共情渲染 3阶段 | ⚠️ |
| [OSUM-EChat](https://aslp-lab.github.io/osum-echat.github.io/) 🟡 | — | 理解驱动型端到端共情语音聊天机器人 | ⚠️ |

</details>

---

## 六、中文心理共情大模型

<details>
<summary><b>📎 SoulChat / HeartLink — 不支持日语，架构参考用（点击展开）</b></summary>

| 模型 | 开发商 | 基础模型 | 数据规模 |
|:---|:---|:---|:---|
| [SoulChat](https://github.com/wosiwo/SoulChat) 🟡 | 华南理工大学 | ChatGLM-6B | 120万条＋共情对话 |
| [HeartLink](https://github.com/Nobody-ML/HeartLink) 🟡 | — | InternLM2-7B/20B | 18万条心理咨询QA（12+场景） |

> ⚠️ 中文专用。面向日本老年人请使用 KokoroChat。

</details>

---

## 七、深度拟人情绪系统

> 🔧 情感引擎/记忆系统。与语言无关，可与日语LLM组合使用。

<details>
<summary><b>📎 Xiaoyou Core / PsyArch Agent（点击展开）</b></summary>

### Xiaoyou Core — [hakituo/xiaoyou-core](https://github.com/hakituo/xiaoyou-core) 🟡

| 功能 | 内容 |
|:---|:---|
| 情感 | 13种基本情感的动态管理 |
| 仿生 | 神经递质·昼夜节律·能量系统 |
| 记忆 | 概率回忆＋加权记忆＋向量检索（36模块） |
| 容错 | 自我修复功能（自动故障检测＋恢复） |
| 多终端 | Web / Android / iOS / QQ / Telegram |

### PsyArch Agent — [ginsonko/PsyArch-Agent](https://github.com/ginsonko/PsyArch-Agent) 🔴

> 情感不是LLM"推理"的标签，而是**本地持续演化的内生状态**

</details>

---

## 八、轻量级/快速上手项目

<details>
<summary><b>📎 3个轻量项目（点击展开）</b></summary>

| 项目 | 技术栈 | 特征 | 日语 |
|:---|:---|:---|:---:|
| [LingChat](https://github.com/SlimeBoyOwO/LingChat) 🟡 | Windows版Exe | 18情感识别＋桌面状态识别 | ❌ 中文 |
| [praise-ai](https://github.com/hexart/praise-ai) 🟡 | React 19＋FastAPI | Ollama/OpenAI/Claude自由切换 | ✅ 易于替换 |
| [AI-Emotion-Assistant-2](https://github.com/lihuacatnb/AI-Emotion-Assistant-2) 🟡 | Vue＋Ollama | 完全本地·隐私保护 | ⚠️ |

</details>

---

## 九、养老场景选型建议

### 🏗️ 推荐技术栈

```
┌─────────────────────────────────────────────────────────────┐
│  🔊 语音对话层                                               │
│  J-Moshi 全双工（随声附和优化）＋ Fun-CosyVoice3 家人声纹克隆  │
├─────────────────────────────────────────────────────────────┤
│  💬 共情对话层                                               │
│  KokoroChat 心理咨询 ＋ ICECoT 思维链推理                     │
├─────────────────────────────────────────────────────────────┤
│  👴 老年人专用层                                              │
│  Mei-chan 画像自动构建 ＋ 香川大学 倾听回复                    │
│  ＋ HarmonyLink 认知功能变化检测（可选）                       │
├─────────────────────────────────────────────────────────────┤
│  🧠 情感·记忆层（语言无关）                                   │
│  Xiaoyou Core 13情感＋长期记忆 ＋ PsyArch 内生情感演化         │
├─────────────────────────────────────────────────────────────┤
│  📏 质量评估层                                                │
│  ESC-Judge E-I-A 9维度自动评估 ＋ 日语母语者主观评价           │
└─────────────────────────────────────────────────────────────┘
```

### 📊 开发路线图（按优先级排序）

| Phase | 优先级 | 要做的事 | 使用工具 | 完成标准 |
|:---:|:---:|:---|:---|:---|
| **1** | 🔴 P0 | 语音对话最小基础 | J-Moshi 部署 | 3分钟自然的语音交互 |
| **2** | 🔴 P0 | 引入共情对话 | KokoroChat＋老年人向Prompt | 倾听＋共情回复的自动生成 |
| **3** | 🟡 P1 | 老年人画像 | Mei-chan方式＋向量DB | 基于对话历史的个性化 |
| **4** | 🟡 P1 | 临床效果验证 | 参照BOCCO emo评估方法 | 孤独感量表改善确认 |
| **5** | 🟢 P2 | 声纹克隆 | Fun-CosyVoice3 | 用家人的声音成功呼叫 |
| **6** | 🟢 P2 | 长期记忆＋情感进化 | Xiaoyou Core | 持续2周对话中人格一致性维持 |
| **7** | 🔵 P3 | 认知功能监测 | HarmonyLink | 面向家人的周报自动生成 |

### ⚠️ 风险管理

| 风险 | 严重度 | 对策 |
|:---|:---:|:---|
| J-Moshi无法识别老年人的声音 | 🔴 高 | 与OpenAI Whisper（日语）混合架构作为后备方案 |
| KokoroChat无法应对战争经历·往事等话题 | 🟡 中 | 用老年人领域数据进行额外微调 |
| 方言（关西话·东北话等）不支持 | 🟡 中 | Fun-CosyVoice3方言功能＋各地ASR |
| **独居老年人的紧急情况应对** | 🔴 **致命** | AI专注于异常检测。报警由人工操作员/家人联动处理。**不能仅靠对话解决** |
| 隐私（对话数据的处理） | 🔴 高 | 本地推理＋个人信息脱敏＋取得同意 |

---

## 📋 附录：全部项目一览

| # | 名称 | 类型 | 成熟度 | 日语 | 主要用途 |
|:---:|:---|:---|:---:|:---:|:---|
| 1 | J-Moshi | 语音S2S | 🟡 | ✅ | 全双工语音对话 |
| 2 | KokoroChat | 共情文本 | 🟡 | ✅ | 心理咨询对话 |
| 3 | BOCCO emo | 机器人 | 🟢 | ✅ | RCT验证老年人陪伴 |
| 4 | Fun-CosyVoice3 | TTS/声纹 | 🟡 | ✅ | 家人声音克隆 |
| 5 | Mei-chan | 老年人语音助手 | 🟡 | ✅ | 长期运行实绩 |
| 6 | HarmonyLink | 认知症护理 | 🟡 | ✅ | 认知功能变化检测 |
| 7 | 香川大学 倾听 | 倾听模型 | 🟡 | ✅ | 减轻护理负担 |
| 8 | EmplifAI | 共情数据 | 🟡 | ✅ | 28情感医疗对话 |
| 9 | Fun-Audio-Chat-8B | 语音S2S | 🟡 | ⚠️ | 情感识别＋共情回复 |
| 10 | Xiaoyou Core | 情感引擎 | 🟡 | 🔧 | 长期记忆＋情感演化 |
| 11 | PsyArch Agent | 情感引擎 | 🔴 | 🔧 | 内生情感状态 |
| 12 | ESC-Judge | 质量评估 | 🔴 | 🔧 | E-I-A 9维度评估 |
| 13 | AffectiveFlow | ESC策略 | 🟡 | 🔧 | MCTS情感流 |
| 14 | IntentionESC | ESC策略 | 🟡 | 🔧 | ICECoT思维链 |
| 15 | 小智AI | 硬件/生态 | 🟢 | ❌ | 硬件参考 |
| 16 | SoulChat | 共情文本 | 🟡 | ❌ | 中文心理咨询 |
| 17 | OpenS2S | 语音S2S | 🟡 | ❌ | 架构参考 |

> **图例**：🟢 商用/大规模验证 | 🟡 研究开源 | 🔴 论文/概念 | ✅ 支持 | ⚠️ 部分支持 | ❌ 不支持 | 🔧 语言无关

---

> **许可证注意**：各项目的许可证（MIT/Apache 2.0/学术用途）在商用前需要逐一确认。
