# Elderly Care / OLV Portable

这是一个面向养老陪伴场景的 Windows 本地部署项目，整合了中文陪伴网页、视觉事件接入、语音唤醒、语音识别、流式语音合成、提醒、新闻和数字人动作。

## 主要目录

- `app`：OLV 网关及 Open-LLM-VTuber 后端。
- `companion_cn`：中文陪伴服务、网页、记忆、提醒与 OpenAI 兼容接口。
- `services/funasr`：Fun-ASR HTTP 服务。
- `services/fishspeech`：Fish Speech 流式 TTS 适配服务。
- `services/cosyvoice`：保留的 CosyVoice 代码与参考音频资源。
- `models`：体积较小的离线语音唤醒模型；其余模型由脚本下载。
- `scripts`：安装、启动、停止、状态检查及网络配置脚本。
- `samples`：检测事件调用示例。

## 未纳入仓库的内容

为控制仓库大小并保护本机数据，以下内容不会上传：Python 虚拟环境、运行日志、聊天记录、数据库、缓存、本机绝对路径配置、模型重复压缩包，以及网页未引用的原始动作素材。网页实际使用的 `*_pingpong.png` 往返动作素材已保留。

## 环境要求

- Windows 10/11
- PowerShell 5.1 或更高版本
- NVIDIA 显卡及可用的 CUDA 驱动（语音服务推荐）
- [Ollama](https://ollama.com/download/windows)
- [uv](https://docs.astral.sh/uv/)
- 稳定网络和足够的模型存储空间

安装 uv：

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## 首次部署

克隆仓库后，在项目根目录执行：

```powershell
Copy-Item .\scripts\local-settings.example.ps1 .\scripts\local-settings.ps1
Copy-Item .\companion_cn\.env.example .\companion_cn\.env
```

根据当前电脑修改以上两个本地配置文件。若使用默认本地服务且不需要聊天记录上报，可保持 `.env` 中的上报功能关闭。

随后安装依赖和模型：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1
```

安装脚本会创建所需虚拟环境、安装 Python 依赖、拉取 Ollama 模型并下载语音模型。也可单独执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\download-models.ps1
```

## 启动与停止

```powershell
# 启动全部服务
.\scripts\start-all.ps1

# 查看状态
.\scripts\status.ps1

# 停止全部服务
.\scripts\stop-all.ps1
```

启动完成后访问 <http://127.0.0.1:12393/>。

默认端口：

| 端口 | 服务 |
| --- | --- |
| `8016` | Companion CN / OpenAI 兼容接口 |
| `10095` | Fun-ASR |
| `11434` | Ollama |
| `12393` | OLV 网关与网页 |
| `50000` | Fish Speech 流式 TTS |

## 局域网检测事件接入

检测电脑向 OLV 电脑发送：

```http
POST http://<OLV电脑IP>:12393/api/detection-events
Content-Type: application/json
```

坐起事件示例：

```json
{
  "event_id": "0001_00012p34s_bed_exit_level_1",
  "event_type": "sit_up",
  "occurred_at": "2026-07-15T15:30:12+08:00"
}
```

跌倒事件使用 `event_type: "fall_detected"`。需要放行指定检测电脑时，以管理员身份执行：

```powershell
.\scripts\allow-detection-client.ps1 -DetectionComputerIp '192.168.252.XX'
```

请使用 OLV 电脑真实局域网地址，不要从另一台电脑访问 `127.0.0.1`。摄像头和麦克风在跨设备访问时通常需要 HTTPS 或浏览器站点权限配置。

## 配置与隐私

- 密钥和服务令牌只写入本地 `.env`，不要提交到 Git。
- `scripts/local-settings.ps1` 用于本机 Python、模型和外部服务路径覆盖。
- 对话、提醒及新闻数据库保存在 `companion_cn/data`，该目录不会被 Git 跟踪。
- 检测事件日志默认保存在 `logs/detection_events.jsonl`，该目录不会被 Git 跟踪。
- 若启用聊天记录上报，请使用 HTTPS、限制来源 IP，并为接收端配置独立的强令牌。

更多接口说明见 [`companion_cn/API.md`](companion_cn/API.md)。
