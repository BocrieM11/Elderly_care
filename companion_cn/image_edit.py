"""
Qwen-Image-Edit-2511 客户端封装
支持 vLLM OpenAI 兼容 API + 自定义 /edit 端点
"""
import base64
import time
import requests
from pathlib import Path
from typing import Optional
from openai import OpenAI


class ImageEditor:
    """Qwen-Image-Edit-2511 客户端。

    用法:
        editor = ImageEditor("http://192.168.253.91:8200")
        if editor.ping():
            editor.edit("photo.png", "把背景换成海边日落", output="result.png")
    """

    def __init__(self, base_url: str = "http://192.168.253.91:8200", timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._mode: Optional[str] = None  # "openai" | "custom" | None

    # ── 连通性检测 ──

    def ping(self) -> dict:
        """检测服务是否可达 + 自动识别 API 模式。返回状态字典。"""
        result = {"online": False, "mode": None, "model": None, "endpoints": []}

        # 1) 尝试 OpenAI 兼容路径
        for path in ["/v1/models", "/models"]:
            try:
                r = requests.get(f"{self.base_url}{path}", timeout=5)
                if r.status_code == 200:
                    result["online"] = True
                    result["mode"] = "openai"
                    data = r.json()
                    models = data.get("data", [])
                    if models:
                        result["model"] = models[0].get("id", "")
                    result["endpoints"].append(f"{self.base_url}/v1/images/edits")
                    return result
            except Exception:
                pass

        # 2) 尝试自定义 /edit 端点
        try:
            r = requests.post(
                f"{self.base_url}/edit",
                json={"prompt": "ping"},
                timeout=5,
            )
            if r.status_code in (200, 400, 422):  # 400/422 说明端点存在只是参数不全
                result["online"] = True
                result["mode"] = "custom"
                result["endpoints"].append(f"{self.base_url}/edit")
                return result
        except Exception:
            pass

        # 3) 简单 TCP 探测
        try:
            import socket
            host = self.base_url.split("://")[1].split(":")[0]
            port = int(self.base_url.split(":")[-1])
            s = socket.socket()
            s.settimeout(3)
            s.connect((host, port))
            s.close()
            result["online"] = True
            result["mode"] = "unknown"
            result["note"] = "端口可达但API未响应"
        except Exception:
            pass

        return result

    # ── 图片编辑 ──

    def edit(
        self,
        image: str | Path,
        prompt: str,
        output: str | Path | None = None,
        guidance_scale: float = 7.5,
        size: str = "1024x1024",
    ) -> Optional[bytes]:
        """编辑图片。

        参数:
            image:  输入图片路径
            prompt: 编辑指令（中文即可）如 "把背景换成海边日落"
            output: 输出路径，不指定则只返回 bytes
            guidance_scale: 指令遵循度 (1-20)，越高越听话但可能失真
            size:   输出尺寸

        返回:
            图片 bytes 或 None（失败时）
        """
        image = Path(image)
        if not image.exists():
            print(f"[ERROR] 图片不存在: {image}")
            return None

        if not self._mode:
            print("[INFO] 未探测过，自动 ping...")
            self.ping()

        if self._mode == "openai":
            return self._edit_via_openai(image, prompt, output, guidance_scale, size)
        elif self._mode == "custom":
            return self._edit_via_custom(image, prompt, output)
        else:
            print("[ERROR] 服务不可达，请先调用 ping() 确认")
            return None

    def _edit_via_openai(self, image: Path, prompt: str, output, guidance_scale, size):
        """vLLM / OpenAI 兼容 API"""
        client = OpenAI(base_url=f"{self.base_url}/v1", api_key="x")
        ext = image.suffix.lower()
        mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
        mime = mime_map.get(ext, "image/png")

        with open(image, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        print(f"[EDIT] 发送编辑请求: {prompt}")
        t0 = time.time()
        try:
            r = client.images.edit(
                model="Qwen-Image-Edit-2511",
                image=f"data:{mime};base64,{b64}",
                prompt=prompt,
                n=1,
                size=size,
                extra_body={"guidance_scale": guidance_scale},
                timeout=self.timeout,
            )
        except Exception as e:
            print(f"[ERROR] API 调用失败: {e}")
            return None

        elapsed = time.time() - t0
        url = r.data[0].url if r.data else ""
        print(f"[OK] 完成 → {elapsed:.1f}s")

        if url:
            img_bytes = self._download_url(url)
            return self._save_or_return(img_bytes, output)

    def _edit_via_custom(self, image: Path, prompt: str, output):
        """自定义 /edit 端点"""
        print(f"[EDIT] 发送编辑请求: {prompt}")
        t0 = time.time()
        try:
            with open(image, "rb") as f:
                r = requests.post(
                    f"{self.base_url}/edit",
                    files={"image": (image.name, f, "image/png")},
                    data={"prompt": prompt},
                    timeout=self.timeout,
                )
            if r.status_code != 200:
                print(f"[ERROR] HTTP {r.status_code}: {r.text[:200]}")
                return None
        except Exception as e:
            print(f"[ERROR] 请求失败: {e}")
            return None

        elapsed = time.time() - t0
        print(f"[OK] 完成 → {elapsed:.1f}s")
        return self._save_or_return(r.content, output)

    # ── helpers ──

    def _download_url(self, url: str) -> Optional[bytes]:
        try:
            r = requests.get(url, timeout=30)
            return r.content
        except Exception as e:
            print(f"[ERROR] 下载结果失败: {e}")
            return None

    def _save_or_return(self, img_bytes, output):
        if output:
            Path(output).write_bytes(img_bytes)
            print(f"[SAVE] → {output}  ({len(img_bytes)} bytes)")
        return img_bytes


# ── CLI ──

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Qwen-Image-Edit-2511 客户端")
    parser.add_argument("--url", default="http://192.168.253.91:8200", help="服务地址")
    parser.add_argument("--ping", action="store_true", help="检测连通性")
    parser.add_argument("--image", help="输入图片路径")
    parser.add_argument("--prompt", help="编辑指令 (中文)")
    parser.add_argument("--output", "-o", help="输出图片路径")
    parser.add_argument("--guidance", type=float, default=7.5, help="指令遵循度")
    args = parser.parse_args()

    editor = ImageEditor(args.url)

    if args.ping:
        import json
        print(json.dumps(editor.ping(), ensure_ascii=False, indent=2))
    elif args.image and args.prompt:
        editor.edit(args.image, args.prompt, args.output, args.guidance)
    else:
        parser.print_help()
