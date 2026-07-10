"""
ArrowCanaria 8B inference backend — pure model serving, no frontend or business logic.
Frontend + LangGraph + translate + logs → companion (:8015)
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch, torchvision
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from threading import Thread
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn, time, uuid, json, os

load_dotenv()

# ---- モデル読み込み ----
MODEL_NAME = 'DataPilot/ArrowCanaria-Llama-8B-SFT-v0.1'
print(f'Loading {MODEL_NAME}...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    attn_implementation="sdpa",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
    ),
    device_map='auto',
    torch_dtype=torch.float16,
)
print(f'Loaded! VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB')

# ---- FastAPI ----
app = FastAPI(title="ArrowCanaria Inference")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
async def root():
    return {"service": "ArrowCanaria Inference", "models": "/v1/models"}


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "arrowcanaria-8b"
    messages: list[Message]
    temperature: float = 0.75
    max_tokens: int = 256
    top_p: float = 0.9
    repetition_penalty: float = 1.12
    stream: bool = False


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[dict]


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": "arrowcanaria-8b", "object": "model"}]}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    msgs = [{"role": m.role, "content": m.content} for m in req.messages]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    if req.stream:
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            do_sample=True,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )
        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        async def generate():
            cid = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            try:
                for text in streamer:
                    chunk = {
                        "id": cid, "object": "chat.completion.chunk",
                        "created": int(time.time()), "model": req.model,
                        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                thread.join()  # 确保生成线程结束
                torch.cuda.empty_cache()  # 释放 KV cache

        return StreamingResponse(generate(), media_type="text/event-stream")

    # 非流式
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=req.max_tokens,
            temperature=req.temperature, do_sample=True,
            top_p=req.top_p, repetition_penalty=req.repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
        )
    response_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    del outputs
    torch.cuda.empty_cache()
    return ChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}", created=int(time.time()), model=req.model,
        choices=[{"index": 0, "message": {"role": "assistant", "content": response_text}, "finish_reason": "stop"}]
    )


if __name__ == "__main__":
    print("Starting ArrowCanaria inference on http://0.0.0.0:8014")
    uvicorn.run(app, host="0.0.0.0", port=8014)
