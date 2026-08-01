import os
import json
import time
import threading
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import anthropic
from openai import OpenAI

from system_prompt import STAGE1_PROMPT, STAGE2_PROMPT, STAGE3_PROMPT

app = Flask(__name__)
CORS(app)
app.secret_key = os.environ.get("SECRET_KEY", "system-engineering-paper-generator-2024")

ADMIN_ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ADMIN_DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

DEEPSEEK_BASE_URL = "https://api.deepseek.com"

ANTHROPIC_MODELS = {
    "claude-opus-4-7": "Claude Opus 4.7（最强）",
    "claude-sonnet-4-6": "Claude Sonnet 4.6（推荐）",
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5（最快）",
}
DEEPSEEK_MODELS = {
    "deepseek-v4-pro": "DeepSeek V4 Pro（推荐，128K上下文）",
    "deepseek-v4-flash": "DeepSeek V4 Flash（快速）",
    "deepseek-chat": "DeepSeek Chat（通用）",
}

MODEL_MAX_OUTPUT = {
    "claude-opus-4-7": 32768,
    "claude-sonnet-4-6": 16384,
    "claude-haiku-4-5-20251001": 8192,
    "deepseek-v4-pro": 32768,
    "deepseek-v4-flash": 8192,
    "deepseek-chat": 8192,
}


def get_client(provider, api_key):
    if provider == "deepseek":
        return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    else:
        return anthropic.Anthropic(api_key=api_key)


def call_llm_stream(provider, client, system_prompt, user_message, model, max_tokens):
    """流式 LLM 调用，内置心跳防止超时"""
    q = []
    error = [None]
    done = [False]

    def _call():
        try:
            if provider == "deepseek":
                stream = client.chat.completions.create(
                    model=model, max_tokens=max_tokens, temperature=0.7,
                    stream=True,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ]
                )
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        q.append(chunk.choices[0].delta.content)
            else:
                with client.messages.stream(
                    model=model, max_tokens=max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}]
                ) as stream:
                    for text in stream.text_stream:
                        q.append(text)
        except Exception as e:
            error[0] = str(e)
        finally:
            done[0] = True

    thread = threading.Thread(target=_call, daemon=True)
    thread.start()

    # Yield from queue with heartbeat every 12 seconds to prevent timeout
    last_heartbeat = time.time()
    while not done[0] or q:
        if q:
            yield q.pop(0)
            last_heartbeat = time.time()
        elif not done[0]:
            if time.time() - last_heartbeat > 12:
                yield None  # heartbeat signal
                last_heartbeat = time.time()
            else:
                time.sleep(0.3)
        else:
            break

    thread.join(timeout=5)
    if error[0]:
        raise Exception(error[0])


def get_api_key_and_model(data):
    provider = data.get("provider", "anthropic")
    if provider == "deepseek":
        api_key = data.get("api_key", "").strip() or ADMIN_DEEPSEEK_KEY
        model = data.get("model", "") or "deepseek-v4-pro"
    else:
        api_key = data.get("api_key", "").strip() or ADMIN_ANTHROPIC_KEY
        model = data.get("model", "") or "claude-sonnet-4-6"
    return provider, api_key, model


# ========== 路由 ==========

@app.route("/")
def index():
    has_anthropic = bool(ADMIN_ANTHROPIC_KEY)
    has_deepseek = bool(ADMIN_DEEPSEEK_KEY)
    return render_template(
        "index.html",
        has_admin_key=has_anthropic or has_deepseek,
        has_anthropic=has_anthropic,
        has_deepseek=has_deepseek,
        anthropic_models=ANTHROPIC_MODELS,
        deepseek_models=DEEPSEEK_MODELS,
    )


@app.route("/api/stage1", methods=["POST"])
def stage1():
    """阶段1：期刊适配诊断 + 建议全文结构"""
    data = request.get_json()
    paper_text = data.get("paper_text", "").strip()
    paper_title = data.get("paper_title", "").strip()
    provider, api_key, model = get_api_key_and_model(data)

    if not paper_text or not api_key:
        return jsonify({"error": "缺少论文内容或 API Key"}), 400

    max_tok = MODEL_MAX_OUTPUT.get(model, 16384)
    user_msg = f"论文标题：{paper_title or '（未提供）'}\n\n论文全文：\n---\n{paper_text}\n---"

    def generate():
        try:
            client = get_client(provider, api_key)
            for chunk in call_llm_stream(provider, client, STAGE1_PROMPT, user_msg, model, max_tok):
                if chunk is None:
                    yield ": hb\n\n"  # heartbeat
                else:
                    yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': simplify_error(str(e))}, ensure_ascii=False)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    )


@app.route("/api/stage2", methods=["POST"])
def stage2():
    """阶段2：完整修改稿"""
    data = request.get_json()
    paper_text = data.get("paper_text", "").strip()
    stage1_output = data.get("stage1_output", "").strip()
    provider, api_key, model = get_api_key_and_model(data)

    if not paper_text or not stage1_output:
        return jsonify({"error": "缺少论文内容或阶段1结果"}), 400

    max_tok = MODEL_MAX_OUTPUT.get(model, 16384)
    s2_prompt = STAGE2_PROMPT.format(
        stage1_output=stage1_output[:6000],
        paper_text=paper_text
    )

    def generate():
        try:
            client = get_client(provider, api_key)
            for chunk in call_llm_stream(provider, client, s2_prompt, "请输出完整修改稿全文。", model, max_tok):
                if chunk is None:
                    yield ": hb\n\n"
                else:
                    yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': simplify_error(str(e))}, ensure_ascii=False)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    )


@app.route("/api/stage3", methods=["POST"])
def stage3():
    """阶段3：修改说明 + 风险清单与质量评分"""
    data = request.get_json()
    stage1_output = data.get("stage1_output", "").strip()
    stage2_output = data.get("stage2_output", "").strip()
    provider, api_key, model = get_api_key_and_model(data)

    if not stage2_output:
        return jsonify({"error": "缺少阶段2结果"}), 400

    max_tok = MODEL_MAX_OUTPUT.get(model, 16384)
    s3_prompt = STAGE3_PROMPT.format(
        stage1_output=stage1_output[:4000],
        stage2_output=stage2_output[:8000]
    )

    def generate():
        try:
            client = get_client(provider, api_key)
            for chunk in call_llm_stream(provider, client, s3_prompt, "请输出修改说明、风险清单和质量评分。", model, max_tok):
                if chunk is None:
                    yield ": hb\n\n"
                else:
                    yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': simplify_error(str(e))}, ensure_ascii=False)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    )


def simplify_error(msg):
    if "401" in msg or "Unauthorized" in msg:
        return "API Key 无效"
    if "429" in msg:
        return "API 调用频率超限，请稍后重试"
    if "timeout" in msg.lower():
        return "请求超时，请重试"
    return msg


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "version": "3.0.0",
        "providers": ["anthropic", "deepseek"]
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "production") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
