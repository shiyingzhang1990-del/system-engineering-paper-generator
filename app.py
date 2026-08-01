import os
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import anthropic
from openai import OpenAI

from system_prompt import STAGE1_PROMPT, STAGE2_PROMPT, STAGE3_PROMPT

app = Flask(__name__)
CORS(app)
app.secret_key = os.environ.get("SECRET_KEY", "x")

ADMIN_ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ADMIN_DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

ANTHROPIC_MODELS = {
    "claude-opus-4-7": "Claude Opus 4.7（最强）",
    "claude-sonnet-4-6": "Claude Sonnet 4.6（推荐）",
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5（最快）",
}
DEEPSEEK_MODELS = {
    "deepseek-v4-pro": "DeepSeek V4 Pro（推荐）",
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
    return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL) if provider == "deepseek" else anthropic.Anthropic(api_key=api_key)


def stream_llm(provider, client, system_prompt, user_message, model, max_tokens):
    """直接流式调用 LLM，不做额外的线程封装"""
    if provider == "deepseek":
        s = client.chat.completions.create(
            model=model, max_tokens=max_tokens, temperature=0.7, stream=True,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]
        )
        for chunk in s:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    else:
        with client.messages.stream(
            model=model, max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}]
        ) as s:
            for text in s.text_stream:
                yield text


def make_sse(fn, provider, api_key, model, system_prompt, user_msg):
    """构建 SSE 响应的 generator"""
    max_tok = MODEL_MAX_OUTPUT.get(model, 16384)

    def generate():
        try:
            client = get_client(provider, api_key)
            for chunk in stream_llm(provider, client, system_prompt, user_msg, model, max_tok):
                yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': simplify_error(str(e))}, ensure_ascii=False)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    )


def get_api_key_and_model(data):
    provider = data.get("provider", "anthropic")
    if provider == "deepseek":
        return provider, data.get("api_key", "").strip() or ADMIN_DEEPSEEK_KEY, data.get("model", "") or "deepseek-v4-pro"
    return provider, data.get("api_key", "").strip() or ADMIN_ANTHROPIC_KEY, data.get("model", "") or "claude-sonnet-4-6"


def simplify_error(msg):
    msg = str(msg)
    if "401" in msg or "Unauthorized" in msg: return "API Key 无效"
    if "429" in msg: return "API 调用频率超限，请稍后重试"
    if "timeout" in msg.lower(): return "请求超时，请重试"
    return msg[:200]


# ========== 路由 ==========

@app.route("/")
def index():
    return render_template("index.html",
        has_admin_key=bool(ADMIN_ANTHROPIC_KEY) or bool(ADMIN_DEEPSEEK_KEY),
        has_anthropic=bool(ADMIN_ANTHROPIC_KEY),
        has_deepseek=bool(ADMIN_DEEPSEEK_KEY),
        anthropic_models=ANTHROPIC_MODELS, deepseek_models=DEEPSEEK_MODELS)


@app.route("/api/stage1", methods=["POST"])
def stage1():
    data = request.get_json()
    paper_text = data.get("paper_text", "").strip()
    paper_title = data.get("paper_title", "").strip()
    provider, api_key, model = get_api_key_and_model(data)
    if not paper_text or not api_key:
        return jsonify({"error": "缺少论文内容或 API Key"}), 400
    user_msg = f"论文标题：{paper_title or '（未提供）'}\n\n论文全文：\n---\n{paper_text}\n---"
    return make_sse(stage1, provider, api_key, model, STAGE1_PROMPT, user_msg)


@app.route("/api/stage2", methods=["POST"])
def stage2():
    data = request.get_json()
    paper_text = data.get("paper_text", "").strip()
    stage1_output = data.get("stage1_output", "").strip()
    provider, api_key, model = get_api_key_and_model(data)
    if not paper_text or not stage1_output:
        return jsonify({"error": "缺少论文内容或阶段1结果"}), 400
    s2_prompt = STAGE2_PROMPT.format(stage1_output=stage1_output[:6000], paper_text=paper_text)
    return make_sse(stage2, provider, api_key, model, s2_prompt, "请输出完整修改稿全文。")


@app.route("/api/stage3", methods=["POST"])
def stage3():
    data = request.get_json()
    stage1_output = data.get("stage1_output", "").strip()
    stage2_output = data.get("stage2_output", "").strip()
    provider, api_key, model = get_api_key_and_model(data)
    if not stage2_output:
        return jsonify({"error": "缺少阶段2结果"}), 400
    s3_prompt = STAGE3_PROMPT.format(stage1_output=stage1_output[:4000], stage2_output=stage2_output[:8000])
    return make_sse(stage3, provider, api_key, model, s3_prompt, "请输出修改说明、风险清单和质量评分。")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "version": "4.0.0"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_ENV") == "development")
