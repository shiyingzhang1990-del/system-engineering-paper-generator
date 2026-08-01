import os
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import anthropic
from openai import OpenAI

from system_prompt import SYSTEM_PROMPT

app = Flask(__name__)
CORS(app)
app.secret_key = os.environ.get("SECRET_KEY", "system-engineering-paper-generator-2024")

# 管理员预设 API Keys（环境变量）
ADMIN_ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ADMIN_DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

# DeepSeek API 配置
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_DEFAULT_MODEL = "deepseek-v4-pro"

# Anthropic 默认模型
ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-4-6"

# 模型列表
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


def build_user_prompt(paper_text, paper_title=""):
    """构建用户提示词"""
    return f"""请处理以下论文。严格按照系统指令输出五部分内容。

论文标题：{paper_title or "（未提供）"}

论文全文：
---
{paper_text}
---"""


# ========== Anthropic 处理函数 ==========

# 各模型最大输出 token 限制
MODEL_MAX_OUTPUT = {
    # Anthropic
    "claude-opus-4-7": 32768,
    "claude-sonnet-4-6": 16384,
    "claude-haiku-4-5-20251001": 8192,
    # DeepSeek
    "deepseek-v4-pro": 32768,
    "deepseek-v4-flash": 8192,
    "deepseek-chat": 8192,
}
DEFAULT_MAX_TOKENS = 32000


def get_max_tokens(model):
    """根据模型获取合适的 max_tokens，留一些余量"""
    return MODEL_MAX_OUTPUT.get(model, DEFAULT_MAX_TOKENS)


def process_paper_anthropic_sync(paper_text, api_key, model, paper_title):
    """Anthropic API 同步处理"""
    client = anthropic.Anthropic(api_key=api_key)
    max_tok = get_max_tokens(model)
    response = client.messages.create(
        model=model,
        max_tokens=max_tok,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": build_user_prompt(paper_text, paper_title)}
        ]
    )
    return response.content[0].text


def process_paper_anthropic_stream(paper_text, api_key, model, paper_title):
    """Anthropic API 流式处理"""
    client = anthropic.Anthropic(api_key=api_key)
    max_tok = get_max_tokens(model)
    with client.messages.stream(
        model=model,
        max_tokens=max_tok,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": build_user_prompt(paper_text, paper_title)}
        ]
    ) as stream:
        for text in stream.text_stream:
            yield text


# ========== DeepSeek 处理函数 ==========

def process_paper_deepseek_sync(paper_text, api_key, model, paper_title):
    """DeepSeek API 同步处理（OpenAI 兼容格式）"""
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    max_tok = get_max_tokens(model)

    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tok,
        temperature=0.7,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(paper_text, paper_title)}
        ]
    )
    return response.choices[0].message.content


def process_paper_deepseek_stream(paper_text, api_key, model, paper_title):
    """DeepSeek API 流式处理（OpenAI 兼容格式）"""
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    max_tok = get_max_tokens(model)

    stream = client.chat.completions.create(
        model=model,
        max_tokens=max_tok,
        temperature=0.7,
        stream=True,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(paper_text, paper_title)}
        ]
    )
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


# ========== 路由 ==========

@app.route("/")
def index():
    """主页"""
    has_anthropic = bool(ADMIN_ANTHROPIC_KEY)
    has_deepseek = bool(ADMIN_DEEPSEEK_KEY)
    has_admin_key = has_anthropic or has_deepseek
    return render_template(
        "index.html",
        has_admin_key=has_admin_key,
        has_anthropic=has_anthropic,
        has_deepseek=has_deepseek,
        anthropic_models=ANTHROPIC_MODELS,
        deepseek_models=DEEPSEEK_MODELS,
    )


@app.route("/api/process", methods=["POST"])
def process():
    """处理论文（非流式，支持双 API）"""
    try:
        data = request.get_json()
        paper_text = data.get("paper_text", "").strip()
        paper_title = data.get("paper_title", "").strip()
        provider = data.get("provider", "anthropic")  # "anthropic" 或 "deepseek"
        model = data.get("model", "")

        # 获取 API Key
        if provider == "deepseek":
            api_key = data.get("api_key", "").strip() or ADMIN_DEEPSEEK_KEY
            if not model:
                model = DEEPSEEK_DEFAULT_MODEL
        else:
            api_key = data.get("api_key", "").strip() or ADMIN_ANTHROPIC_KEY
            if not model:
                model = ANTHROPIC_DEFAULT_MODEL

        if not paper_text:
            return jsonify({"error": "请输入论文内容"}), 400
        if len(paper_text) < 500:
            return jsonify({"error": "论文内容过短（至少500字），请检查后重新提交"}), 400
        if not api_key:
            provider_name = "DeepSeek" if provider == "deepseek" else "Anthropic"
            return jsonify({"error": f"请提供 {provider_name} API Key"}), 400

        # 按 provider 分发
        if provider == "deepseek":
            result = process_paper_deepseek_sync(paper_text, api_key, model, paper_title)
        else:
            result = process_paper_anthropic_sync(paper_text, api_key, model, paper_title)

        return jsonify({
            "success": True,
            "result": result,
            "provider": provider,
            "model": model,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        error_msg = str(e)
        # 简化常见错误信息
        if "401" in error_msg or "Unauthorized" in error_msg:
            error_msg = "API Key 无效，请检查后重试"
        elif "429" in error_msg or "rate" in error_msg.lower():
            error_msg = "API 调用频率超限，请稍后重试"
        elif "timeout" in error_msg.lower():
            error_msg = "请求超时，论文过长或网络不稳定"
        return jsonify({"error": f"处理失败: {error_msg}"}), 500


@app.route("/api/process/stream", methods=["POST"])
def process_stream():
    """处理论文（流式，支持双 API）"""
    data = request.get_json()
    paper_text = data.get("paper_text", "").strip()
    paper_title = data.get("paper_title", "").strip()
    provider = data.get("provider", "anthropic")
    model = data.get("model", "")

    if provider == "deepseek":
        api_key = data.get("api_key", "").strip() or ADMIN_DEEPSEEK_KEY
        if not model:
            model = DEEPSEEK_DEFAULT_MODEL
    else:
        api_key = data.get("api_key", "").strip() or ADMIN_ANTHROPIC_KEY
        if not model:
            model = ANTHROPIC_DEFAULT_MODEL

    if not paper_text:
        return jsonify({"error": "请输入论文内容"}), 400
    if len(paper_text) < 500:
        return jsonify({"error": "论文内容过短（至少500字）"}), 400
    if not api_key:
        provider_name = "DeepSeek" if provider == "deepseek" else "Anthropic"
        return jsonify({"error": f"请提供 {provider_name} API Key"}), 400

    def generate():
        try:
            if provider == "deepseek":
                stream = process_paper_deepseek_stream(paper_text, api_key, model, paper_title)
            else:
                stream = process_paper_anthropic_stream(paper_text, api_key, model, paper_title)

            for chunk in stream:
                yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            error_msg = str(e)
            yield f"data: {json.dumps({'error': error_msg}, ensure_ascii=False)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.route("/api/models", methods=["GET"])
def get_models():
    """返回可用模型列表"""
    return jsonify({
        "anthropic": ANTHROPIC_MODELS,
        "deepseek": DEEPSEEK_MODELS,
    })


@app.route("/api/health", methods=["GET"])
def health():
    """健康检查"""
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "providers": ["anthropic", "deepseek"]
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "production") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
