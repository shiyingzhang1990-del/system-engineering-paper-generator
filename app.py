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
    """获取 API 客户端"""
    if provider == "deepseek":
        return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    else:
        return anthropic.Anthropic(api_key=api_key)


def call_llm(provider, client, system_prompt, user_message, model, max_tokens):
    """统一的 LLM 调用（非流式）"""
    if provider == "deepseek":
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0.7,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )
        return response.choices[0].message.content
    else:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}]
        )
        return response.content[0].text


def call_llm_stream(provider, client, system_prompt, user_message, model, max_tokens):
    """统一的 LLM 调用（流式）"""
    if provider == "deepseek":
        stream = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0.7,
            stream=True,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    else:
        with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}]
        ) as stream:
            for text in stream.text_stream:
                yield text


def get_api_key_and_model(data):
    """从请求中提取 API key 和 model"""
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


@app.route("/api/process/staged", methods=["POST"])
def process_staged():
    """三阶段分步处理论文（流式），每阶段独立 API 调用避免输出截断"""
    data = request.get_json()
    paper_text = data.get("paper_text", "").strip()
    paper_title = data.get("paper_title", "").strip()
    provider, api_key, model = get_api_key_and_model(data)

    if not paper_text:
        return jsonify({"error": "请输入论文内容"}), 400
    if len(paper_text) < 500:
        return jsonify({"error": "论文内容过短"}), 400
    if not api_key:
        return jsonify({"error": "请提供 API Key"}), 400

    max_tok = MODEL_MAX_OUTPUT.get(model, 16384)

    def generate():
        client = get_client(provider, api_key)
        stage1_output = ""
        stage2_output = ""

        try:
            # ===== 阶段1：期刊适配诊断 + 建议全文结构 =====
            yield f"data: {json.dumps({'stage': '1_start', 'label': '阶段1/3：正在诊断期刊适配性与系统结构...'}, ensure_ascii=False)}\n\n"

            user_msg = f"论文标题：{paper_title or '（未提供）'}\n\n论文全文：\n---\n{paper_text}\n---"
            s1_buffer = ""
            for chunk in call_llm_stream(provider, client, STAGE1_PROMPT, user_msg, model, max_tok):
                s1_buffer += chunk
                yield f"data: {json.dumps({'stage': '1', 'content': chunk}, ensure_ascii=False)}\n\n"

            stage1_output = s1_buffer
            yield f"data: {json.dumps({'stage': '1_done'}, ensure_ascii=False)}\n\n"

            # ===== 阶段2：完整修改稿 =====
            yield f"data: {json.dumps({'stage': '2_start', 'label': '阶段2/3：正在重构全文...'}, ensure_ascii=False)}\n\n"

            s2_prompt = STAGE2_PROMPT.format(
                stage1_output=stage1_output[:6000],  # 截断上下文避免过长
                paper_text=paper_text
            )
            s2_buffer = ""
            for chunk in call_llm_stream(provider, client, s2_prompt, "请输出完整修改稿全文。", model, max_tok):
                s2_buffer += chunk
                yield f"data: {json.dumps({'stage': '2', 'content': chunk}, ensure_ascii=False)}\n\n"

            stage2_output = s2_buffer
            yield f"data: {json.dumps({'stage': '2_done'}, ensure_ascii=False)}\n\n"

            # ===== 阶段3：修改说明 + 风险清单与质量评分 =====
            yield f"data: {json.dumps({'stage': '3_start', 'label': '阶段3/3：正在审核修改稿并评分...'}, ensure_ascii=False)}\n\n"

            s3_prompt = STAGE3_PROMPT.format(
                stage1_output=stage1_output[:4000],
                stage2_output=stage2_output[:8000]
            )
            for chunk in call_llm_stream(provider, client, s3_prompt, "请输出修改说明、风险清单和质量评分。", model, max_tok):
                yield f"data: {json.dumps({'stage': '3', 'content': chunk}, ensure_ascii=False)}\n\n"

            yield f"data: {json.dumps({'stage': '3_done'}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'stage': 'all_done'}, ensure_ascii=False)}\n\n"

        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "Unauthorized" in error_msg:
                error_msg = "API Key 无效"
            elif "429" in error_msg:
                error_msg = "API 调用频率超限，请稍后重试"
            elif "timeout" in error_msg.lower():
                error_msg = "请求超时"
            yield f"data: {json.dumps({'stage': 'error', 'error': error_msg}, ensure_ascii=False)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "version": "3.0.0-staged",
        "providers": ["anthropic", "deepseek"]
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "production") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
