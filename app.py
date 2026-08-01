import os
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import anthropic

from system_prompt import SYSTEM_PROMPT

app = Flask(__name__)
CORS(app)
app.secret_key = os.environ.get("SECRET_KEY", "system-engineering-paper-generator-2024")

# API Key 配置
# 优先级：环境变量 > 用户传入（前端表单）
ADMIN_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


def build_user_prompt(paper_text, paper_title=""):
    """构建用户提示词"""
    return f"""请处理以下论文。严格按照系统指令输出五部分内容。

论文标题：{paper_title or "（未提供）"}

论文全文：
---
{paper_text}
---"""


def process_paper_sync(paper_text, api_key, model, paper_title):
    """同步处理论文（非流式）"""
    client = anthropic.Anthropic(api_key=api_key)

    max_tokens = 16000  # 长输出模式

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": build_user_prompt(paper_text, paper_title)
            }
        ]
    )

    return response.content[0].text


def process_paper_stream(paper_text, api_key, model, paper_title):
    """流式处理论文"""
    client = anthropic.Anthropic(api_key=api_key)

    max_tokens = 16000

    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": build_user_prompt(paper_text, paper_title)
            }
        ]
    ) as stream:
        for text in stream.text_stream:
            yield text


@app.route("/")
def index():
    """主页"""
    has_admin_key = bool(ADMIN_API_KEY)
    return render_template("index.html", has_admin_key=has_admin_key)


@app.route("/api/process", methods=["POST"])
def process():
    """处理论文（非流式）"""
    try:
        data = request.get_json()
        paper_text = data.get("paper_text", "").strip()
        paper_title = data.get("paper_title", "").strip()
        api_key = data.get("api_key", "").strip() or ADMIN_API_KEY
        model = data.get("model", "claude-sonnet-4-6")

        if not paper_text:
            return jsonify({"error": "请输入论文内容"}), 400
        if len(paper_text) < 500:
            return jsonify({"error": "论文内容过短（至少500字），请检查后重新提交"}), 400
        if not api_key:
            return jsonify({"error": "请提供 Anthropic API Key，或在环境变量中配置 ANTHROPIC_API_KEY"}), 400

        result = process_paper_sync(paper_text, api_key, model, paper_title)

        return jsonify({
            "success": True,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })

    except anthropic.APIError as e:
        return jsonify({"error": f"Anthropic API 错误: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"处理失败: {str(e)}"}), 500


@app.route("/api/process/stream", methods=["POST"])
def process_stream():
    """处理论文（流式）"""
    data = request.get_json()
    paper_text = data.get("paper_text", "").strip()
    paper_title = data.get("paper_title", "").strip()
    api_key = data.get("api_key", "").strip() or ADMIN_API_KEY
    model = data.get("model", "claude-sonnet-4-6")

    if not paper_text:
        return jsonify({"error": "请输入论文内容"}), 400
    if len(paper_text) < 500:
        return jsonify({"error": "论文内容过短（至少500字）"}), 400
    if not api_key:
        return jsonify({"error": "请提供 Anthropic API Key"}), 400

    def generate():
        try:
            for chunk in process_paper_stream(paper_text, api_key, model, paper_title):
                yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except anthropic.APIError as e:
            yield f"data: {json.dumps({'error': f'API 错误: {str(e)}'}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': f'处理失败: {str(e)}'}, ensure_ascii=False)}\n\n"

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
    """健康检查"""
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "production") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
