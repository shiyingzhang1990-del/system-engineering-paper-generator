#!/bin/bash
# 本地开发启动脚本
echo "=== 《系统工程》实证论文重构系统 ==="
echo "启动本地开发服务器..."

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
fi

source venv/bin/activate

# 安装依赖
pip install -r requirements.txt -q

# 设置开发模式
export FLASK_ENV=development

# 启动
echo ""
echo "服务已启动：http://localhost:5000"
echo "按 Ctrl+C 停止"
echo ""

python app.py
