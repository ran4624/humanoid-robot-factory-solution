#!/bin/bash
# 基金追踪日报生成脚本
# 每天中午12点执行

set -e

WORKSPACE="/root/.openclaw/workspace"
REPORTS_DIR="$WORKSPACE/reports/fund_daily"
DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M)

echo "========================================"
echo "基金追踪日报生成"
echo "日期: $DATE"
echo "时间: $TIME"
echo "========================================"

# 创建报告目录
mkdir -p "$REPORTS_DIR"

# 生成报告
cd "$WORKSPACE"

# 使用Python生成报告内容
python3 << 'PYTHON_SCRIPT'
import sys
sys.path.insert(0, '/root/.openclaw/workspace/tools/fund_tracker')

from config import FUNDS
from datetime import datetime

print(f"# 基金追踪日报 - {datetime.now().strftime('%Y年%m月%d日')}")
print("\n## 📊 今日概览\n")

for fund_name, fund_info in FUNDS.items():
    print(f"### {fund_name}")
    print(f"- 类型: {fund_info['type']}")
    print(f"- 板块: {fund_info['sector']}")
    print(f"- 关键公司: {', '.join([c['name'] for c in fund_info['key_companies'][:3]])}")
    print()

PYTHON_SCRIPT

echo ""
echo "报告生成完成"
echo "========================================"
