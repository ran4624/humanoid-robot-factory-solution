#!/bin/bash
# Jina AI Reader 演示脚本

echo "========================================"
echo "  Jina AI Reader 功能演示"
echo "========================================"
echo ""

# 演示 1: 提取网页内容
echo "📖 演示 1: 提取网页内容"
echo "URL: https://example.com"
echo "---"
curl -s "https://r.jina.ai/https://example.com" | head -10
echo ""
echo ""

# 演示 2: 提取 GitHub 页面
echo "📖 演示 2: 提取 GitHub README"
echo "URL: https://github.com/jina-ai/reader"
echo "---"
curl -s "https://r.jina.ai/https://github.com/jina-ai/reader" | head -15
echo ""
echo ""

# 演示 3: Markdown 格式输出
echo "📖 演示 3: Markdown 格式输出"
echo "URL: https://example.com"
echo "---"
curl -s "https://r.jina.ai/https://example.com" -H "Accept: text/markdown" | head -10
echo ""
echo ""

# 演示 4: JSON 格式输出
echo "📖 演示 4: JSON 格式输出"
echo "URL: https://example.com"
echo "---"
curl -s "https://r.jina.ai/https://example.com" -H "Accept: application/json" | head -5
echo ""
echo ""

echo "========================================"
echo "  演示完成!"
echo "========================================"
echo ""
echo "使用方法:"
echo "  curl 'https://r.jina.ai/https://your-url.com'"
echo ""
echo "搜索功能 (需要 API key):"
echo "  curl 'https://s.jina.ai/your+search+query'"
