---
name: brave-search
description: "Web search using Brave Search API. Provides high-quality search results with privacy focus. Use when: user asks for web search, latest news, research information, or current events. Requires API key (already configured)."
homepage: https://brave.com/search/api/
metadata: { "openclaw": { "emoji": "🔎", "requires": { "bins": ["python3"] } } }
---

# Brave Search Skill

High-quality web search using Brave Search API.

## When to Use

✅ **USE this skill when:**

- User asks for "search"
- "Find information about..."
- "What's the latest news on..."
- "Research [topic]"
- "Look up [fact/query]"
- Current events or trending topics

## When NOT to Use

❌ **DON'T use this skill when:**

- Searching local files → use `find` or `grep`
- Code repository search → use GitHub search
- Historical data → use archives

---

## Usage

### Command Line

```bash
# Basic search (自动选择最佳搜索源)
brave-search "Python programming"

# Limit results
brave-search "machine learning" --count 3

# JSON output
brave-search "AI news" --json

# Time filter
brave-search "latest news" --freshness day
```

### 多源搜索 (当Brave API限制时自动切换)

当Brave Search API超出使用限制时，工具会自动切换到备用搜索源：
1. **Brave Search** (首选) - 高质量结果
2. **DuckDuckGo** (备用) - 隐私保护
3. **其他源** (扩展) - 可配置

```bash
# 智能搜索 - 自动处理API限制
smart-search "fund analysis" --count 5 --freshness week
```

### Python

```python
import subprocess
import json

# Search and get results
result = subprocess.run(
    ['brave-search', 'Python tutorial', '--json'],
    capture_output=True, text=True
)
data = json.loads(result.stdout)

# Access results
for item in data.get('web', {}).get('results', []):
    print(f"{item['title']}: {item['url']}")
```

### Direct API

```python
import urllib.request
import urllib.parse
import json
import gzip
import io

API_KEY = "your-api-key"

def search(query, count=5):
    url = f"https://api.search.brave.com/res/v1/web/search?q={urllib.parse.quote(query)}&count={count}"
    
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "X-Subscription-Token": API_KEY
        }
    )
    
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read())
```

---

## Features

- **Privacy-focused**: Brave doesn't track searches
- **High-quality results**: Independent index
- **Fast**: Optimized for speed
- **Free tier**: 2,000 queries/month

---

## Rate Limits & Fallback

### Brave Search API
- **Free tier**: 2,000 queries/month
- **Paid tier**: Starting at $3 per 1,000 queries

### 自动故障转移 (Auto Fallback)

当Brave API达到使用限制时，系统会自动：

1. **检测错误** - 识别 `USAGE_LIMIT_EXCEEDED` 错误
2. **切换源** - 自动切换到DuckDuckGo等备用搜索源
3. **继续服务** - 保证搜索功能可用

```python
# 示例：API限制时的自动切换
searcher = SmartSearch()
result = searcher.search("query")
# 如果Brave失败，自动使用DuckDuckGo
print(f"使用搜索源: {result['source']}")  # "DuckDuckGo"
```

### 备用搜索源

| 优先级 | 搜索源 | 类型 | 特点 |
|:---:|:---|:---|:---|
| 1 | Brave Search | API | 高质量，隐私保护 |
| 2 | DuckDuckGo | 网页 | 无API限制，隐私 |
| 3 | Google (SerpAPI) | API | 需要配置API key |

---

## Example Output

```
🔍 Brave Search: 'OpenAI GPT-5'
Found 3 results
======================================================================

1. Introducing GPT-5 | OpenAI
   URL: https://openai.com/index/introducing-gpt-5/
   We are introducing GPT‑5, our best AI system yet...

2. GPT-5 is here | OpenAI
   URL: https://openai.com/gpt-5/
   GPT‑5 produces high-quality code...

======================================================================
```

---

## Installation

```bash
# Copy to PATH
cp brave-search.py /usr/local/bin/brave-search
chmod +x /usr/local/bin/brave-search
```

---

**API Key Status:** ✅ Configured  
**Last Updated:** 2026-03-12
