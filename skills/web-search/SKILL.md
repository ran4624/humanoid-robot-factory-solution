---
name: web-search
description: "Search the web using multiple free and paid providers. Supports DuckDuckGo (free, no API key), Tavily (LLM-optimized), Google Custom Search, Bing Search, and SearXNG (self-hosted). Use when: user asks for web search, latest information, news, facts, or research. NOT for: searching within documents, file content search, or private data search."
homepage: https://github.com/duckduckgo/duckduckgo
metadata: { "openclaw": { "emoji": "🔍", "requires": { "bins": ["curl", "python3"] } } }
---

# Web Search Skill

Multiple search providers for different use cases and budgets.

## When to Use

✅ **USE this skill when:**

- "Search for [topic]"
- "What's the latest news about [subject]?"
- "Find information about [query]"
- "Research [topic]"
- "Look up [fact/definition]"
- "Get current events"

## When NOT to Use

❌ **DON'T use this skill when:**

- Searching within local documents → use `find` or `grep`
- Searching code in repositories → use GitHub search
- Private/internal data search → use internal tools
- Historical archives → use Wayback Machine

---

## Search Providers

### 1. DuckDuckGo (Recommended - Free)

**Pros:** Free, no API key, privacy-focused, good results  
**Cons:** Rate limited, occasional blocks  
**Best for:** General searches, quick lookups

```bash
# Install ddgr (DuckDuckGo CLI)
pip3 install ddgr

# Search
ddgr --num 5 "artificial intelligence news"

# JSON output
ddgr --json --num 3 "machine learning"
```

**Python Alternative:**
```python
# pip install duckduckgo-search
from duckduckgo_search import DDGS

with DDGS() as ddgs:
    results = ddgs.text("python programming", max_results=5)
    for r in results:
        print(f"{r['title']}: {r['href']}")
```

### 2. Tavily (LLM-Optimized)

**Pros:** Optimized for LLMs, clean text extraction, includes sources  
**Cons:** Requires API key (1000 free requests/month)  
**Best for:** Research, RAG applications, content extraction  
**Signup:** https://tavily.com

```bash
# Using curl
export TAVILY_API_KEY="tvly-your-key"

curl -X POST "https://api.tavily.com/search" \
  -H "Content-Type: application/json" \
  -d '{
    "api_key": "'"$TAVILY_API_KEY"'",
    "query": "latest AI breakthroughs 2024",
    "search_depth": "advanced",
    "include_answer": true,
    "max_results": 5
  }'
```

**Python:**
```python
# pip install tavily-python
from tavily import TavilyClient

tavily = TavilyClient(api_key="tvly-your-key")
results = tavily.search(
    query="latest AI breakthroughs",
    search_depth="advanced",
    max_results=5
)
```

### 3. Google Custom Search

**Pros:** Google's index, reliable results  
**Cons:** Requires API key + Search Engine ID, 100 queries/day free  
**Setup:** https://developers.google.com/custom-search/v1/overview

```bash
export GOOGLE_API_KEY="your-key"
export GOOGLE_CX="your-search-engine-id"

# Search
curl "https://www.googleapis.com/customsearch/v1?q=python+tutorial&key=$GOOGLE_API_KEY&cx=$GOOGLE_CX&num=5"
```

### 4. Bing Search API (Azure)

**Pros:** Microsoft's index, good for enterprise  
**Cons:** Requires Azure subscription, free tier limited  
**Signup:** https://azure.microsoft.com/services/cognitive-services/bing-web-search-api/

```bash
export BING_API_KEY="your-key"

curl -H "Ocp-Apim-Subscription-Key: $BING_API_KEY" \
  "https://api.bing.microsoft.com/v7.0/search?q=python+tutorial&count=5"
```

### 5. SearXNG (Self-Hosted)

**Pros:** Privacy-focused, aggregates multiple engines, fully self-hosted  
**Cons:** Requires setup and hosting  
**Best for:** Privacy-conscious users, organizations  
**Docker Setup:**

```bash
# Run SearXNG instance
docker run -d --name searxng \
  -p 8080:8080 \
  -v searxng-data:/etc/searx \
  searxng/searxng

# Search locally
curl "http://localhost:8080/search?q=python&format=json"
```

### 6. Jina AI Search

**Pros:** Clean results, LLM-friendly format  
**Cons:** Requires API key for search (reading URLs is free)  
**API Key:** https://jina.ai/reader

```bash
# Search (requires API key)
curl "https://s.jina.ai/artificial+intelligence" \
  -H "Authorization: Bearer your-api-key"
```

### 7. SerpAPI (Aggregator)

**Pros:** Multiple engines (Google, Bing, Yahoo, etc.), structured data  
**Cons:** Paid (100 free searches/month)  
**Signup:** https://serpapi.com

```bash
export SERP_API_KEY="your-key"

curl "https://serpapi.com/search?q=python+tutorial&api_key=$SERP_API_KEY&engine=google&num=5"
```

---

## Quick Setup Guide

### Option 1: DuckDuckGo (Easiest, Free)

```bash
# One-time setup
pip3 install -q duckduckgo-search

# Ready to use!
```

### Option 2: Tavily (Best for LLMs)

```bash
# 1. Get API key from https://tavily.com
# 2. Set environment variable
export TAVILY_API_KEY="tvly-your-key"

# 3. Install Python package
pip3 install -q tavily-python
```

### Option 3: SearXNG (Self-Hosted)

```bash
# Run with Docker
docker run -d --name searxng -p 8080:8080 searxng/searxng

# Access at http://localhost:8080
```

---

## Usage Examples

### Basic Search (DuckDuckGo)

```bash
# Command line
ddgr --num 3 "latest AI news"

# Python
python3 << 'EOF'
from duckduckgo_search import DDGS

with DDGS() as ddgs:
    for r in ddgs.text("python tutorial", max_results=3):
        print(f"{r['title']}\n{r['href']}\n{r['body'][:100]}...\n")
EOF
```

### Research Search (Tavily)

```python
from tavily import TavilyClient

tavily = TavilyClient(api_key="your-key")
response = tavily.search(
    query="climate change latest research 2024",
    search_depth="advanced",
    include_answer=True,
    max_results=5
)

print(response['answer'])
for result in response['results']:
    print(f"- {result['title']}: {result['url']}")
```

### News Search

```bash
# DuckDuckGo news
ddgr --num 5 --news "technology"

# Or filter by time
ddgr --num 5 "AI breakthroughs 2024"
```

---

## Comparison Table

| Provider | Free Tier | API Key | Best For | Rate Limit |
|----------|-----------|---------|----------|------------|
| **DuckDuckGo** | ✅ Unlimited | ❌ No | General search | ~100/day IP |
| **Tavily** | 1000/mo | ✅ Yes | LLM/RAG | 1000/mo free |
| **Google** | 100/day | ✅ Yes | Reliability | 100/day free |
| **Bing** | 1000/mo | ✅ Yes | Enterprise | 1000/mo free |
| **SearXNG** | ✅ Unlimited | ❌ No | Privacy | Self-hosted |
| **Jina Search** | Limited | ✅ Yes | Clean format | 200/mo free |
| **SerpAPI** | 100/mo | ✅ Yes | Multiple engines | 100/mo free |

---

## Recommendations

| Use Case | Recommended Provider |
|----------|---------------------|
| Quick lookups, general search | DuckDuckGo |
| Research, RAG applications | Tavily |
| Enterprise/Production | Google or Bing |
| Privacy-focused | SearXNG (self-hosted) |
| Multiple engine comparison | SerpAPI |

---

## Troubleshooting

### DuckDuckGo Rate Limit
```bash
# If you hit rate limits, wait a few minutes
# Or use a different IP/proxy
```

### Tavily API Key Issues
```bash
# Verify key is set
echo $TAVILY_API_KEY

# Test with curl
curl -X POST "https://api.tavily.com/search" \
  -H "Content-Type: application/json" \
  -d '{"api_key": "'"$TAVILY_API_KEY"'", "query": "test"}'
```

---

## Rate Limits Summary

- **DuckDuckGo**: ~100 requests/day per IP (unofficial)
- **Tavily**: 1000 requests/month (free)
- **Google**: 100 queries/day (free tier)
- **Bing**: 1000 transactions/month (free tier)
- **SerpAPI**: 100 searches/month (free)

---

**Last Updated:** 2026-03-12
