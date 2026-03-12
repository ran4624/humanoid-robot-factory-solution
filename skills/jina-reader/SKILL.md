---
name: jina-reader
description: "Extract clean, LLM-friendly content from any URL or search the web using jina.ai Reader API. Use when: user wants to read/extract content from a webpage, summarize articles, fetch readable text from URLs, or search for latest information. NOT for: downloading files, accessing login-required pages, or sites blocking bots."
homepage: https://jina.ai/reader
metadata: { "openclaw": { "emoji": "📖", "requires": { "bins": ["curl"] } } }
---

# Jina AI Reader Skill

Extract clean, readable content from any URL or search the web.

## When to Use

✅ **USE this skill when:**

- "Extract content from this URL"
- "Read this article for me"
- "Summarize this webpage"
- "Get the text from this link"
- "Search for latest information about [topic]"
- "Fetch article content without visiting the site"
- "Convert webpage to markdown/text"

## When NOT to Use

❌ **DON'T use this skill when:**

- Downloading files (PDF, images, etc.) → use direct download tools
- Accessing login-required pages → use browser automation
- Sites with strong bot protection → use browser with stealth mode
- Real-time streaming data → use dedicated APIs
- Complex JavaScript-rendered SPAs → use browser automation

## API Endpoints

### Read URL Content

Extract clean text from any URL:

```bash
# Basic usage
curl "https://r.jina.ai/https://example.com"

# With markdown output
curl "https://r.jina.ai/https://example.com" -H "Accept: text/markdown"

# Get JSON response with metadata
curl "https://r.jina.ai/https://example.com" -H "Accept: application/json"
```

### Search the Web

Search for information and get AI-friendly results:

```bash
# Basic search
curl "https://s.jina.ai/artificial+intelligence+latest+news"

# Search with spaces (URL encoded)
curl "https://s.jina.ai/what+is+machine+learning"
```

## Usage Examples

### Extract Article Content

**User:** "Read this article https://example.com/article"

```bash
# Fetch and extract
curl -s "https://r.jina.ai/https://example.com/article"
```

### Get Markdown Format

```bash
# Get content as markdown
curl -s "https://r.jina.ai/https://github.com/jina-ai/reader" \
  -H "Accept: text/markdown"
```

### Search for Information

**User:** "What's the latest news about AI?"

```bash
# Search and summarize
curl -s "https://s.jina.ai/latest+AI+news+2024"
```

### Extract with Options

```bash
# Include images reference
curl -s "https://r.jina.ai/https://example.com" \
  -H "x-with-images-summary: true"

# Include link summaries
curl -s "https://r.jina.ai/https://example.com" \
  -H "x-with-links-summary: true"
```

## Response Formats

### Text (Default)
Plain text with title, content, and optional metadata.

### Markdown
Structured markdown with headers, lists, and links preserved.

### JSON
```json
{
  "title": "Article Title",
  "content": "Clean extracted text...",
  "url": "https://example.com",
  "timestamp": "..."
}
```

## Rate Limits

- **Read (r.jina.ai)**: Free, no API key required, 200 requests/minute
- **Search (s.jina.ai)**: Requires API key (free tier available)
- Production-ready and stable

## API Key (Optional)

For higher rate limits or search functionality:

```bash
# Get API key from https://jina.ai/reader
export JINA_API_KEY="your-api-key"

# Use with API key
curl -s "https://s.jina.ai/your+query" \
  -H "Authorization: Bearer $JINA_API_KEY"
```

## Tips

1. **URL Encoding**: Always encode URLs with special characters
2. **Redirects**: Automatically follows redirects
3. **Timeout**: Default timeout is 30 seconds
4. **Content Length**: Optimized for articles and blog posts

## Examples in Practice

### Research Assistant
```bash
# Gather information from multiple sources
curl -s "https://r.jina.ai/https://en.wikipedia.org/wiki/Artificial_intelligence"
curl -s "https://s.jina.ai/AI+applications+in+healthcare+2024"
```

### Content Summarization
```bash
# Extract main content from news article
curl -s "https://r.jina.ai/https://news.ycombinator.com/item?id=..." | head -100
```

### Web Search
```bash
# Quick factual lookup
curl -s "https://s.jina.ai/who+invented+the+telephone"
```
