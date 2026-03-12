#!/bin/bash
# Web Search Skill Setup Script

echo "========================================"
echo "  Web Search Skill Setup"
echo "========================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

echo "✓ Python 3 found"

# Install DuckDuckGo search
echo ""
echo "📦 Installing duckduckgo-search..."
pip3 install -q duckduckgo-search

if [ $? -eq 0 ]; then
    echo "✓ duckduckgo-search installed"
else
    echo "⚠ Installation may have failed, but let's try anyway"
fi

# Install Tavily (optional)
echo ""
echo "📦 Installing tavily-python (optional)..."
pip3 install -q tavily-python

if [ $? -eq 0 ]; then
    echo "✓ tavily-python installed"
else
    echo "⚠ Tavily installation skipped (optional)"
fi

# Create symlink
echo ""
echo "🔗 Creating command symlink..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ln -sf "$SCRIPT_DIR/web-search.py" /usr/local/bin/web-search 2>/dev/null || echo "⚠ Could not create global symlink (may need sudo)"

echo ""
echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo ""
echo "Usage:"
echo "  python3 $SCRIPT_DIR/web-search.py 'your search query'"
echo "  web-search 'your search query' (if symlink created)"
echo ""
echo "Examples:"
echo "  web-search 'python tutorial'"
echo "  web-search 'latest AI news' --num 10"
echo "  web-search 'machine learning' --provider tavily"
echo ""
echo "Optional: Set Tavily API key for advanced search:"
echo "  export TAVILY_API_KEY='tvly-your-key'"
