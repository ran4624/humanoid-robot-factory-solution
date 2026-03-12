#!/bin/bash
# Jina AI Reader CLI wrapper
# Usage: jina-read <URL> or jina-search <query>

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

show_help() {
    echo "Jina AI Reader CLI"
    echo ""
    echo "Usage:"
    echo "  jina-read <URL>           Extract content from URL"
    echo "  jina-search <query>       Search the web"
    echo ""
    echo "Options:"
    echo "  -m, --markdown           Output as markdown"
    echo "  -j, --json               Output as JSON"
    echo "  -h, --help               Show this help"
    echo ""
    echo "Examples:"
    echo "  jina-read https://example.com"
    echo "  jina-read https://github.com/jina-ai/reader --markdown"
    echo "  jina-search 'latest AI news'"
}

# Parse arguments
FORMAT=""
URL=""
QUERY=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--markdown)
            FORMAT="markdown"
            shift
            ;;
        -j|--json)
            FORMAT="json"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            if [ -z "$URL" ] && [[ $1 == http* ]]; then
                URL="$1"
            elif [ -z "$QUERY" ]; then
                QUERY="$1"
            fi
            shift
            ;;
    esac
done

# Build headers
HEADERS=""
if [ "$FORMAT" == "markdown" ]; then
    HEADERS="-H 'Accept: text/markdown'"
elif [ "$FORMAT" == "json" ]; then
    HEADERS="-H 'Accept: application/json'"
fi

# Execute command
if [ -n "$URL" ]; then
    # Read mode
    ENCODED_URL=$(echo "$URL" | sed 's/ /%20/g')
    echo -e "${GREEN}📖 Reading: $URL${NC}"
    echo ""
    eval "curl -s \"https://r.jina.ai/$ENCODED_URL\" $HEADERS"
elif [ -n "$QUERY" ]; then
    # Search mode
    ENCODED_QUERY=$(echo "$QUERY" | sed 's/ /+/g')
    echo -e "${GREEN}🔍 Searching: $QUERY${NC}"
    echo ""
    eval "curl -s \"https://s.jina.ai/$ENCODED_QUERY\" $HEADERS"
else
    show_help
    exit 1
fi
