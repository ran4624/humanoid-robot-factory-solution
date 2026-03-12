#!/usr/bin/env python3
"""
Brave Search CLI - Direct API access
Usage: brave-search "your query" [--count 5] [--json]
API Key is embedded for convenience.
"""
import argparse
import json
import urllib.request
import urllib.parse
import gzip
import io

# API Configuration
API_KEY = "BSAy2VgwNZvCIVvZ_GHvSxUf2awl3iI"
API_URL = "https://api.search.brave.com/res/v1/web/search"

def brave_search(query, count=5, offset=0):
    """Search using Brave Search API"""
    
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": API_KEY
    }
    
    params = {
        "q": query,
        "count": count,
        "offset": offset,
        "text_decorations": False,
        "text_snippets": True
    }
    
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    
    try:
        req = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(req, timeout=15) as response:
            # Handle gzip compression
            if response.info().get('Content-Encoding') == 'gzip':
                buf = io.BytesIO(response.read())
                with gzip.GzipFile(fileobj=buf) as f:
                    data = json.loads(f.read().decode('utf-8'))
            else:
                data = json.loads(response.read().decode('utf-8'))
            return data
    except Exception as e:
        return {"error": str(e)}

def format_results(data):
    """Format search results for display"""
    if "error" in data:
        print(f"❌ Error: {data['error']}")
        return
    
    results = data.get("web", {}).get("results", [])
    query = data.get("query", {}).get("q", "Unknown")
    
    print(f"🔍 Brave Search: '{query}'")
    print(f"Found {len(results)} results\n")
    print("="*70)
    
    for i, result in enumerate(results, 1):
        title = result.get("title", "No title")
        url = result.get("url", "No URL")
        description = result.get("description", "No description")
        
        print(f"\n{i}. {title}")
        print(f"   URL: {url}")
        if len(description) > 200:
            print(f"   {description[:200]}...")
        else:
            print(f"   {description}")
    
    print("\n" + "="*70)

def main():
    parser = argparse.ArgumentParser(description='Brave Search CLI')
    parser.add_argument('query', help='Search query')
    parser.add_argument('--count', '-n', type=int, default=5, 
                       help='Number of results (1-20)')
    parser.add_argument('--json', '-j', action='store_true', 
                       help='Output as JSON')
    
    args = parser.parse_args()
    
    # Limit count
    count = min(max(args.count, 1), 20)
    
    # Search
    results = brave_search(args.query, count)
    
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        format_results(results)

if __name__ == '__main__':
    main()
