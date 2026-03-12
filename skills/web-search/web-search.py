#!/usr/bin/env python3
"""
Web Search CLI - Multiple search providers
Usage: web-search <query> [--provider duckduckgo|tavily] [--num 5]
"""
import argparse
import sys
import os
import json

def search_duckduckgo(query, num_results=5):
    """Search using DuckDuckGo (free, no API key)"""
    try:
        from duckduckgo_search import DDGS
        
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))
            
        print(f"🔍 DuckDuckGo Search: '{query}'")
        print(f"Found {len(results)} results\n")
        
        for i, r in enumerate(results, 1):
            print(f"{i}. {r['title']}")
            print(f"   URL: {r['href']}")
            print(f"   {r['body'][:150]}...")
            print()
            
        return results
    except ImportError:
        print("❌ duckduckgo-search not installed. Run: pip3 install duckduckgo-search")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def search_tavily(query, num_results=5, api_key=None):
    """Search using Tavily (requires API key)"""
    if not api_key:
        api_key = os.environ.get('TAVILY_API_KEY')
    
    if not api_key:
        print("❌ Tavily API key required. Set TAVILY_API_KEY environment variable.")
        print("   Get your key at: https://tavily.com")
        return None
    
    try:
        from tavily import TavilyClient
        
        tavily = TavilyClient(api_key=api_key)
        response = tavily.search(
            query=query,
            search_depth="basic",
            max_results=num_results
        )
        
        print(f"🔍 Tavily Search: '{query}'")
        print(f"Found {len(response.get('results', []))} results\n")
        
        if response.get('answer'):
            print(f"💡 Quick Answer: {response['answer']}\n")
        
        for i, r in enumerate(response.get('results', []), 1):
            print(f"{i}. {r['title']}")
            print(f"   URL: {r['url']}")
            if r.get('content'):
                print(f"   {r['content'][:150]}...")
            print()
            
        return response
    except ImportError:
        print("❌ tavily-python not installed. Run: pip3 install tavily-python")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Web Search CLI')
    parser.add_argument('query', help='Search query')
    parser.add_argument('--provider', '-p', choices=['duckduckgo', 'tavily', 'ddg', 't'],
                      default='duckduckgo', help='Search provider')
    parser.add_argument('--num', '-n', type=int, default=5, help='Number of results')
    parser.add_argument('--json', '-j', action='store_true', help='Output as JSON')
    
    args = parser.parse_args()
    
    # Map short names
    provider_map = {
        'ddg': 'duckduckgo',
        't': 'tavily'
    }
    provider = provider_map.get(args.provider, args.provider)
    
    # Search
    if provider == 'duckduckgo':
        results = search_duckduckgo(args.query, args.num)
    elif provider == 'tavily':
        results = search_tavily(args.query, args.num)
    else:
        print(f"❌ Unknown provider: {provider}")
        sys.exit(1)
    
    # JSON output
    if args.json and results:
        print(json.dumps(results, indent=2, default=str))

if __name__ == '__main__':
    main()
