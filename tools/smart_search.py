#!/usr/bin/env python3
"""
智能搜索工具 - 多源搜索聚合
当主搜索API限制时自动切换到备用源
"""

import sys
import json
import urllib.request
import urllib.parse
import subprocess
from datetime import datetime

# 搜索源配置
SEARCH_SOURCES = {
    "brave": {
        "name": "Brave Search",
        "enabled": True,
        "priority": 1,
        "api_key": "BSA8Q5x_QeqzJ9e6C1Dq7cQp4E8r9Tj",  # 从环境变量或配置读取
        "endpoint": "https://api.search.brave.com/res/v1/web/search"
    },
    "ddg": {
        "name": "DuckDuckGo",
        "enabled": True,
        "priority": 2,
        "type": "scrape"  # 使用ddgr工具
    },
    "google": {
        "name": "Google (via SerpAPI)",
        "enabled": False,  # 需要API key
        "priority": 3
    }
}

class SmartSearch:
    """智能搜索类"""
    
    def __init__(self):
        self.current_source = None
        self.results = []
        
    def search(self, query, count=5, freshness=None):
        """
        智能搜索 - 自动切换源
        
        Args:
            query: 搜索关键词
            count: 结果数量
            freshness: 时间筛选 (day, week, month)
        """
        # 按优先级尝试搜索源
        sources = sorted(SEARCH_SOURCES.items(), 
                        key=lambda x: x[1].get('priority', 99))
        
        for source_id, source_config in sources:
            if not source_config.get('enabled', False):
                continue
                
            try:
                print(f"尝试使用 {source_config['name']}...", file=sys.stderr)
                
                if source_id == "brave":
                    results = self._search_brave(query, count, freshness)
                elif source_id == "ddg":
                    results = self._search_ddg(query, count, freshness)
                else:
                    continue
                
                if results:
                    self.current_source = source_config['name']
                    self.results = results
                    return {
                        "success": True,
                        "source": source_config['name'],
                        "query": query,
                        "count": len(results),
                        "results": results
                    }
                    
            except Exception as e:
                print(f"{source_config['name']} 失败: {e}", file=sys.stderr)
                continue
        
        # 所有源都失败
        return {
            "success": False,
            "error": "所有搜索源均不可用",
            "query": query
        }
    
    def _search_brave(self, query, count, freshness):
        """Brave Search API"""
        api_key = SEARCH_SOURCES["brave"]["api_key"]
        endpoint = SEARCH_SOURCES["brave"]["endpoint"]
        
        # 构建URL
        params = {
            "q": query,
            "count": count
        }
        
        if freshness:
            # Brave freshness: pd (past day), pw (past week), pm (past month)
            freshness_map = {
                "day": "pd",
                "week": "pw",
                "month": "pm"
            }
            if freshness in freshness_map:
                params["freshness"] = freshness_map[freshness]
        
        url = f"{endpoint}?{urllib.parse.urlencode(params)}"
        
        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": api_key
            }
        )
        
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read())
            
            # 检查是否超出限制
            if "error" in data:
                error_code = data.get("error", {}).get("code", "")
                if "USAGE_LIMIT_EXCEEDED" in str(error_code):
                    raise Exception("API使用限制 exceeded")
            
            # 解析结果
            results = []
            for item in data.get("web", {}).get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "description": item.get("description", ""),
                    "published": item.get("age", "")
                })
            
            return results
    
    def _search_ddg(self, query, count, freshness):
        """DuckDuckGo搜索 (使用ddgr)"""
        try:
            # 检查ddgr是否安装
            subprocess.run(["ddgr", "--version"], 
                         capture_output=True, check=True)
        except:
            # 尝试安装
            print("安装 ddgr...", file=sys.stderr)
            subprocess.run(["pip", "install", "ddgr"], 
                         capture_output=True)
        
        # 构建命令
        cmd = ["ddgr", "--json", "-n", str(count)]
        
        if freshness:
            # ddgr时间筛选
            freshness_map = {
                "day": "d",
                "week": "w",
                "month": "m"
            }
            if freshness in freshness_map:
                cmd.extend(["-t", freshness_map[freshness]])
        
        cmd.append(query)
        
        # 执行搜索
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            raise Exception(f"ddgr失败: {result.stderr}")
        
        # 解析结果
        results = []
        try:
            data = json.loads(result.stdout)
            for item in data:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "description": item.get("abstract", ""),
                    "published": ""
                })
        except:
            # 如果JSON解析失败，尝试文本解析
            lines = result.stdout.strip().split("\n")
            for line in lines:
                if line.startswith("http"):
                    results.append({
                        "title": line,
                        "url": line,
                        "description": "",
                        "published": ""
                    })
        
        return results
    
    def format_output(self, data, format_type="text"):
        """格式化输出"""
        if not data.get("success"):
            return f"搜索失败: {data.get('error', '未知错误')}"
        
        if format_type == "json":
            return json.dumps(data, ensure_ascii=False, indent=2)
        
        # 文本格式
        lines = [
            f"🔍 {data['source']}: '{data['query']}'",
            f"找到 {data['count']} 条结果",
            "=" * 70,
            ""
        ]
        
        for i, item in enumerate(data['results'], 1):
            lines.extend([
                f"{i}. {item['title']}",
                f"   URL: {item['url']}",
            ])
            if item.get('description'):
                lines.append(f"   {item['description'][:150]}...")
            if item.get('published'):
                lines.append(f"   发布时间: {item['published']}")
            lines.append("")
        
        return "\n".join(lines)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Smart Search Tool')
    parser.add_argument('query', help='Search query')
    parser.add_argument('-c', '--count', type=int, default=5, help='Number of results')
    parser.add_argument('-f', '--freshness', choices=['day', 'week', 'month'], 
                       help='Time filter')
    parser.add_argument('-j', '--json', action='store_true', help='JSON output')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 1.0')
    
    args = parser.parse_args()
    
    # 执行搜索
    searcher = SmartSearch()
    result = searcher.search(args.query, args.count, args.freshness)
    
    # 输出结果
    output_format = "json" if args.json else "text"
    print(searcher.format_output(result, output_format))

if __name__ == '__main__':
    main()
