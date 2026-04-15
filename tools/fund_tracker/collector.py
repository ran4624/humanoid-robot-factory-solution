#!/usr/bin/env python3
"""
基金新闻收集器 - 使用 OpenClaw web_search
收集五只基金相关的新闻、财报、行业动态
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# 添加配置
sys.path.insert(0, str(Path(__file__).parent))
from config import FUNDS

class FundNewsCollector:
    """基金新闻收集器"""
    
    def __init__(self):
        self.data_dir = Path(__file__).parent / "data"
        self.data_dir.mkdir(exist_ok=True)
        self.cache_file = self.data_dir / "news_cache.json"
        self.cache = self._load_cache()
        
    def _load_cache(self):
        """加载缓存"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def _save_cache(self):
        """保存缓存"""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
    
    def _search_web(self, query, count=3):
        """使用 OpenClaw web_search 工具"""
        # 这里返回模拟数据，实际使用时可以通过外部调用获取
        # 由于工具限制，我们在外部调用时填充真实数据
        return []
    
    def collect_news(self, fund_name):
        """收集单只基金相关新闻 - 返回搜索关键词供外部调用"""
        fund_info = FUNDS.get(fund_name, {})
        if not fund_info:
            return []
        
        # 返回搜索查询列表
        queries = []
        
        # 1. 关键公司查询 (前2家)
        companies = fund_info.get("key_companies", [])[:2]
        for company in companies:
            if company["ticker"] != "N/A":
                queries.append({
                    "type": "company",
                    "name": company["name"],
                    "ticker": company["ticker"],
                    "query": f"{company['name']} stock news"
                })
        
        # 2. 行业查询
        query_map = {
            "摩根美国基金": "US tech stocks Apple Microsoft NVIDIA news",
            "摩根亚洲股息基金": "Asia dividend stocks Toyota Mitsubishi news",
            "摩根太平洋证券基金": "Asia Pacific Sony Samsung Toyota stock",
            "摩根中证A500指数基金": "China A-share CSI 500 Kweichow Moutai CATL",
            "百达策略收益基金": "global multi-asset investment bonds stocks"
        }
        
        sector_query = query_map.get(fund_name, fund_info.get("sector", ""))
        queries.append({
            "type": "sector",
            "fund": fund_name,
            "query": sector_query
        })
        
        return queries
    
    def get_market_summary_queries(self):
        """获取市场摘要的搜索查询"""
        return [
            {"market": "us", "query": "US stock market today Dow S&P 500 Nasdaq"},
            {"market": "asia", "query": "Asian stock markets Nikkei Hang Seng today"},
            {"market": "china", "query": "China stock market Shanghai Composite today"}
        ]
    
    def generate_daily_report(self):
        """生成每日报告数据 - 返回搜索查询"""
        report = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "funds": {},
            "market_queries": self.get_market_summary_queries()
        }
        
        for fund_name in FUNDS.keys():
            report["funds"][fund_name] = {
                "queries": self.collect_news(fund_name),
            }
        
        return report

def format_news_for_report(news_items):
    """格式化新闻为markdown"""
    if not news_items:
        return "- 暂无相关新闻"
    
    lines = []
    for item in news_items[:3]:  # 只显示前3条
        title = item.get("title", "")
        url = item.get("url", "")
        snippet = item.get("snippet", "")[:60] + "..." if len(item.get("snippet", "")) > 60 else item.get("snippet", "")
        
        if url and title:
            lines.append(f"- [{title[:50]}...]({url}) - {snippet}")
        elif title:
            lines.append(f"- {title}")
    
    return "\n".join(lines) if lines else "- 暂无相关新闻"

def format_market_summary(market_data):
    """格式化市场摘要为markdown"""
    lines = []
    
    # 美股
    lines.append("### 美股市场")
    if market_data.get("us_stocks"):
        for news in market_data["us_stocks"][:2]:
            if news:
                lines.append(f"- {news[:80]}...")
    else:
        lines.append("- 数据收集中...")
    lines.append("")
    
    # 亚太
    lines.append("### 亚太市场")
    if market_data.get("asia_stocks"):
        for news in market_data["asia_stocks"][:2]:
            if news:
                lines.append(f"- {news[:80]}...")
    else:
        lines.append("- 数据收集中...")
    lines.append("")
    
    # A股
    lines.append("### A股市场")
    if market_data.get("china_stocks"):
        for news in market_data["china_stocks"][:2]:
            if news:
                lines.append(f"- {news[:80]}...")
    else:
        lines.append("- 数据收集中...")
    
    return "\n".join(lines)

if __name__ == "__main__":
    collector = FundNewsCollector()
    
    # 测试生成报告
    print("基金新闻收集器 - 搜索查询生成")
    print("=" * 50)
    
    report = collector.generate_daily_report()
    
    print(f"\n报告日期: {report['date']}")
    print(f"\n市场搜索查询:")
    for q in report['market_queries']:
        print(f"  [{q['market']}] {q['query']}")
    
    for fund_name, data in report['funds'].items():
        print(f"\n{fund_name}:")
        for q in data['queries']:
            print(f"  [{q['type']}] {q['query']}")
