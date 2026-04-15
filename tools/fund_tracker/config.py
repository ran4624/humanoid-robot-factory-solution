#!/usr/bin/env python3
"""
基金追踪与新闻收集系统
追踪五只基金：摩根美国基金、摩根亚洲股息、摩根太平洋证券、摩根中证A500、百达策略收益
"""

import json
import os
from datetime import datetime
from pathlib import Path

# 基金配置 - 汇丰平台五只基金
FUNDS = {
    "摩根美国基金": {
        "type": "QDII",
        "sector": "美股大盘成长",
        "manager": "JPMorgan",
        "key_companies": [
            {"name": "Apple", "ticker": "AAPL", "weight": 0.11},
            {"name": "Microsoft", "ticker": "MSFT", "weight": 0.10},
            {"name": "NVIDIA", "ticker": "NVDA", "weight": 0.08},
            {"name": "Amazon", "ticker": "AMZN", "weight": 0.05},
            {"name": "Meta", "ticker": "META", "weight": 0.04},
            {"name": "Google", "ticker": "GOOGL", "weight": 0.04},
            {"name": "Tesla", "ticker": "TSLA", "weight": 0.03},
            {"name": "Berkshire Hathaway", "ticker": "BRK.B", "weight": 0.03},
        ],
        "keywords": ["美股", "科技", "大盘成长", "标普500", "纳斯达克"],
        "fee": "3%",
        "alternative": "华夏纳指100 (0.8%)"
    },
    "摩根亚洲股息基金": {
        "type": "QDII",
        "sector": "亚太高股息",
        "manager": "JPMorgan",
        "key_companies": [
            {"name": "丰田汽车", "ticker": "7203.T", "weight": 0.05},
            {"name": "三菱UFJ", "ticker": "8306.T", "weight": 0.04},
            {"name": "必和必拓", "ticker": "BHP.AX", "weight": 0.04},
            {"name": "汇丰控股", "ticker": "0005.HK", "weight": 0.03},
            {"name": "友邦保险", "ticker": "1299.HK", "weight": 0.03},
            {"name": "星展银行", "ticker": "D05.SG", "weight": 0.03},
        ],
        "keywords": ["亚太", "高股息", "日本", "澳洲", "香港", "新加坡"],
        "fee": "3%",
        "alternative": "南方亚洲美元债 (1.0%)"
    },
    "摩根太平洋证券基金": {
        "type": "QDII",
        "sector": "太平洋股票",
        "manager": "JPMorgan",
        "key_companies": [
            {"name": "索尼", "ticker": "6758.T", "weight": 0.04},
            {"name": "软银", "ticker": "9984.T", "weight": 0.03},
            {"name": "三星电子", "ticker": "005930.KS", "weight": 0.03},
            {"name": "丰田汽车", "ticker": "7203.T", "weight": 0.03},
            {"name": "必和必拓", "ticker": "BHP.AX", "weight": 0.03},
        ],
        "keywords": ["太平洋", "日本", "澳洲", "韩国", "科技", "工业"],
        "fee": "3%",
        "alternative": "富达太平洋基金 (1.5%)"
    },
    "摩根中证A500指数基金": {
        "type": "QDII",
        "sector": "A股中盘指数",
        "manager": "JPMorgan",
        "key_companies": [
            {"name": "贵州茅台", "ticker": "600519.SS", "weight": 0.03},
            {"name": "宁德时代", "ticker": "300750.SZ", "weight": 0.03},
            {"name": "中国平安", "ticker": "601318.SS", "weight": 0.02},
            {"name": "招商银行", "ticker": "600036.SS", "weight": 0.02},
            {"name": "五粮液", "ticker": "000858.SZ", "weight": 0.02},
        ],
        "keywords": ["A股", "中证500", "中盘股", "指数"],
        "fee": "3%",
        "alternative": "中证500ETF联接 (0.5%)"
    },
    "百达策略收益基金": {
        "type": "QDII",
        "sector": "全球多资产",
        "manager": "Pictet",
        "key_companies": [
            {"name": "全球高股息股票组合", "ticker": "N/A", "weight": 0.45},
            {"name": "投资级债券", "ticker": "N/A", "weight": 0.35},
            {"name": "REITs/基建", "ticker": "N/A", "weight": 0.12},
            {"name": "现金", "ticker": "N/A", "weight": 0.08},
        ],
        "keywords": ["全球", "多资产", "收益", "股票", "债券", "另类"],
        "fee": "3%",
        "alternative": "自建组合 (1.0%)"
    }
}

def get_fund_info(fund_name):
    """获取基金信息"""
    return FUNDS.get(fund_name, {})

def get_all_keywords():
    """获取所有关键词"""
    keywords = set()
    for fund in FUNDS.values():
        keywords.update(fund.get("keywords", []))
    return list(keywords)

def get_all_companies():
    """获取所有关键公司"""
    companies = []
    for fund_name, fund_info in FUNDS.items():
        for company in fund_info.get("key_companies", []):
            companies.append({
                "fund": fund_name,
                **company
            })
    return companies

if __name__ == "__main__":
    # 测试输出
    print("基金追踪系统配置 - 汇丰平台五只基金")
    print("=" * 50)
    for name, info in FUNDS.items():
        print(f"\n{name}")
        print(f"  管理人: {info['manager']}")
        print(f"  类型: {info['type']}")
        print(f"  板块: {info['sector']}")
        print(f"  管理费: {info['fee']}")
        print(f"  替代方案: {info['alternative']}")
        print(f"  关键公司: {len(info['key_companies'])}家")
