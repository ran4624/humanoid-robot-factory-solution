#!/usr/bin/env python3
"""
市场数据收集器 - 使用 yfinance
获取股票行情、企业信息、市场数据
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

# 基金持仓企业映射
FUND_HOLDINGS = {
    "摩根美国基金": {
        "AAPL": {"name": "Apple", "sector": "科技", "weight": 0.11},
        "MSFT": {"name": "Microsoft", "sector": "科技", "weight": 0.10},
        "NVDA": {"name": "NVIDIA", "sector": "半导体", "weight": 0.08},
        "AMZN": {"name": "Amazon", "sector": "电商/云", "weight": 0.05},
        "META": {"name": "Meta", "sector": "社交媒体", "weight": 0.04},
        "GOOGL": {"name": "Google", "sector": "搜索/广告", "weight": 0.04},
        "TSLA": {"name": "Tesla", "sector": "电动车", "weight": 0.03},
        "BRK-B": {"name": "Berkshire Hathaway", "sector": "综合金融", "weight": 0.03},
    },
    "摩根中证A500指数基金": {
        "600519.SS": {"name": "贵州茅台", "market": "A股", "sector": "白酒", "weight": 0.03},
        "300750.SZ": {"name": "宁德时代", "market": "A股", "sector": "新能源", "weight": 0.03},
        "601318.SS": {"name": "中国平安", "market": "A股", "sector": "保险", "weight": 0.02},
        "600036.SS": {"name": "招商银行", "market": "A股", "sector": "银行", "weight": 0.02},
        "000858.SZ": {"name": "五粮液", "market": "A股", "sector": "白酒", "weight": 0.02},
    },
    "摩根亚洲股息基金": {
        "TM": {"name": "丰田汽车", "market": "日本", "sector": "汽车", "weight": 0.05, "ticker_jp": "7203.T"},
        "BHP": {"name": "必和必拓", "market": "澳洲", "sector": "矿业", "weight": 0.04, "ticker_au": "BHP.AX"},
        "HSBC": {"name": "汇丰控股", "market": "香港", "sector": "银行", "weight": 0.03, "ticker_hk": "0005.HK"},
        "AAGIY": {"name": "友邦保险", "market": "香港", "sector": "保险", "weight": 0.03, "ticker_hk": "1299.HK"},
    },
    "摩根太平洋证券基金": {
        "SONY": {"name": "索尼", "market": "日本", "sector": "科技/娱乐", "weight": 0.04, "ticker_jp": "6758.T"},
        "SFTBY": {"name": "软银", "market": "日本", "sector": "投资/电信", "weight": 0.03, "ticker_jp": "9984.T"},
        "005930.KS": {"name": "三星电子", "market": "韩国", "sector": "科技", "weight": 0.03},
        "TM": {"name": "丰田汽车", "market": "日本", "sector": "汽车", "weight": 0.03, "ticker_jp": "7203.T"},
        "BHP": {"name": "必和必拓", "market": "澳洲", "sector": "矿业", "weight": 0.03, "ticker_au": "BHP.AX"},
    }
}

# 主要指数
MARKET_INDICES = {
    "美股": {
        "^GSPC": {"name": "标普500", "description": "美股大盘指数"},
        "^IXIC": {"name": "纳斯达克", "description": "科技股指数"},
        "^DJI": {"name": "道琼斯", "description": "工业指数"},
    },
    "A股": {
        "000001.SS": {"name": "上证指数", "description": "A股主板指数"},
        "000905.SS": {"name": "中证500", "description": "中盘指数"},
        "399001.SZ": {"name": "深证成指", "description": "深圳成指"},
    },
    "亚太": {
        "^N225": {"name": "日经225", "description": "日本股市"},
        "^HSI": {"name": "恒生指数", "description": "香港股市"},
        "^AXJO": {"name": "澳洲ASX200", "description": "澳洲股市"},
        "^KS11": {"name": "韩国KOSPI", "description": "韩国股市"},
    }
}

class MarketDataCollector:
    """市场数据收集器"""
    
    def __init__(self):
        self.cache_dir = Path(__file__).parent / "data"
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "yfinance_cache.json"
        self.cache = self._load_cache()
        self.yf = None
        self._init_yfinance()
    
    def _init_yfinance(self):
        """初始化yfinance"""
        try:
            import yfinance as yf
            self.yf = yf
        except ImportError:
            print("警告: yfinance未安装，使用模拟数据")
            self.yf = None
    
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
    
    def get_stock_info(self, symbol):
        """获取股票信息"""
        if not self.yf:
            return self._get_mock_data(symbol)
        
        cache_key = f"info_{symbol}_{datetime.now().strftime('%Y%m%d')}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            ticker = self.yf.Ticker(symbol)
            info = ticker.info
            
            result = {
                "name": info.get("longName", info.get("shortName", symbol)),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "market_cap": info.get("marketCap", "N/A"),
                "pe_ratio": info.get("trailingPE", "N/A"),
                "forward_pe": info.get("forwardPE", "N/A"),
                "dividend_yield": info.get("dividendYield", "N/A"),
                "52_week_high": info.get("fiftyTwoWeekHigh", "N/A"),
                "52_week_low": info.get("fiftyTwoWeekLow", "N/A"),
                "website": info.get("website", "N/A"),
                "business_summary": info.get("longBusinessSummary", "N/A")[:200] + "..." if info.get("longBusinessSummary") else "N/A",
            }
            
            self.cache[cache_key] = result
            self._save_cache()
            return result
        except Exception as e:
            print(f"  获取{symbol}信息失败: {e}")
            return self._get_mock_data(symbol)
    
    def get_stock_price(self, symbol):
        """获取股票价格"""
        if not self.yf:
            return self._get_mock_price(symbol)
        
        cache_key = f"price_{symbol}_{datetime.now().strftime('%Y%m%d%H')}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            ticker = self.yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            
            if len(hist) >= 2:
                latest = hist.iloc[-1]
                prev = hist.iloc[-2]
                
                change = latest['Close'] - prev['Close']
                change_pct = (change / prev['Close']) * 100
                
                result = {
                    "price": round(latest['Close'], 2),
                    "change": round(change, 2),
                    "change_percent": round(change_pct, 2),
                    "volume": int(latest['Volume']),
                    "high": round(latest['High'], 2),
                    "low": round(latest['Low'], 2),
                    "date": latest.name.strftime("%Y-%m-%d")
                }
                
                self.cache[cache_key] = result
                self._save_cache()
                return result
        except Exception as e:
            print(f"  获取{symbol}价格失败: {e}")
        
        return self._get_mock_price(symbol)
    
    def get_index_data(self, symbol):
        """获取指数数据"""
        return self.get_stock_price(symbol)
    
    def _get_mock_data(self, symbol):
        """模拟数据"""
        return {
            "name": symbol,
            "sector": "N/A",
            "industry": "N/A",
            "market_cap": "N/A",
            "pe_ratio": "N/A",
            "note": "使用模拟数据 (yfinance未安装或API限制)"
        }
    
    def _get_mock_price(self, symbol):
        """模拟价格数据"""
        return {
            "price": "N/A",
            "change": "N/A",
            "change_percent": "N/A",
            "note": "使用模拟数据 (yfinance未安装或API限制)"
        }

def get_fund_company_analysis(fund_name, collector=None):
    """获取基金持仓企业分析"""
    if collector is None:
        collector = MarketDataCollector()
    
    holdings = FUND_HOLDINGS.get(fund_name, {})
    if not holdings:
        return {}
    
    print(f"  分析 {fund_name} 持仓企业...")
    
    analysis = {
        "fund_name": fund_name,
        "update_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "companies": []
    }
    
    for symbol, info in holdings.items():
        print(f"    获取 {info['name']} ({symbol})...")
        
        company_data = {
            "symbol": symbol,
            "name": info['name'],
            "sector": info.get('sector', 'N/A'),
            "weight": info.get('weight', 0),
            "price_data": collector.get_stock_price(symbol),
            "company_info": collector.get_stock_info(symbol)
        }
        
        analysis["companies"].append(company_data)
    
    return analysis

def get_market_summary(collector=None):
    """获取市场摘要"""
    if collector is None:
        collector = MarketDataCollector()
    
    summary = {
        "update_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "indices": {}
    }
    
    print("  获取市场指数...")
    
    for market, indices in MARKET_INDICES.items():
        summary["indices"][market] = {}
        for symbol, info in indices.items():
            print(f"    获取 {info['name']}...")
            price_data = collector.get_stock_price(symbol)
            summary["indices"][market][info['name']] = price_data
    
    return summary

if __name__ == "__main__":
    print("市场数据收集器测试")
    print("=" * 60)
    
    collector = MarketDataCollector()
    
    # 测试美股
    print("\n测试美股行情:")
    data = get_fund_company_analysis("摩根美国基金", collector)
    print(json.dumps(data, ensure_ascii=False, indent=2))
    
    # 测试市场摘要
    print("\n测试市场摘要:")
    summary = get_market_summary(collector)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
