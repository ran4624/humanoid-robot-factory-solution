#!/usr/bin/env python3
"""
基金追踪日报生成器 - 企业分析版
重点分析基金持仓企业的行情、基本面、动态
"""

import sys
import os
from datetime import datetime
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))
from config import FUNDS
from market_data import get_fund_company_analysis, get_market_summary, MarketDataCollector, FUND_HOLDINGS

class FundDailyReport:
    """基金日报生成器 - 企业分析版"""
    
    def __init__(self):
        self.date = datetime.now().strftime("%Y-%m-%d")
        self.report_lines = []
        self.collector = MarketDataCollector()
        
    def add_header(self):
        """添加报告头部"""
        self.report_lines.extend([
            f"# 汇丰基金追踪日报 - {self.date}",
            "",
            "> **追踪对象**: 汇丰平台五只代客境外理财基金",
            "> **报告重点**: 持仓企业分析、费率优化、市场动态",
            "> **核心发现**: 五只基金费率3%，替代方案可节省50-80%费用",
            "",
            "## 📊 今日概览",
            "",
            "| 基金名称 | 管理人 | 板块 | 费率 | 替代费率 | 年节省(100万) |",
            "|:---|:---|:---|:---:|:---:|:---:|",
        ])
        
        # 节省计算
        savings = {
            "摩根美国基金": "2.2万",
            "摩根亚洲股息基金": "2.0万",
            "摩根太平洋证券基金": "1.5万",
            "摩根中证A500指数基金": "2.5万",
            "百达策略收益基金": "2.0万"
        }
        
        for fund_name, fund_info in FUNDS.items():
            alt_fee = fund_info['alternative'].split('(')[1].replace(')', '') if '(' in fund_info['alternative'] else '1.0%'
            self.report_lines.append(
                f"| {fund_name} | {fund_info['manager']} | {fund_info['sector']} | {fund_info['fee']} | {alt_fee} | {savings.get(fund_name, 'N/A')} |"
            )
        
        self.report_lines.extend([
            "",
            "**⚠️ 费用警报**: 当前费率是正常水平的2-6倍！",
            "",
            "**💰 总节省**: 100万投资5年可节省 **7.5-12.5万**",
            "",
        ])
    
    def add_market_summary(self):
        """添加市场摘要"""
        print("\n📈 正在获取市场数据...")
        summary = get_market_summary(self.collector)
        
        self.report_lines.extend([
            "## 📈 全球市场概览",
            "",
            f"*数据更新时间: {summary['update_time']}*",
            "",
        ])
        
        for market, indices in summary['indices'].items():
            self.report_lines.append(f"### {market}")
            for name, data in indices.items():
                if 'price' in data and data['price'] != 'N/A':
                    change_str = f"{data['change']} ({data['change_percent']}%)" if 'change' in data else "N/A"
                    # 添加涨跌颜色标记
                    try:
                        change_val = float(data.get('change_percent', 0))
                        if change_val > 0:
                            emoji = "🟢"
                        elif change_val < 0:
                            emoji = "🔴"
                        else:
                            emoji = "⚪"
                    except:
                        emoji = ""
                    self.report_lines.append(f"- {emoji} **{name}**: {data['price']} | 涨跌: {change_str}")
                else:
                    self.report_lines.append(f"- **{name}**: 数据获取中...")
            self.report_lines.append("")
    
    def add_fund_company_analysis(self, fund_name):
        """添加基金持仓企业分析"""
        print(f"\n📊 正在分析 {fund_name}...")
        
        fund_info = FUNDS.get(fund_name, {})
        analysis = get_fund_company_analysis(fund_name, self.collector)
        
        self.report_lines.extend([
            f"## {fund_name}",
            "",
            f"**管理人**: {fund_info.get('manager', 'N/A')} | **类型**: {fund_info.get('type', 'N/A')} | **板块**: {fund_info.get('sector', 'N/A')}",
            f"**当前费率**: {fund_info.get('fee', 'N/A')} | **替代方案**: {fund_info.get('alternative', 'N/A')}",
            "",
            "### 持仓企业分析",
            "",
        ])
        
        # 企业表格
        self.report_lines.extend([
            "| 企业 | 代码 | 行业 | 权重 | 最新价 | 涨跌 | 市值/PE |",
            "|:---|:---:|:---:|:---:|:---:|:---:|:---:|"
        ])
        
        for company in analysis.get('companies', []):
            name = company['name']
            symbol = company['symbol']
            sector = company['sector']
            weight = f"{company['weight']*100:.0f}%" if company['weight'] else "N/A"
            
            price_data = company.get('price_data', {})
            info = company.get('company_info', {})
            
            price = price_data.get('price', 'N/A')
            change = price_data.get('change_percent', 'N/A')
            if change != 'N/A':
                change = f"{change}%"
            
            market_cap = info.get('market_cap', 'N/A')
            pe = info.get('pe_ratio', 'N/A')
            market_pe = f"{market_cap/1e9:.0f}B/PE{pe}" if isinstance(market_cap, (int, float)) and market_cap else "N/A"
            
            self.report_lines.append(
                f"| {name} | {symbol} | {sector} | {weight} | {price} | {change} | {market_pe} |"
            )
        
        self.report_lines.extend([
            "",
            "### 重点企业动态",
            "",
        ])
        
        # 重点企业详细信息
        for company in analysis.get('companies', [])[:3]:  # 前3大持仓
            name = company['name']
            info = company.get('company_info', {})
            price = company.get('price_data', {})
            
            self.report_lines.extend([
                f"**{name}** ({company['symbol']})",
                f"- 行业: {info.get('sector', 'N/A')} / {info.get('industry', 'N/A')}",
            ])
            
            if price.get('price') != 'N/A':
                self.report_lines.append(f"- 行情: {price['price']} | 涨跌: {price.get('change', 'N/A')} ({price.get('change_percent', 'N/A')}%)")
            
            if info.get('pe_ratio') != 'N/A':
                self.report_lines.append(f"- 估值: PE {info.get('pe_ratio', 'N/A')} | 前瞻PE {info.get('forward_pe', 'N/A')}")
            
            if info.get('business_summary') and info['business_summary'] != 'N/A':
                summary = info['business_summary'][:150] + "..." if len(info['business_summary']) > 150 else info['business_summary']
                self.report_lines.append(f"- 业务: {summary}")
            
            self.report_lines.append("")
        
        # 费率对比
        self.report_lines.extend([
            "### 费率优化建议",
            "",
            f"| 渠道 | 管理费 | 5年费用(100万) | 节省 |",
            f"|:---|:---:|:---:|:---:|"
        ])
        
        if fund_name == "摩根美国基金":
            self.report_lines.append(f"| 汇丰代购 | 3.0% | 15万 | - |")
            self.report_lines.append(f"| 华夏纳指100 | 0.6% | 3万 | **12万** |")
        elif fund_name == "摩根中证A500指数基金":
            self.report_lines.append(f"| 汇丰代购 | 3.0% | 15万 | - |")
            self.report_lines.append(f"| 南方中证500ETF | 0.6% | 3万 | **12万** |")
        elif fund_name == "摩根亚洲股息基金":
            self.report_lines.append(f"| 汇丰代购 | 3.0% | 15万 | - |")
            self.report_lines.append(f"| 南方亚洲美元债 | 1.0% | 5万 | **10万** |")
        elif fund_name == "摩根太平洋证券基金":
            self.report_lines.append(f"| 汇丰代购 | 3.0% | 15万 | - |")
            self.report_lines.append(f"| 富达太平洋基金 | 1.5% | 7.5万 | **7.5万** |")
        elif fund_name == "百达策略收益基金":
            self.report_lines.append(f"| 汇丰代购 | 3.0% | 15万 | - |")
            self.report_lines.append(f"| 自建组合 | 0.8% | 4万 | **11万** |")
        else:
            self.report_lines.append(f"| 汇丰代购 | 3.0% | 15万 | - |")
            self.report_lines.append(f"| 替代方案 | 0.5-1.5% | 2.5-7.5万 | **7.5-12.5万** |")
        
        self.report_lines.extend([
            "",
            "---",
            "",
        ])
    
    def add_summary(self):
        """添加总结"""
        self.report_lines.extend([
            "## 💡 优化行动指南",
            "",
            "### 立即行动清单",
            "1. [ ] **联系汇丰客户经理**，获取五只基金的完整费用明细（申购费、赎回费、管理费、托管费）",
            "2. [ ] **开通低费率账户**：",
            "   - 天天基金/蚂蚁财富（国内基金，费率通常0.6%）",
            "   - 华夏/南方/易方达直销（QDII基金，费率0.8-1.0%）",
            "   - 富达/Vanguard（海外基金，费率0.5-1.0%）",
            "3. [ ] **制定分批转移计划**，建议每季度转移1只基金，避免集中赎回损失",
            "4. [ ] **关注汇率波动**，选择人民币升值时换汇，降低QDII成本",
            "",
            "### 优先替换顺序（按节省金额）",
            "| 优先级 | 基金 | 当前费率 | 最佳替代 | 替代费率 | 年节省 |",
            "|:---:|:---|:---:|:---|:---:|:---:|",
            "| 1 | **摩根中证A500** | 3.0% | 南方中证500ETF | 0.6% | **2.4万** |",
            "| 2 | **摩根美国基金** | 3.0% | 华夏纳指100 | 0.6% | **2.4万** |",
            "| 3 | **摩根亚洲股息** | 3.0% | 南方亚洲美元债 | 1.0% | **2.0万** |",
            "| 4 | **百达策略收益** | 3.0% | 自建组合 | 0.8% | **2.2万** |",
            "| 5 | **摩根太平洋证券** | 3.0% | 富达太平洋 | 1.5% | **1.5万** |",
            "",
            "### 预期节省测算",
            "| 投资金额 | 年节省 | 5年节省 | 10年节省 |",
            "|:---:|:---:|:---:|:---:|",
            "| 50万 | 1.0-1.25万 | 5-6.25万 | 10-12.5万 |",
            "| **100万** | **2.0-2.5万** | **10-12.5万** | **20-25万** |",
            "| 200万 | 4-5万 | 20-25万 | 40-50万 |",
            "| 500万 | 10-12.5万 | 50-62.5万 | 100-125万 |",
            "",
            "> 💡 **复利效应**: 节省的费用再投资，10年差距可达20-25万！",
            "",
            "---",
            "",
            "## 📚 替代方案详解",
            "",
            "### 1️⃣ 摩根美国基金 → 华夏纳指100",
            "| 对比项 | 汇丰代购 | 华夏纳指100 |",
            "|:---|:---|:---|"
        ])
        
        self.report_lines.extend([
            "| 产品 | 摩根美国基金 | 华夏纳指100ETF联接A (000834) |",
            "| 管理费率 | **3.0%** | **0.5%** |",
            "| 托管费率 | 约0.2% | 0.1% |",
            "| 总费率 | **3.2%** | **0.6%** |",
            "| 业绩基准 | 美股大盘 | 纳斯达克100指数 |",
            "| 优势 | 汇丰渠道便利 | 费率低80%、流动性好、规模超100亿 |",
            "",
            "### 2️⃣ 摩根中证A500 → 南方中证500ETF",
            "| 对比项 | 汇丰代购 | 南方中证500ETF |",
            "|:---|:---|:---|"
        ])
        
        self.report_lines.extend([
            "| 产品 | 摩根中证A500 | 南方中证500ETF联接A (160119) |",
            "| 管理费率 | **3.0%** | **0.5%** |",
            "| 托管费率 | 约0.2% | 0.1% |",
            "| 总费率 | **3.2%** | **0.6%** |",
            "| 优势 | 海外配置 | 国内最大中证500ETF，规模超500亿，费率最低 |",
            "",
            "### 3️⃣ 摩根亚洲股息 → 南方亚洲美元债",
            "| 对比项 | 汇丰代购 | 南方亚洲美元债 |",
            "|:---|:---|:---|"
        ])
        
        self.report_lines.extend([
            "| 产品 | 摩根亚洲股息 | 南方亚洲美元债A (002400) |",
            "| 管理费率 | **3.0%** | **0.8%** |",
            "| 托管费率 | 约0.2% | 0.22% |",
            "| 总费率 | **3.2%** | **1.02%** |",
            "| 优势 | 汇丰渠道 | 专注亚洲美元债券，费率低68% |",
            "",
            "### 4️⃣ 百达策略收益 → 自建组合",
            "**建议配置**:",
            "- 40% 全球高股息股票基金 (费率0.5-0.8%)",
            "- 35% 投资级债券基金 (费率0.3-0.5%)",
            "- 15% REITs/基建基金 (费率0.5-0.8%)",
            "- 10% 货币基金/短债 (费率0.2-0.3%)",
            "",
            "**预期综合费率**: 0.5-0.7% (节省75-80%)",
            "",
            "### 5️⃣ 摩根太平洋证券 → 富达太平洋基金",
            "- **富达太平洋基金** (可通过海外券商如盈透、富途购买)",
            "- 管理费率: **1.0-1.5%** (节省50%)",
            "- 投资范围: 日本、澳洲、韩国、香港等",
            "",
            "---",
            "",
            "**报告生成时间**: " + datetime.now().strftime("%Y-%m-%d %H:%M"),
            "**数据来源**: Yahoo Finance、公开市场信息、基金招募说明书",
            "**免责声明**: 本报告仅供参考，不构成投资建议。投资有风险，入市需谨慎。费率数据以实际产品为准，请购买前仔细阅读招募说明书。",
        ])
    
    def generate(self):
        """生成完整报告"""
        print("=" * 70)
        print("开始生成基金日报 - 企业分析版")
        print("=" * 70)
        
        self.add_header()
        self.add_market_summary()
        
        for fund_name in FUNDS.keys():
            self.add_fund_company_analysis(fund_name)
        
        self.add_summary()
        
        return "\n".join(self.report_lines)
    
    def save(self, output_dir=None):
        """保存报告"""
        if output_dir is None:
            output_dir = Path(__file__).parent / "reports"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成报告内容
        content = self.generate()
        
        # 保存Markdown
        md_file = output_dir / f"hsbc_fund_daily_{self.date}.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"\n✅ 报告已保存: {md_file}")
        return md_file

if __name__ == "__main__":
    report = FundDailyReport()
    report.save("/root/.openclaw/workspace/reports/fund_daily")
