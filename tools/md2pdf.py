#!/usr/bin/env python3
"""
MD to PDF Converter Tool
支持中文的Markdown转PDF工具
"""

import sys
import os
import argparse
from pathlib import Path

def check_dependencies():
    """检查必要的依赖"""
    try:
        import markdown
        from weasyprint import HTML, CSS
        return True
    except ImportError as e:
        print(f"缺少依赖: {e}")
        print("请安装: pip install markdown weasyprint")
        return False

def md_to_html(md_content):
    """将Markdown转换为HTML"""
    import markdown
    
    # 配置markdown扩展
    extensions = [
        'tables',
        'fenced_code',
        'toc',
        'nl2br',
    ]
    
    html_content = markdown.markdown(md_content, extensions=extensions)
    return html_content

def get_css_style():
    """获取CSS样式"""
    return """
    @page {
        size: A4;
        margin: 2cm;
        @bottom-center {
            content: counter(page);
            font-size: 10pt;
            font-family: "Noto Sans CJK SC", sans-serif;
        }
    }
    
    body {
        font-family: "Noto Sans CJK SC", "Noto Sans SC", "Noto Serif CJK SC", sans-serif;
        font-size: 11pt;
        line-height: 1.6;
        color: #333;
    }
    
    h1 {
        font-size: 20pt;
        color: #1a1a1a;
        border-bottom: 2px solid #333;
        padding-bottom: 10px;
        margin-top: 30px;
    }
    
    h2 {
        font-size: 16pt;
        color: #2a2a2a;
        border-bottom: 1px solid #ccc;
        padding-bottom: 5px;
        margin-top: 25px;
    }
    
    h3 {
        font-size: 13pt;
        color: #3a3a3a;
        margin-top: 20px;
    }
    
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
        font-size: 10pt;
    }
    
    th, td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
    }
    
    th {
        background-color: #f5f5f5;
        font-weight: bold;
    }
    
    tr:nth-child(even) {
        background-color: #fafafa;
    }
    
    code {
        background-color: #f4f4f4;
        padding: 2px 6px;
        border-radius: 3px;
        font-family: "Courier New", monospace;
        font-size: 10pt;
    }
    
    pre {
        background-color: #f4f4f4;
        padding: 15px;
        border-radius: 5px;
        overflow-x: auto;
        font-size: 9pt;
    }
    
    blockquote {
        border-left: 4px solid #ccc;
        margin: 15px 0;
        padding: 10px 20px;
        background-color: #f9f9f9;
        color: #666;
    }
    
    ul, ol {
        margin: 10px 0;
        padding-left: 30px;
    }
    
    li {
        margin: 5px 0;
    }
    
    strong {
        color: #000;
    }
    
    .header {
        text-align: center;
        margin-bottom: 30px;
    }
    
    .footer {
        text-align: center;
        margin-top: 30px;
        font-size: 9pt;
        color: #666;
    }
    """

def wrap_html(content, title="Document"):
    """包装完整的HTML文档"""
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
</head>
<body>
    {content}
</body>
</html>"""

def convert_md_to_pdf(md_file, pdf_file=None):
    """转换Markdown文件为PDF"""
    from weasyprint import HTML, CSS
    
    # 读取Markdown文件
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # 转换为HTML
    html_body = md_to_html(md_content)
    
    # 包装完整HTML
    title = Path(md_file).stem
    html_content = wrap_html(html_body, title)
    
    # 确定输出文件名
    if pdf_file is None:
        pdf_file = str(Path(md_file).with_suffix('.pdf'))
    
    # 转换为PDF
    html_doc = HTML(string=html_content, encoding='utf-8')
    css = CSS(string=get_css_style())
    
    html_doc.write_pdf(pdf_file, stylesheets=[css])
    
    return pdf_file

def main():
    parser = argparse.ArgumentParser(description='Markdown to PDF Converter')
    parser.add_argument('input', help='Input Markdown file')
    parser.add_argument('-o', '--output', help='Output PDF file (optional)')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 1.0')
    
    args = parser.parse_args()
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误: 文件不存在: {args.input}")
        sys.exit(1)
    
    try:
        output_file = convert_md_to_pdf(args.input, args.output)
        print(f"✓ 转换成功: {output_file}")
    except Exception as e:
        print(f"✗ 转换失败: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
