#!/usr/bin/env python3
"""
Chrome 高级自动化演示 - 抓取动态内容
"""
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import json

print("="*50)
print("  高级演示：页面交互与数据提取")
print("="*50)
print()

options = Options()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-gpu')
options.add_argument('--window-size=1920,1080')

print("🚀 启动浏览器...")
driver = webdriver.Chrome(options=options)

try:
    # 访问一个测试页面
    print("📄 访问网页...")
    driver.get('https://httpbin.org/html')
    
    # 等待元素加载
    wait = WebDriverWait(driver, 10)
    heading = wait.until(EC.presence_of_element_located((By.TAG_NAME, 'h1')))
    
    print(f"✅ 找到标题元素: {heading.text}")
    
    # 获取所有段落
    paragraphs = driver.find_elements(By.TAG_NAME, 'p')
    print(f"✅ 页面共有 {len(paragraphs)} 个段落")
    
    for i, p in enumerate(paragraphs[:3], 1):
        text = p.text[:50] + "..." if len(p.text) > 50 else p.text
        print(f"   段落 {i}: {text}")
    
    # 截图
    driver.save_screenshot('/root/demo5_advanced.png')
    print("📸 截图已保存: /root/demo5_advanced.png")
    
    # 获取页面性能数据
    print("\n⚡ 页面性能数据:")
    navigation = driver.execute_script("return performance.getEntriesByType('navigation')[0]")
    if navigation:
        print(f"   DNS 查询: {navigation.get('domainLookupEnd', 0) - navigation.get('domainLookupStart', 0):.0f}ms")
        print(f"   连接时间: {navigation.get('connectEnd', 0) - navigation.get('connectStart', 0):.0f}ms")
        print(f"   页面加载: {navigation.get('loadEventEnd', 0) - navigation.get('startTime', 0):.0f}ms")
    
    print("\n✅ 高级演示完成!")
    
finally:
    driver.quit()
    print("🔒 浏览器已关闭")
