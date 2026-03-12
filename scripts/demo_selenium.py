#!/usr/bin/env python3
"""
Chrome + Selenium 自动化演示
"""
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import os

print("="*50)
print("  Chrome + Selenium 自动化演示")
print("="*50)
print()

# 配置 Chrome 选项
options = Options()
options.add_argument('--headless')  # 无头模式
options.add_argument('--no-sandbox')
options.add_argument('--disable-gpu')
options.add_argument('--window-size=1920,1080')
options.add_argument('--disable-blink-features=AutomationControlled')
options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')

print("🚀 启动 Chrome 浏览器...")
driver = webdriver.Chrome(options=options)

try:
    # 1. 访问网页
    print("📄 访问 example.com...")
    driver.get('https://www.example.com')
    time.sleep(2)
    
    # 2. 获取页面信息
    print(f"✅ 页面标题: {driver.title}")
    print(f"✅ 当前 URL: {driver.current_url}")
    
    # 3. 截图
    screenshot_path = '/root/demo4_selenium.png'
    driver.save_screenshot(screenshot_path)
    print(f"📸 截图已保存: {screenshot_path}")
    
    # 4. 获取页面内容
    page_text = driver.find_element(By.TAG_NAME, 'body').text
    print(f"📝 页面内容预览: {page_text[:100]}...")
    
    # 5. 执行 JavaScript
    print("⚡ 执行 JavaScript...")
    result = driver.execute_script('return document.title + " - " + navigator.userAgent')
    print(f"   返回: {result[:80]}...")
    
    print()
    print("✅ 演示完成!")
    
finally:
    driver.quit()
    print("🔒 浏览器已关闭")
