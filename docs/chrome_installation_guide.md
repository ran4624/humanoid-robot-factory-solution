# Google Chrome 浏览器安装完成

## ✅ 安装状态

| 组件 | 版本 | 状态 |
|------|------|------|
| Google Chrome | 146.0.7680.71 | ✅ 已安装 |
| ChromeDriver | 146.0.7680.71 | ✅ 已安装 |
| 无头模式 | - | ✅ 已测试 |
| 截图功能 | - | ✅ 已测试 |

---

## 📍 安装路径

- **Chrome 可执行文件**: `/usr/bin/google-chrome`
- **ChromeDriver**: `/usr/local/bin/chromedriver`
- **测试脚本**: `/root/test_chrome.sh`
- **示例截图**: `/root/chrome_screenshot.png`

---

## 🚀 使用方法

### 1. 无头模式 (Headless) - 推荐用于服务器

无头模式不需要图形界面，适合自动化任务：

```bash
# 获取网页 HTML
google-chrome --headless --no-sandbox --disable-gpu --dump-dom https://www.example.com

# 截取网页截图
google-chrome --headless --no-sandbox --disable-gpu \
  --screenshot=output.png \
  --window-size=1920,1080 \
  https://www.google.com

# 保存为 PDF
google-chrome --headless --no-sandbox --disable-gpu \
  --print-to-pdf=output.pdf \
  https://www.example.com

# 设置自定义 User-Agent
google-chrome --headless --no-sandbox --disable-gpu \
  --user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)" \
  --dump-dom https://www.example.com
```

### 2. 有头模式 (Headed) - 需要图形界面

如果需要显示浏览器窗口（需要安装桌面环境或 VNC）：

```bash
# 直接打开浏览器（需要 DISPLAY 环境变量）
google-chrome --no-sandbox https://www.google.com

# 指定窗口大小
google-chrome --no-sandbox --window-size=1920,1080 https://www.google.com
```

### 3. Python + Selenium 自动化

```python
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

# 无头模式配置
chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-gpu')
chrome_options.add_argument('--window-size=1920,1080')

# 启动浏览器
driver = webdriver.Chrome(options=chrome_options)

# 访问网页
driver.get('https://www.google.com')

# 截图
driver.save_screenshot('screenshot.png')

# 获取页面源码
print(driver.page_source)

# 关闭浏览器
driver.quit()
```

### 4. Playwright 自动化

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    # 启动无头浏览器
    browser = p.chromium.launch(
        headless=True,
        args=['--no-sandbox', '--disable-gpu']
    )
    
    page = browser.new_page(viewport={'width': 1920, 'height': 1080})
    page.goto('https://www.google.com')
    
    # 截图
    page.screenshot(path='screenshot.png')
    
    # PDF
    page.pdf(path='page.pdf')
    
    browser.close()
```

---

## 🔧 常用参数说明

| 参数 | 说明 |
|------|------|
| `--headless` | 无头模式（不显示界面）|
| `--no-sandbox` | 禁用沙箱（root 用户必需）|
| `--disable-gpu` | 禁用 GPU 加速 |
| `--dump-dom` | 输出页面 DOM |
| `--screenshot=path` | 截图保存路径 |
| `--print-to-pdf=path` | 保存为 PDF |
| `--window-size=W,H` | 设置窗口大小 |
| `--user-agent=UA` | 设置 User-Agent |
| `--proxy-server=host:port` | 设置代理 |
| `--disable-javascript` | 禁用 JavaScript |

---

## 🧪 测试验证

运行测试脚本验证安装：

```bash
bash /root/test_chrome.sh
```

---

## 📝 注意事项

1. **root 用户运行**: 必须使用 `--no-sandbox` 参数
2. **无头模式**: 服务器环境推荐使用无头模式
3. **内存占用**: Chrome 内存占用较大，建议 2GB+ 内存
4. **多实例**: 避免同时启动过多 Chrome 实例

---

## 🔗 相关链接

- [Chrome Headless 文档](https://developer.chrome.com/docs/chromium/new_headless)
- [ChromeDriver 文档](https://chromedriver.chromium.org/)
- [Selenium 文档](https://www.selenium.dev/documentation/)
- [Playwright 文档](https://playwright.dev/python/)

---

**安装完成时间**: 2026-03-12
