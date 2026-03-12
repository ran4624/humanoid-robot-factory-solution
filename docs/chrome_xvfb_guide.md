# Xvfb + Chrome 有头模式使用指南

## ✅ 安装状态

| 组件 | 版本 | 状态 |
|------|------|------|
| Xvfb | 1.20.11 | ✅ 已安装 |
| X11 字体 | 7.5 | ✅ 已安装 |
| DISPLAY :99 | 1920x1080x24 | ✅ 可用 |

---

## 🚀 快速开始

### 1. 启动 Xvfb 虚拟显示

```bash
# 方法1: 使用脚本启动
source /root/xvfb_start.sh

# 方法2: 手动启动
Xvfb :99 -screen 0 1920x1080x24 -ac +extension GLX +render -noreset &
export DISPLAY=:99
```

### 2. 验证 Xvfb 运行

```bash
# 检查进程
ps aux | grep Xvfb

# 检查 DISPLAY 环境变量
echo $DISPLAY
# 输出应显示 :99
```

### 3. 使用有头模式 Chrome

```bash
# 确保 DISPLAY 环境变量已设置
export DISPLAY=:99

# 运行有头模式 Chrome（截图）
google-chrome --no-sandbox \
    --window-size=1920,1080 \
    --screenshot=output.png \
    https://www.example.com

# 完整页面截图
google-chrome --no-sandbox \
    --window-size=1920,1080 \
    --full-page-screenshot \
    --screenshot=fullpage.png \
    https://www.example.com
```

### 4. Python + Selenium 有头模式

```python
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import os

# 设置 DISPLAY 环境变量
os.environ['DISPLAY'] = ':99'

# 配置 Chrome（不使用 --headless）
options = Options()
options.add_argument('--no-sandbox')
options.add_argument('--disable-gpu')
options.add_argument('--window-size=1920,1080')
# 注意: 不添加 --headless，这样就能"看到"浏览器

# 启动浏览器
driver = webdriver.Chrome(options=options)

# 访问网页
driver.get('https://www.example.com')

# 截图（将渲染完整的浏览器窗口）
driver.save_screenshot('headed_screenshot.png')

# 执行 JavaScript
result = driver.execute_script('return document.title')
print(f"页面标题: {result}")

# 关闭浏览器
driver.quit()
```

### 5. Playwright 有头模式

```python
from playwright.sync_api import sync_playwright
import os

os.environ['DISPLAY'] = ':99'

with sync_playwright() as p:
    # headless=False 启用有头模式
    browser = p.chromium.launch(
        headless=False,
        args=['--no-sandbox', '--disable-gpu']
    )
    
    page = browser.new_page(viewport={'width': 1920, 'height': 1080})
    page.goto('https://www.example.com')
    
    # 截图
    page.screenshot(path='headed.png', full_page=True)
    
    browser.close()
```

---

## 📁 相关脚本

| 脚本 | 路径 | 用途 |
|------|------|------|
| 启动 Xvfb | `/root/xvfb_start.sh` | 启动虚拟显示 |
| 停止 Xvfb | `/root/xvfb_stop.sh` | 停止虚拟显示 |
| 完整测试 | `/root/test_chrome_xvfb.sh` | 有头模式测试 |
| Chrome 测试 | `/root/test_chrome.sh` | 基础功能测试 |

---

## 🔧 高级配置

### 自定义分辨率和显示号

```bash
# 使用 4K 分辨率
SCREEN_WIDTH=3840 SCREEN_HEIGHT=2160 source /root/xvfb_start.sh

# 使用不同的显示号
DISPLAY_NUM=100 source /root/xvfb_start.sh
```

### 多显示器支持

```bash
# 启动多个 Xvfb 实例
Xvfb :99 -screen 0 1920x1080x24 &
Xvfb :100 -screen 0 1280x720x24 &

# 在不同显示器上运行 Chrome
DISPLAY=:99 google-chrome --no-sandbox ... &
DISPLAY=:100 google-chrome --no-sandbox ... &
```

### VNC 远程查看（可选）

如果你想远程"看到"虚拟显示器的内容，可以安装 VNC：

```bash
# 安装 VNC 服务器
yum install -y tigervnc-server

# 启动 VNC，连接到 Xvfb
x11vnc -display :99 -nopw -forever &

# 然后使用 VNC 客户端连接
```

---

## 📝 有头 vs 无头模式对比

| 特性 | 无头模式 | 有头模式 (Xvfb) |
|------|---------|----------------|
| **资源占用** | 低 | 中等（需要 Xvfb）|
| **WebGL** | 部分支持 | 完整支持 |
| **插件扩展** | 部分支持 | 完整支持 |
| **文件下载** | 有限制 | 正常 |
| **截图效果** | 基础 | 完整（包括滚动条等）|
| **适用场景** | 简单抓取 | 复杂交互、视觉测试 |

---

## ⚠️ 注意事项

1. **必须先启动 Xvfb** 再运行 Chrome，否则会出现 `DISPLAY not set` 错误

2. **root 用户必须使用** `--no-sandbox` 参数

3. **内存占用**: Xvfb + Chrome 大约需要 200-500MB 内存

4. **清理**: 使用完后记得停止 Xvfb
   ```bash
   bash /root/xvfb_stop.sh
   ```

5. **并行运行**: 可以启动多个 Xvfb 实例，使用不同的显示号

---

## 🐛 故障排查

### 问题: `cannot open display: :99`
```bash
# 解决: 确保 Xvfb 已启动
source /root/xvfb_start.sh
```

### 问题: `The display is already in use`
```bash
# 解决: 停止现有 Xvfb 或更换显示号
bash /root/xvfb_stop.sh
DISPLAY_NUM=100 source /root/xvfb_start.sh
```

### 问题: Chrome 启动慢
```bash
# 禁用不必要的功能加速启动
google-chrome --no-sandbox \
    --disable-extensions \
    --disable-plugins \
    --disable-images \
    --disable-javascript \
    ...
```

---

## 🔗 相关文档

- [Chrome 安装文档](./chrome_installation_guide.md)
- [Xvfb 官方文档](https://www.x.org/releases/X11R7.6/doc/man/man1/Xvfb.1.xhtml)
- [Selenium 文档](https://www.selenium.dev/documentation/)
- [Playwright 文档](https://playwright.dev/python/)

---

**配置完成时间**: 2026-03-12
