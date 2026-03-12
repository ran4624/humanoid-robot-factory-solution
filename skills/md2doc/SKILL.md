---
name: md2doc
description: "Convert Markdown (.md) files to Word (.docx) documents. Supports tables, images, formatting, and custom styles. Use when: user needs to convert markdown to Word format, create documents from markdown content, or generate reports in docx format."
homepage: https://pandoc.org
metadata: { "openclaw": { "emoji": "📝", "requires": { "bins": ["pandoc"] } } }
---

# Markdown to Word Converter

Convert Markdown files to Microsoft Word (.docx) format using Pandoc.

## When to Use

✅ **USE this skill when:**

- Converting README.md to Word document
- Creating reports from markdown notes
- Sharing markdown content in Word format
- Generating formatted documents
- Converting documentation to .docx

## When NOT to Use

❌ **DON'T use this skill when:**

- Converting to PDF → use `pandoc -o output.pdf`
- Converting to HTML → use `pandoc -o output.html`
- Editing Word files → use LibreOffice or Microsoft Word

---

## Installation

### Install Pandoc (if not already installed)

**Linux (yum):**
```bash
yum install -y pandoc
```

**Linux (apt):**
```bash
apt-get install -y pandoc
```

**Or download latest version:**
```bash
# Download latest pandoc
wget https://github.com/jgm/pandoc/releases/download/3.1.11/pandoc-3.1.11-linux-amd64.tar.gz
tar xvzf pandoc-3.1.11-linux-amd64.tar.gz
cp pandoc-3.1.11/bin/pandoc /usr/local/bin/
```

---

## Basic Usage

### Command Line

```bash
# Basic conversion
pandoc input.md -o output.docx

# With table of contents
pandoc input.md -o output.docx --toc

# With custom reference document
pandoc input.md -o output.docx --reference-doc=template.docx
```

### Python Script

```python
import subprocess

def md_to_docx(input_file, output_file=None, **options):
    """Convert markdown to docx"""
    if output_file is None:
        output_file = input_file.replace('.md', '.docx')
    
    cmd = ['pandoc', input_file, '-o', output_file]
    
    # Add options
    if options.get('toc'):
        cmd.append('--toc')
    if options.get('reference_doc'):
        cmd.extend(['--reference-doc', options['reference_doc']])
    
    subprocess.run(cmd, check=True)
    return output_file

# Usage
md_to_docx('document.md', 'output.docx', toc=True)
```

---

## Advanced Options

### Add Table of Contents

```bash
pandoc input.md -o output.docx --toc --toc-depth=2
```

### Use Custom Template

```bash
# First, create a reference document
pandoc -o reference.docx --print-default-data-file reference.docx

# Edit reference.docx with your styles
# Then use it
pandoc input.md -o output.docx --reference-doc=reference.docx
```

### Include Images

Images referenced in markdown will be included automatically:

```markdown
![Alt text](image.png)
```

```bash
pandoc input.md -o output.docx --resource-path=./images
```

### Set Document Metadata

```bash
pandoc input.md -o output.docx \
  --metadata title="My Document" \
  --metadata author="John Doe" \
  --metadata date="2024-03-12"
```

---

## Markdown Features Supported

| Feature | Status | Notes |
|---------|--------|-------|
| Headers | ✅ | All levels supported |
| Bold/Italic | ✅ | `**bold**`, `*italic*` |
| Lists | ✅ | Ordered and unordered |
| Tables | ✅ | Pipe tables |
| Links | ✅ | Clickable in Word |
| Images | ✅ | Embedded in document |
| Code blocks | ✅ | With syntax highlighting |
| Blockquotes | ✅ | Styled as quotes |
| Footnotes | ✅ | Endnotes in Word |
| Math | ✅ | LaTeX math support |

---

## Examples

### Example 1: Simple Conversion

```bash
# Convert single file
pandoc README.md -o README.docx

# Batch convert
for file in *.md; do
    pandoc "$file" -o "${file%.md}.docx"
done
```

### Example 2: With Custom Styling

```bash
# Create styled document
pandoc report.md -o report.docx \
  --reference-doc=company-template.docx \
  --metadata title="Annual Report" \
  --metadata author="Tech Team"
```

### Example 3: Python Batch Processing

```python
import os
import subprocess

def batch_convert_md_to_docx(directory):
    """Convert all markdown files in directory to docx"""
    for filename in os.listdir(directory):
        if filename.endswith('.md'):
            md_path = os.path.join(directory, filename)
            docx_path = os.path.join(directory, filename.replace('.md', '.docx'))
            
            subprocess.run(['pandoc', md_path, '-o', docx_path])
            print(f"Converted: {filename} → {docx_path}")

# Usage
batch_convert_md_to_docx('/path/to/markdown/files')
```

---

## Troubleshooting

### Images Not Showing

```bash
# Ensure images are in correct path
pandoc input.md -o output.docx --resource-path=./images
```

### Chinese Characters Not Displaying

```bash
# Use xelatex for better font support
pandoc input.md -o output.docx --pdf-engine=xelatex
```

### Formatting Issues

```bash
# Create and customize reference document
pandoc -o custom-reference.docx --print-default-data-file reference.docx
# Edit custom-reference.docx in Word, then:
pandoc input.md -o output.docx --reference-doc=custom-reference.docx
```

---

## Alternative: Python-only Solution

If Pandoc is not available, use Python libraries:

```bash
pip install python-docx markdown
```

```python
from docx import Document
import markdown
from bs4 import BeautifulSoup

def md_to_docx_simple(md_file, docx_file):
    # Read markdown
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert to HTML
    html = markdown.markdown(md_content)
    
    # Parse HTML
    soup = BeautifulSoup(html, 'html.parser')
    
    # Create Word document
    doc = Document()
    
    # Add content (basic implementation)
    for elem in soup.find_all(['p', 'h1', 'h2', 'h3', 'ul', 'ol']):
        if elem.name == 'h1':
            doc.add_heading(elem.text, level=1)
        elif elem.name == 'h2':
            doc.add_heading(elem.text, level=2)
        elif elem.name == 'p':
            doc.add_paragraph(elem.text)
    
    doc.save(docx_file)
```

---

## References

- [Pandoc Documentation](https://pandoc.org/MANUAL.html)
- [Pandoc Markdown](https://pandoc.org/MANUAL.html#pandocs-markdown)
- [python-docx](https://python-docx.readthedocs.io/)

---

**Version:** 1.0  
**Last Updated:** 2026-03-12
