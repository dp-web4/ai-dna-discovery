#!/usr/bin/env python3
"""
Simple markdown to HTML converter for viewing the report
"""

import re
import os

def convert_md_to_html(md_file, html_file):
    """Convert markdown to HTML with embedded images"""
    
    with open(md_file, 'r') as f:
        content = f.read()
    
    # HTML template with dark theme
    html_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Weight Analysis Progress Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #e0e0e0;
            background-color: #1a1a1a;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1 {{ color: #4ecdc4; margin-top: 30px; }}
        h2 {{ color: #45b7d1; margin-top: 25px; }}
        h3 {{ color: #96ceb4; margin-top: 20px; }}
        code {{
            background-color: #2d2d2d;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
        pre {{
            background-color: #2d2d2d;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        blockquote {{
            border-left: 4px solid #4ecdc4;
            padding-left: 15px;
            margin-left: 0;
            font-style: italic;
        }}
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 20px auto;
            border: 1px solid #333;
        }}
        hr {{
            border: none;
            border-top: 1px solid #444;
            margin: 30px 0;
        }}
        strong {{ color: #ffd93d; }}
        a {{ color: #4ecdc4; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        .meta-info {{
            background-color: #2d2d2d;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 30px;
        }}
    </style>
</head>
<body>
{content}
</body>
</html>"""
    
    # Convert markdown to HTML
    html_content = content
    
    # Headers
    html_content = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html_content, flags=re.MULTILINE)
    html_content = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html_content, flags=re.MULTILINE)
    html_content = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html_content, flags=re.MULTILINE)
    
    # Bold and italic
    html_content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html_content)
    html_content = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html_content)
    
    # Code blocks
    html_content = re.sub(r'```python\n(.*?)\n```', r'<pre><code>\1</code></pre>', html_content, flags=re.DOTALL)
    html_content = re.sub(r'```bash\n(.*?)\n```', r'<pre><code>\1</code></pre>', html_content, flags=re.DOTALL)
    html_content = re.sub(r'```\n(.*?)\n```', r'<pre><code>\1</code></pre>', html_content, flags=re.DOTALL)
    
    # Inline code
    html_content = re.sub(r'`([^`]+)`', r'<code>\1</code>', html_content)
    
    # Images - embed as base64
    def replace_image(match):
        alt_text = match.group(1)
        img_path = match.group(2)
        if os.path.exists(img_path):
            return f'<img src="{img_path}" alt="{alt_text}" />'
        return f'<p>[Image: {alt_text}]</p>'
    
    html_content = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', replace_image, html_content)
    
    # Lists
    html_content = re.sub(r'^- (.+)$', r'<li>\1</li>', html_content, flags=re.MULTILINE)
    html_content = re.sub(r'(<li>.*</li>\n)+', r'<ul>\g<0></ul>\n', html_content, flags=re.MULTILINE)
    
    # Numbered lists
    html_content = re.sub(r'^\d+\. (.+)$', r'<li>\1</li>', html_content, flags=re.MULTILINE)
    
    # Horizontal rules
    html_content = re.sub(r'^---$', '<hr/>', html_content, flags=re.MULTILINE)
    
    # Paragraphs
    html_content = re.sub(r'\n\n', '</p>\n<p>', html_content)
    html_content = f'<p>{html_content}</p>'
    
    # Clean up empty paragraphs
    html_content = re.sub(r'<p>\s*</p>', '', html_content)
    html_content = re.sub(r'<p>(<h[123]>)', r'\1', html_content)
    html_content = re.sub(r'(</h[123]>)</p>', r'\1', html_content)
    
    # Write HTML file
    final_html = html_template.format(content=html_content)
    with open(html_file, 'w') as f:
        f.write(final_html)
    
    print(f"HTML report created: {html_file}")
    print("\nTo view the report:")
    print(f"  1. Open file: {html_file}")
    print("  2. Or use a browser to open it")
    print("\nNote: The HTML version preserves all content and images.")

if __name__ == "__main__":
    convert_md_to_html(
        'weight_analysis_progress_report.md',
        'weight_analysis_progress_report.html'
    )