#!/usr/bin/env python3
"""
Convert Markdown report to PDF with embedded images
"""

import markdown2
import pdfkit
import base64
import os

# Read the markdown file
with open('/home/dp/ai-workspace/AI_DNA_Discovery_Full_Report.md', 'r') as f:
    md_content = f.read()

# Function to embed images as base64
def embed_images(html_content):
    """Replace image paths with base64 encoded data"""
    import re
    
    # Find all image tags
    img_pattern = r'<img[^>]+src="([^"]+)"[^>]*>'
    
    def replace_img(match):
        img_path = match.group(1)
        full_path = f'/home/dp/ai-workspace/{img_path}'
        
        if os.path.exists(full_path):
            with open(full_path, 'rb') as img_file:
                img_data = base64.b64encode(img_file.read()).decode()
                return f'<img src="data:image/png;base64,{img_data}" style="max-width: 100%; height: auto;">'
        return match.group(0)
    
    return re.sub(img_pattern, replace_img, html_content)

# Convert markdown to HTML
html_content = markdown2.markdown(md_content, extras=['tables', 'fenced-code-blocks'])

# Add CSS styling
html_with_style = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        h1 {{
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            border-bottom: 2px solid #3498db;
            padding-bottom: 5px;
            margin-top: 30px;
        }}
        h3 {{
            color: #34495e;
            margin-top: 20px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 4px;
            border-radius: 3px;
        }}
        pre {{
            background-color: #f4f4f4;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        blockquote {{
            border-left: 4px solid #3498db;
            padding-left: 20px;
            margin-left: 0;
            color: #555;
            font-style: italic;
        }}
        .highlight {{
            background-color: #fff3cd;
            padding: 2px 4px;
            border-radius: 3px;
        }}
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 20px auto;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        strong {{
            color: #2c3e50;
        }}
        em {{
            color: #27ae60;
        }}
    </style>
</head>
<body>
{html_content}
</body>
</html>
"""

# Embed images
html_with_images = embed_images(html_with_style)

# Configure PDF options
options = {
    'page-size': 'A4',
    'margin-top': '0.75in',
    'margin-right': '0.75in',
    'margin-bottom': '0.75in',
    'margin-left': '0.75in',
    'encoding': "UTF-8",
    'no-outline': None,
    'enable-local-file-access': None
}

# Generate PDF
output_path = '/home/dp/ai-workspace/AI_DNA_Discovery_Report.pdf'

try:
    pdfkit.from_string(html_with_images, output_path, options=options)
    print(f"✓ PDF successfully generated: {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
except Exception as e:
    print(f"Error generating PDF: {e}")
    print("\nAlternative: Saving as HTML...")
    
    # Save as HTML as fallback
    html_path = '/home/dp/ai-workspace/AI_DNA_Discovery_Report.html'
    with open(html_path, 'w') as f:
        f.write(html_with_images)
    print(f"✓ HTML report saved: {html_path}")
    print("  You can open this in a browser and print to PDF")