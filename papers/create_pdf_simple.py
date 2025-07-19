#!/usr/bin/env python3
"""
Create a formatted PDF using markdown and weasyprint (HTML to PDF)
Falls back to creating formatted HTML if PDF libraries aren't available
"""

import os
import re
from datetime import datetime

def markdown_to_html(markdown_text):
    """Convert markdown to HTML with basic formatting"""
    
    # HTML template with CSS styling
    html_template = r"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Memory Systems as the Foundation of Machine Awareness</title>
    <style>
        body {
            font-family: 'Georgia', 'Times New Roman', serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 40px 20px;
            background-color: #fff;
        }
        
        h1 {
            color: #1a1a1a;
            font-size: 28px;
            text-align: center;
            margin-bottom: 10px;
            page-break-after: avoid;
        }
        
        h2 {
            color: #2c3e50;
            font-size: 22px;
            margin-top: 40px;
            margin-bottom: 15px;
            page-break-after: avoid;
        }
        
        h3 {
            color: #34495e;
            font-size: 18px;
            margin-top: 25px;
            margin-bottom: 10px;
            page-break-after: avoid;
        }
        
        p {
            text-align: justify;
            margin-bottom: 15px;
            orphans: 3;
            widows: 3;
        }
        
        code {
            background-color: #f5f5f5;
            padding: 2px 4px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            border-radius: 3px;
        }
        
        pre {
            background-color: #f5f5f5;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            overflow-x: auto;
            font-size: 0.85em;
            page-break-inside: avoid;
        }
        
        blockquote {
            border-left: 4px solid #ddd;
            margin-left: 0;
            padding-left: 20px;
            color: #666;
            font-style: italic;
        }
        
        ul, ol {
            margin-bottom: 15px;
        }
        
        li {
            margin-bottom: 5px;
        }
        
        strong {
            font-weight: bold;
            color: #2c3e50;
        }
        
        em {
            font-style: italic;
        }
        
        .metadata {
            text-align: center;
            color: #666;
            font-size: 14px;
            margin-bottom: 40px;
        }
        
        .abstract {
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 30px;
            font-size: 0.95em;
        }
        
        @media print {
            body {
                font-size: 11pt;
                line-height: 1.5;
            }
            
            h1 { font-size: 24pt; }
            h2 { font-size: 18pt; }
            h3 { font-size: 14pt; }
            
            pre {
                page-break-inside: avoid;
            }
        }
    </style>
</head>
<body>
    <div class="metadata">
        <p>AI DNA Discovery Project<br>
        {date}<br>
        Technical Report</p>
    </div>
    
    {content}
    
</body>
</html>
    """
    
    # Convert markdown to HTML
    html_content = markdown_text
    
    # Convert headers
    html_content = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html_content, flags=re.MULTILINE)
    html_content = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html_content, flags=re.MULTILINE)
    html_content = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html_content, flags=re.MULTILINE)
    
    # Convert bold and italic
    html_content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html_content)
    html_content = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html_content)
    
    # Convert code blocks
    html_content = re.sub(r'```python\n(.*?)\n```', r'<pre><code>\1</code></pre>', html_content, flags=re.DOTALL)
    html_content = re.sub(r'```\n(.*?)\n```', r'<pre><code>\1</code></pre>', html_content, flags=re.DOTALL)
    
    # Convert inline code
    html_content = re.sub(r'`(.+?)`', r'<code>\1</code>', html_content)
    
    # Convert lists
    lines = html_content.split('\n')
    in_list = False
    new_lines = []
    
    for line in lines:
        if line.strip().startswith('- '):
            if not in_list:
                new_lines.append('<ul>')
                in_list = True
            new_lines.append(f'<li>{line.strip()[2:]}</li>')
        elif line.strip().startswith(tuple(f'{i}. ' for i in range(1, 10))):
            if not in_list:
                new_lines.append('<ol>')
                in_list = 'ol'
            content = line.strip().split('. ', 1)[1]
            new_lines.append(f'<li>{content}</li>')
        else:
            if in_list:
                tag = '</ol>' if in_list == 'ol' else '</ul>'
                new_lines.append(tag)
                in_list = False
            new_lines.append(line)
    
    if in_list:
        tag = '</ol>' if in_list == 'ol' else '</ul>'
        new_lines.append(tag)
    
    html_content = '\n'.join(new_lines)
    
    # Convert paragraphs
    paragraphs = html_content.split('\n\n')
    formatted_paragraphs = []
    
    for para in paragraphs:
        para = para.strip()
        if para and not para.startswith('<') and not para.startswith('#'):
            # Special handling for abstract
            if para.startswith('This paper explores'):
                para = f'<div class="abstract"><strong>Abstract:</strong> {para}</div>'
            else:
                para = f'<p>{para}</p>'
        formatted_paragraphs.append(para)
    
    html_content = '\n'.join(formatted_paragraphs)
    
    # Fill template
    final_html = html_template.replace('{date}', datetime.now().strftime('%B %Y'))
    final_html = final_html.replace('{content}', html_content)
    
    return final_html

def create_pdf_from_html(html_content, output_filename):
    """Try to create PDF using various methods"""
    
    # First, save as HTML
    html_filename = output_filename.replace('.pdf', '.html')
    with open(html_filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"✅ HTML version created: {html_filename}")
    
    # Try to create PDF using weasyprint
    try:
        import weasyprint
        weasyprint.HTML(string=html_content).write_pdf(output_filename)
        print(f"✅ PDF created with weasyprint: {output_filename}")
        return True
    except ImportError:
        pass
    
    # Try wkhtmltopdf
    try:
        import subprocess
        result = subprocess.run(
            ['wkhtmltopdf', '-', output_filename],
            input=html_content.encode('utf-8'),
            capture_output=True
        )
        if result.returncode == 0:
            print(f"✅ PDF created with wkhtmltopdf: {output_filename}")
            return True
    except:
        pass
    
    print("\n⚠️  PDF libraries not available. HTML version has been created.")
    print("To convert to PDF, you can:")
    print(f"1. Open {html_filename} in a browser and print to PDF")
    print("2. Use an online converter at https://www.web2pdfconvert.com/")
    print("3. Install weasyprint: pip install weasyprint")
    
    return False

def main():
    # Read the markdown file
    with open('memory_and_awareness.md', 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    # Convert to HTML
    html_content = markdown_to_html(markdown_content)
    
    # Create PDF
    output_filename = 'Memory_Systems_as_Foundation_of_Machine_Awareness.pdf'
    create_pdf_from_html(html_content, output_filename)

if __name__ == "__main__":
    main()