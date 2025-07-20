#!/usr/bin/env python3
"""
Convert CUMULATIVE_PROGRESS_REPORT.md to a beautifully formatted PDF
"""

import markdown
import pdfkit
from datetime import datetime

def convert_to_pdf():
    # Read the markdown file
    with open('CUMULATIVE_PROGRESS_REPORT.md', 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML with extensions
    html = markdown.markdown(md_content, extensions=[
        'extra',
        'codehilite',
        'tables',
        'toc',
        'nl2br',
        'sane_lists'
    ])
    
    # Add custom CSS for beautiful PDF formatting
    css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            line-height: 1.6;
            color: #1a1a1a;
            max-width: 800px;
            margin: 0 auto;
            padding: 40px;
            background: white;
        }
        
        h1 {
            color: #2563eb;
            border-bottom: 3px solid #e5e7eb;
            padding-bottom: 0.5em;
            margin-top: 2em;
            font-size: 2.5em;
            font-weight: 700;
        }
        
        h2 {
            color: #1e40af;
            margin-top: 1.5em;
            font-size: 1.8em;
            font-weight: 600;
        }
        
        h3 {
            color: #3730a3;
            margin-top: 1.2em;
            font-size: 1.4em;
            font-weight: 600;
        }
        
        h4 {
            color: #4c1d95;
            margin-top: 1em;
            font-size: 1.2em;
            font-weight: 600;
        }
        
        code {
            background: #f3f4f6;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 0.9em;
            color: #dc2626;
        }
        
        pre {
            background: #1e293b;
            color: #e2e8f0;
            padding: 16px;
            border-radius: 8px;
            overflow-x: auto;
            line-height: 1.4;
            font-size: 0.9em;
        }
        
        pre code {
            background: none;
            color: #e2e8f0;
            padding: 0;
        }
        
        blockquote {
            border-left: 4px solid #6366f1;
            padding-left: 16px;
            margin-left: 0;
            color: #4b5563;
            font-style: italic;
        }
        
        ul, ol {
            margin-left: 24px;
            margin-bottom: 16px;
        }
        
        li {
            margin-bottom: 4px;
        }
        
        strong {
            font-weight: 600;
            color: #111827;
        }
        
        em {
            color: #6b7280;
        }
        
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 16px 0;
        }
        
        th {
            background: #f3f4f6;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            border: 1px solid #e5e7eb;
        }
        
        td {
            padding: 12px;
            border: 1px solid #e5e7eb;
        }
        
        a {
            color: #2563eb;
            text-decoration: none;
        }
        
        a:hover {
            text-decoration: underline;
        }
        
        /* Special formatting for status badges */
        p:contains("✅") {
            background: #f0fdf4;
            padding: 8px;
            border-radius: 4px;
            border-left: 3px solid #22c55e;
        }
        
        /* Page break handling */
        h1, h2 {
            page-break-after: avoid;
        }
        
        /* Footer */
        .footer {
            margin-top: 60px;
            padding-top: 20px;
            border-top: 2px solid #e5e7eb;
            text-align: center;
            color: #6b7280;
            font-size: 0.9em;
        }
        
        /* Emoji support */
        .emoji {
            font-size: 1.2em;
        }
        
        /* Special Phoenician text styling */
        code:contains("𐤀"), code:contains("𐤄") {
            font-size: 1.3em;
            color: #7c3aed;
            background: #f3e8ff;
        }
    </style>
    """
    
    # Wrap the HTML with proper structure
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>AI DNA Discovery - Cumulative Progress Report</title>
        {css}
    </head>
    <body>
        {html}
        <div class="footer">
            <p>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
            <p>AI DNA Discovery Project - From Universal Patterns to Deployed Semantic-Neutral Languages</p>
        </div>
    </body>
    </html>
    """
    
    # PDF options for better rendering
    options = {
        'page-size': 'Letter',
        'margin-top': '0.75in',
        'margin-right': '0.75in',
        'margin-bottom': '0.75in',
        'margin-left': '0.75in',
        'encoding': "UTF-8",
        'enable-local-file-access': None,
        'no-outline': None
    }
    
    # Convert HTML to PDF
    output_file = 'AI_DNA_Discovery_Cumulative_Progress_Report.pdf'
    pdfkit.from_string(full_html, output_file, options=options)
    
    print(f"✅ PDF created successfully: {output_file}")
    
    # Get file size
    import os
    size = os.path.getsize(output_file)
    print(f"📄 File size: {size / 1024:.1f} KB")

if __name__ == "__main__":
    convert_to_pdf()