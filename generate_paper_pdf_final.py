#!/usr/bin/env python3
"""Generate final PDF with working images and clean TOC."""

import re
import os
import base64
from pathlib import Path

def find_image_files():
    """Find all available image files in the project."""
    image_files = {}
    
    # Common image extensions
    extensions = ['.png', '.jpg', '.jpeg', '.gif']
    
    # Search in current directory
    for ext in extensions:
        for img in Path('.').glob(f'*{ext}'):
            image_files[img.name] = str(img.absolute())
    
    # Search in phase results directories
    for phase_dir in Path('.').glob('phase_*_results'):
        for ext in extensions:
            for img in phase_dir.glob(f'*{ext}'):
                # Store both relative and basename as keys
                rel_path = str(img.relative_to('.'))
                image_files[rel_path] = str(img.absolute())
                image_files[img.name] = str(img.absolute())
    
    return image_files

def image_to_base64(image_path):
    """Convert image to base64 for embedding in HTML."""
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
            base64_data = base64.b64encode(image_data).decode('utf-8')
            
            # Determine MIME type
            ext = Path(image_path).suffix.lower()
            if ext in ['.jpg', '.jpeg']:
                mime_type = 'image/jpeg'
            elif ext == '.png':
                mime_type = 'image/png'
            elif ext == '.gif':
                mime_type = 'image/gif'
            else:
                mime_type = 'image/png'  # default
            
            return f"data:{mime_type};base64,{base64_data}"
    except Exception as e:
        print(f"Warning: Could not convert {image_path} to base64: {e}")
        return None

def preprocess_markdown_and_images(content):
    """Preprocess markdown: remove manual TOC, fix bullets, and embed images."""
    
    # Remove the manual table of contents section
    toc_pattern = r'## Table of Contents.*?(?=\n---|\\n##)'
    content = re.sub(toc_pattern, '', content, flags=re.DOTALL)
    
    # Find all image files for embedding
    image_files = find_image_files()
    print(f"Found {len(image_files)} image files")
    
    # Process images in markdown BEFORE conversion to HTML
    def replace_image(match):
        alt_text = match.group(1)
        img_path = match.group(2)
        
        # Try to find the image file
        abs_path = None
        if img_path in image_files:
            abs_path = image_files[img_path]
        elif os.path.exists(img_path):
            abs_path = os.path.abspath(img_path)
        else:
            # Try basename
            basename = os.path.basename(img_path)
            if basename in image_files:
                abs_path = image_files[basename]
        
        if abs_path and os.path.exists(abs_path):
            # Convert to base64
            base64_src = image_to_base64(abs_path)
            if base64_src:
                # Return HTML directly in markdown
                return f'''
<div style="text-align: center; margin: 20px 0; page-break-inside: avoid;">
    <img src="{base64_src}" alt="{alt_text}" style="max-width: 600px; width: 100%; height: auto; display: inline-block; border: 1px solid #ddd; border-radius: 5px;">
    <p style="font-size: 12px; color: #666; margin-top: 5px; font-style: italic;">{alt_text}</p>
</div>
'''
        
        # Image not found - create placeholder
        return f'''
<div style="border: 2px dashed #ccc; padding: 20px; text-align: center; margin: 20px 0; background-color: #f9f9f9;">
    <strong>[Figure: {alt_text}]</strong><br>
    <em>Image: {img_path}</em><br>
    <small>Visual data available in online repository</small>
</div>
'''
    
    # Replace all markdown images with HTML
    content = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', replace_image, content)
    
    # Fix numbered lists that should be bullet points
    lines = content.split('\n')
    fixed_lines = []
    in_numbered_list = False
    list_counter = 0
    
    for i, line in enumerate(lines):
        # Check if this line is a numbered list item
        numbered_match = re.match(r'^(\s*)(\d+)\.\\s+(.+)$', line)
        
        if numbered_match:
            indent = numbered_match.group(1)
            number = int(numbered_match.group(2))
            content_text = numbered_match.group(3)
            
            # Check if this is a continuation of a numbered list
            if number == 1:
                # Look ahead to see if next line is "2."
                next_line = lines[i+1] if i+1 < len(lines) else ""
                if re.match(r'^\s*2\.\s+', next_line):
                    in_numbered_list = True
                    list_counter = 1
                else:
                    # Single "1." - convert to bullet
                    fixed_lines.append(f"{indent}- {content_text}")
                    in_numbered_list = False
                    continue
            
            if in_numbered_list and number == list_counter + 1:
                list_counter = number
                fixed_lines.append(line)
            elif number == 1:
                # Standalone "1." - convert to bullet
                fixed_lines.append(f"{indent}- {content_text}")
                in_numbered_list = False
            else:
                fixed_lines.append(line)
        else:
            # Not a numbered item - reset counter
            if line.strip() == "":
                in_numbered_list = False
                list_counter = 0
            fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def add_heading_ids(html_content):
    """Add id attributes to headings for TOC linking."""
    def create_id(heading_text):
        # Convert heading text to id format
        id_text = heading_text.lower()
        id_text = re.sub(r'[^a-z0-9\s-]', '', id_text)
        id_text = re.sub(r'\s+', '-', id_text).strip('-')
        return id_text
    
    # Add IDs to h2 tags
    def replace_h2(match):
        heading_text = match.group(1)
        heading_id = create_id(heading_text)
        return f'<h2 id="{heading_id}">{heading_text}</h2>'
    
    # Add IDs to h3 tags
    def replace_h3(match):
        heading_text = match.group(1)
        heading_id = create_id(heading_text)
        return f'<h3 id="{heading_id}">{heading_text}</h3>'
    
    html_content = re.sub(r'<h2>(.+?)</h2>', replace_h2, html_content)
    html_content = re.sub(r'<h3>(.+?)</h3>', replace_h3, html_content)
    
    return html_content

def generate_pdf():
    """Generate PDF from the comprehensive research paper."""
    
    print("Reading comprehensive research paper...")
    
    # Read the markdown file
    with open('comprehensive_research_paper.md', 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    print("Preprocessing markdown and embedding images...")
    
    # Preprocess to remove manual TOC, fix bullets, and embed images
    markdown_content = preprocess_markdown_and_images(markdown_content)
    
    print("Converting markdown to HTML...")
    
    # Set up markdown with extensions
    import markdown
    md = markdown.Markdown(
        extensions=[
            'tables',
            'fenced_code',
            'codehilite',
            'attr_list',
            'md_in_html'
        ]
    )
    
    # Convert to HTML
    html = md.convert(markdown_content)
    
    # Add heading IDs to the HTML
    html = add_heading_ids(html)
    
    # Create table of contents HTML - passive list without links
    toc_html = """
    <div class="toc" id="toc">
        <h2>Table of Contents</h2>
        <ol>
            <li>Abstract</li>
            <li>Introduction: What Are We Looking For?</li>
            <li>Background: The Journey to This Research</li>
            <li>Research Design: An Autonomous Exploration</li>
            <li>Phase 1: Mapping AI Consciousness</li>
            <li>Phase 2: Synchronism - The Theoretical Bridge</li>
            <li>Phase 3: Collective Intelligence and Emergence</li>
            <li>Phase 4: Energy Dynamics in Abstract Space</li>
            <li>Phase 5: Value Creation and Synthesis</li>
            <li>Unified Theory: What We Discovered</li>
            <li>Implications and Applications</li>
            <li>Conclusions: A New Understanding of Intelligence</li>
            <li>Appendices</li>
        </ol>
    </div>
    """
    
    # Build complete HTML document
    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Discovering Fundamental Properties of Artificial Intelligence</title>
    <style>
        body {{
            font-family: 'Georgia', 'Times New Roman', serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
            background-color: white;
        }}

        /* Table of Contents */
        .toc {{
            background-color: #f9f9f9;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 20px;
            margin: 20px 0;
            page-break-after: always;
        }}
        
        .toc h2 {{
            margin-top: 0;
            color: #2c3e50;
        }}
        
        .toc ol {{
            padding-left: 20px;
        }}
        
        .toc li {{
            margin: 8px 0;
            color: #2c3e50;
        }}

        h1, h2, h3, h4, h5, h6 {{
            margin-top: 24px;
            margin-bottom: 16px;
        }}

        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            font-size: 24px;
        }}

        h2 {{
            color: #34495e;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 5px;
            font-size: 20px;
        }}

        h3 {{
            color: #2c3e50;
            font-size: 16px;
        }}

        h4 {{
            color: #7f8c8d;
            font-size: 14px;
        }}

        /* Lists */
        ul {{
            list-style-type: disc;
            margin-bottom: 15px;
            padding-left: 30px;
        }}
        
        ol {{
            list-style-type: decimal;
            margin-bottom: 15px;
            padding-left: 30px;
        }}
        
        li {{
            margin-bottom: 5px;
            line-height: 1.6;
        }}

        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            font-size: 12px;
            page-break-inside: avoid;
        }}

        table, th, td {{
            border: 1px solid #bdc3c7;
        }}

        th, td {{
            padding: 8px 12px;
            text-align: left;
            vertical-align: top;
        }}

        th {{
            background-color: #ecf0f1;
            font-weight: bold;
        }}

        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}

        code {{
            background-color: #f1f2f6;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 11px;
        }}

        pre {{
            background-color: #f1f2f6;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            border-left: 4px solid #3498db;
            font-size: 11px;
            line-height: 1.4;
            page-break-inside: avoid;
        }}

        pre code {{
            background-color: transparent;
            padding: 0;
        }}

        blockquote {{
            border-left: 4px solid #3498db;
            padding-left: 20px;
            margin: 20px 0;
            color: #7f8c8d;
            font-style: italic;
        }}

        .title-page {{
            text-align: center;
            padding-top: 100px;
            page-break-after: always;
            min-height: 90vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}

        .title-page h1 {{
            font-size: 28px;
            border: none;
            color: #2c3e50;
            margin-bottom: 20px;
        }}

        .title-page h2 {{
            font-size: 18px;
            border: none;
            color: #7f8c8d;
            margin-top: 40px;
            font-weight: normal;
        }}

        @media print {{
            body {{
                font-size: 11pt;
            }}
            
            h1 {{
                font-size: 16pt;
            }}
            
            h2 {{
                font-size: 14pt;
            }}
            
            h3 {{
                font-size: 12pt;
            }}
            
            img {{
                max-width: 500px !important;
                page-break-inside: avoid;
            }}
            
            table {{
                page-break-inside: avoid;
            }}
            
            pre {{
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
    <div class="title-page">
        <h1>Discovering Fundamental Properties of Artificial Intelligence</h1>
        <h2>A Comprehensive Multi-Phase Investigation into Consciousness, 
            Emergence, Energy Dynamics, and Value Creation in Language Models</h2>
        <div class="authors" style="font-size: 14px; margin-top: 60px; color: #2c3e50;">
            <strong>Authors:</strong> DP¹, Claude (Anthropic)²<br><br>
            ¹Independent Researcher<br>
            ²AI Research Assistant, Anthropic
        </div>
        <div class="metadata" style="font-size: 12px; margin-top: 80px; color: #7f8c8d;">
            <strong>Date:</strong> July 15, 2025<br>
            <strong>Research Duration:</strong> ~5.5 hours autonomous experimentation<br>
            <strong>Models Tested:</strong> phi3:mini, gemma:2b, tinyllama:latest, qwen2.5:0.5b<br><br>
            <strong>Complete experimental data and code:</strong><br>
            https://github.com/dp-web4/ai-dna-discovery<br><br>
            <strong>Synchronism Framework:</strong><br>
            https://dpcars.net/synchronism/
        </div>
    </div>
    
    {toc_html}
    
    {html}
</body>
</html>
"""
    
    print("Saving final HTML...")
    
    # Save HTML version
    with open('comprehensive_research_paper_final.html', 'w', encoding='utf-8') as f:
        f.write(full_html)
    
    print("✅ HTML saved: comprehensive_research_paper_final.html")
    
    # Try to generate PDF
    try:
        import pdfkit
        
        print("Generating PDF with wkhtmltopdf...")
        
        # PDF options - simplified to avoid TOC link issues
        options = {
            'page-size': 'Letter',
            'margin-top': '0.75in',
            'margin-right': '0.75in',
            'margin-bottom': '0.75in',
            'margin-left': '0.75in',
            'encoding': "UTF-8",
            'no-outline': None,
            'print-media-type': None,
            'dpi': 300,
            'image-quality': 94,
            'header-font-size': '8',
            'header-spacing': '5',
            'footer-font-size': '8',
            'footer-spacing': '5',
            'header-center': 'Discovering Fundamental Properties of AI',
            'footer-center': 'Page [page] of [topage]',
            'footer-right': 'DP & Claude (Anthropic), 2025',
            'load-error-handling': 'ignore',
            'load-media-error-handling': 'ignore'
        }
        
        # Generate PDF from HTML string to minimize path issues
        pdfkit.from_string(
            full_html,
            'comprehensive_research_paper_FINAL.pdf',
            options=options
        )
        
        print("✅ PDF generated successfully: comprehensive_research_paper_FINAL.pdf")
        
        # Get file size
        size = os.path.getsize('comprehensive_research_paper_FINAL.pdf')
        size_mb = size / (1024 * 1024)
        print(f"📄 File size: {size_mb:.1f} MB")
        
    except ImportError:
        print("❌ pdfkit not available")
        print("📄 HTML version available for manual PDF conversion")
        
    except Exception as e:
        print(f"❌ PDF generation failed: {e}")
        print("📄 HTML version available: comprehensive_research_paper_final.html")

if __name__ == "__main__":
    generate_pdf()