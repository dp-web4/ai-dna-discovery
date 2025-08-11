#!/usr/bin/env python3
"""
Simple markdown to PDF conversion using available tools
"""

import subprocess
import os

def convert_with_pandoc():
    """Try pandoc if available"""
    try:
        # Check if pandoc is installed
        subprocess.run(['which', 'pandoc'], check=True, capture_output=True)
        
        print("Using pandoc for conversion...")
        
        # Convert with pandoc
        cmd = [
            'pandoc',
            'CUMULATIVE_PROGRESS_REPORT.md',
            '-o', 'AI_DNA_Discovery_Cumulative_Report.pdf',
            '--pdf-engine=xelatex',
            '-V', 'geometry:margin=1in',
            '-V', 'mainfont=DejaVu Sans',
            '-V', 'monofont=DejaVu Sans Mono',
            '--highlight-style=tango'
        ]
        
        subprocess.run(cmd, check=True)
        print("✅ PDF created with pandoc!")
        return True
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def convert_with_markdown_pdf():
    """Use markdown-pdf if available"""
    try:
        import markdown2
        import pdfkit
        
        print("Using markdown/pdfkit for conversion...")
        
        # Read markdown
        with open('CUMULATIVE_PROGRESS_REPORT.md', 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Convert to HTML
        html = markdown2.markdown(md_content, extras=['tables', 'fenced-code-blocks'])
        
        # Add minimal CSS
        html_with_style = f"""
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 40px; }}
                h1 {{ color: #333; border-bottom: 2px solid #333; }}
                h2 {{ color: #555; margin-top: 30px; }}
                h3 {{ color: #666; }}
                code {{ background: #f4f4f4; padding: 2px 4px; }}
                pre {{ background: #f4f4f4; padding: 10px; overflow-x: auto; }}
                ul, ol {{ margin-left: 30px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background: #f4f4f4; }}
            </style>
        </head>
        <body>
            {html}
        </body>
        </html>
        """
        
        # Convert to PDF
        pdfkit.from_string(html_with_style, 'AI_DNA_Discovery_Cumulative_Report.pdf')
        print("✅ PDF created with markdown/pdfkit!")
        return True
        
    except Exception as e:
        print(f"markdown/pdfkit failed: {e}")
        return False

def create_text_version():
    """Create a clean text version as fallback"""
    print("Creating text version...")
    
    with open('CUMULATIVE_PROGRESS_REPORT.md', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Clean up markdown syntax
    import re
    
    # Remove markdown image syntax
    content = re.sub(r'!\[.*?\]\(.*?\)', '', content)
    
    # Convert headers
    content = re.sub(r'^### (.+)$', r'\n\1\n' + '-'*40, content, flags=re.MULTILINE)
    content = re.sub(r'^## (.+)$', r'\n\n\1\n' + '='*50 + '\n', content, flags=re.MULTILINE)
    content = re.sub(r'^# (.+)$', r'\n\n\1\n' + '='*60 + '\n', content, flags=re.MULTILINE)
    
    # Convert bold
    content = re.sub(r'\*\*(.+?)\*\*', r'\1', content)
    
    # Convert links
    content = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', content)
    
    # Save text version
    with open('AI_DNA_Discovery_Cumulative_Report.txt', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Text version created: AI_DNA_Discovery_Cumulative_Report.txt")
    
    # Check size
    size = os.path.getsize('AI_DNA_Discovery_Cumulative_Report.txt') / 1024
    print(f"📄 Text file size: {size:.1f} KB")

def main():
    print("Converting Cumulative Progress Report to shareable format...")
    
    # Try different conversion methods
    if not convert_with_pandoc():
        if not convert_with_markdown_pdf():
            print("\nPDF conversion tools not available.")
            print("Creating text version instead...")
    
    # Always create text version as backup
    create_text_version()
    
    # Check if PDF was created
    if os.path.exists('AI_DNA_Discovery_Cumulative_Report.pdf'):
        pdf_size = os.path.getsize('AI_DNA_Discovery_Cumulative_Report.pdf') / 1024
        print(f"\n✅ PDF created successfully!")
        print(f"📄 PDF size: {pdf_size:.1f} KB")
        print(f"📍 Location: {os.path.abspath('AI_DNA_Discovery_Cumulative_Report.pdf')}")
    
    print("\n📤 Ready to share!")
    print("You can share either the PDF or TXT version of the cumulative report.")

if __name__ == "__main__":
    main()