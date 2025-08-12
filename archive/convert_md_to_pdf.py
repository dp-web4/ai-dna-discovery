#!/usr/bin/env python3
"""
Convert markdown to PDF using reportlab
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib import colors
import re
import os

def convert_markdown_to_pdf(md_file, pdf_file):
    """Convert markdown file to PDF"""
    
    # Read markdown content
    with open(md_file, 'r') as f:
        content = f.read()
    
    # Create PDF
    doc = SimpleDocTemplate(pdf_file, pagesize=letter,
                          rightMargin=72, leftMargin=72,
                          topMargin=72, bottomMargin=18)
    
    # Container for the 'Flowable' objects
    story = []
    
    # Define styles
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Title',
                            parent=styles['Heading1'],
                            fontSize=24,
                            textColor=colors.HexColor('#1a1a1a'),
                            spaceAfter=30,
                            alignment=TA_CENTER))
    
    styles.add(ParagraphStyle(name='Heading2',
                            parent=styles['Heading2'],
                            fontSize=18,
                            textColor=colors.HexColor('#2c3e50'),
                            spaceAfter=12))
    
    styles.add(ParagraphStyle(name='Heading3',
                            parent=styles['Heading3'],
                            fontSize=14,
                            textColor=colors.HexColor('#34495e'),
                            spaceAfter=10))
    
    styles.add(ParagraphStyle(name='BodyText',
                            parent=styles['Normal'],
                            fontSize=11,
                            textColor=colors.HexColor('#2c3e50'),
                            alignment=TA_JUSTIFY,
                            spaceAfter=12))
    
    # Process content line by line
    lines = content.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines
        if not line:
            story.append(Spacer(1, 0.2*inch))
            i += 1
            continue
        
        # Title (# heading)
        if line.startswith('# ') and not line.startswith('##'):
            text = line[2:].strip()
            story.append(Paragraph(text, styles['Title']))
            story.append(Spacer(1, 0.3*inch))
        
        # Heading 2 (## heading)
        elif line.startswith('## '):
            text = line[3:].strip()
            story.append(Spacer(1, 0.2*inch))
            story.append(Paragraph(text, styles['Heading2']))
        
        # Heading 3 (### heading)
        elif line.startswith('### '):
            text = line[4:].strip()
            story.append(Paragraph(text, styles['Heading3']))
        
        # Horizontal rule
        elif line.startswith('---'):
            story.append(Spacer(1, 0.2*inch))
            story.append(Paragraph('<hr/>', styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
        
        # Image
        elif line.startswith('!['):
            match = re.match(r'!\[([^\]]*)\]\(([^)]+)\)', line)
            if match:
                alt_text, img_path = match.groups()
                if os.path.exists(img_path):
                    img = Image(img_path, width=6*inch, height=4*inch)
                    story.append(img)
                    story.append(Spacer(1, 0.2*inch))
        
        # Bold text
        elif '**' in line:
            # Convert markdown bold to HTML
            line = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', line)
            story.append(Paragraph(line, styles['BodyText']))
        
        # Code block
        elif line.startswith('```'):
            # Skip the opening ```
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code_lines.append(lines[i])
                i += 1
            
            # Format code block
            code_text = '<font name="Courier" size="9">' + '<br/>'.join(code_lines) + '</font>'
            story.append(Paragraph(code_text, styles['Code']))
            story.append(Spacer(1, 0.1*inch))
        
        # Bullet points
        elif line.startswith('- ') or line.startswith('* '):
            text = '• ' + line[2:]
            story.append(Paragraph(text, styles['BodyText']))
        
        # Numbered lists
        elif re.match(r'^\d+\.\s', line):
            story.append(Paragraph(line, styles['BodyText']))
        
        # Regular paragraph
        else:
            # Convert inline code
            line = re.sub(r'`([^`]+)`', r'<font name="Courier">\1</font>', line)
            story.append(Paragraph(line, styles['BodyText']))
        
        i += 1
    
    # Build PDF
    doc.build(story)
    print(f"PDF created: {pdf_file}")

if __name__ == "__main__":
    convert_markdown_to_pdf(
        'weight_analysis_progress_report.md',
        'weight_analysis_progress_report.pdf'
    )