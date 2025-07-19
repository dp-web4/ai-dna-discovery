#!/usr/bin/env python3
"""
Convert the memory awareness paper to PDF format
"""

import os
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.platypus import Table, TableStyle, Preformatted
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self.pages = []
        
    def showPage(self):
        self.pages.append(dict(self.__dict__))
        self._startPage()
        
    def save(self):
        page_count = len(self.pages)
        for page in self.pages:
            self.__dict__.update(page)
            self.draw_page_number(page_count)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)
        
    def draw_page_number(self, page_count):
        self.setFont("Helvetica", 9)
        self.drawRightString(
            letter[0] - 0.75*inch,
            0.75*inch,
            f"Page {self._pageNumber} of {page_count}"
        )

def create_pdf():
    # Read the markdown content
    with open('memory_and_awareness.md', 'r') as f:
        content = f.read()
    
    # Create PDF
    pdf_filename = "Memory_Systems_as_Foundation_of_Machine_Awareness.pdf"
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=36,
    )
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=24,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=24,
    )
    
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=10,
        spaceBefore=16,
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=11,
        leading=16,
        alignment=TA_JUSTIFY,
        spaceAfter=12,
    )
    
    code_style = ParagraphStyle(
        'Code',
        parent=styles['Code'],
        fontSize=9,
        fontName='Courier',
        backgroundColor=colors.HexColor('#f5f5f5'),
        borderColor=colors.HexColor('#ddd'),
        borderWidth=1,
        borderPadding=10,
        spaceAfter=12,
        spaceBefore=12,
    )
    
    # Parse and format content
    lines = content.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Title
        if line.startswith('# ') and not line.startswith('## '):
            text = line[2:].strip()
            elements.append(Paragraph(text, title_style))
            elements.append(Spacer(1, 0.2*inch))
        
        # Section headings
        elif line.startswith('## '):
            text = line[3:].strip()
            elements.append(Paragraph(text, heading1_style))
        
        # Subsection headings
        elif line.startswith('### '):
            text = line[4:].strip()
            elements.append(Paragraph(text, heading2_style))
        
        # Code blocks
        elif line.startswith('```'):
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code_lines.append(lines[i])
                i += 1
            code_text = '\n'.join(code_lines)
            if code_text.strip():
                elements.append(Preformatted(code_text, code_style))
        
        # Bullet points
        elif line.startswith('- '):
            text = f"• {line[2:]}"
            elements.append(Paragraph(text, body_style))
        
        # Numbered lists
        elif line and line[0].isdigit() and '. ' in line:
            elements.append(Paragraph(line, body_style))
        
        # Regular paragraphs
        elif line:
            # Handle bold text
            line = line.replace('**', '<b>', 1)
            while '**' in line:
                line = line.replace('**', '</b>', 1)
                line = line.replace('**', '<b>', 1)
            
            # Handle italic text
            line = line.replace('*', '<i>', 1)
            while '*' in line:
                line = line.replace('*', '</i>', 1)
                line = line.replace('*', '<i>', 1)
            
            elements.append(Paragraph(line, body_style))
        
        # Empty lines
        elif not line and i > 0:
            elements.append(Spacer(1, 0.1*inch))
        
        i += 1
    
    # Add metadata
    elements.insert(0, Spacer(1, 0.5*inch))
    metadata = f"""
    <para align="center">
    <font size="10" color="#666666">
    AI DNA Discovery Project<br/>
    {datetime.now().strftime('%B %Y')}<br/>
    </font>
    </para>
    """
    elements.insert(1, Paragraph(metadata, styles['Normal']))
    elements.insert(2, Spacer(1, 0.5*inch))
    
    # Build PDF
    doc.build(elements, canvasmaker=NumberedCanvas)
    print(f"✅ PDF created: {pdf_filename}")
    
    return pdf_filename

if __name__ == "__main__":
    os.chdir('papers')
    create_pdf()