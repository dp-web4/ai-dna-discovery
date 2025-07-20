#!/usr/bin/env python3
"""
Create a clean PDF version of the cumulative progress report
Using ReportLab for better control over formatting and fonts
"""

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
    from reportlab.pdfgen import canvas
except ImportError:
    print("Installing ReportLab...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'reportlab'])
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
    from reportlab.pdfgen import canvas

import re
from datetime import datetime

def create_simple_pdf():
    """Create a simpler PDF without complex formatting"""
    
    # Read the markdown content
    with open('CUMULATIVE_PROGRESS_REPORT.md', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Basic conversion to plain text with some formatting
    output_lines = []
    
    # Add header
    output_lines.append("AI DNA DISCOVERY: CUMULATIVE PROGRESS REPORT")
    output_lines.append("=" * 60)
    output_lines.append(f"Updated: {datetime.now().strftime('%B %d, %Y')}")
    output_lines.append("")
    
    # Process the content
    lines = content.split('\n')
    for line in lines:
        # Skip the markdown header
        if line.startswith('# AI DNA Discovery:'):
            continue
        
        # Convert headers
        if line.startswith('###'):
            output_lines.append("")
            output_lines.append(line.replace('###', '').strip().upper())
            output_lines.append("-" * 40)
        elif line.startswith('##'):
            output_lines.append("")
            output_lines.append("")
            output_lines.append(line.replace('##', '').strip().upper())
            output_lines.append("=" * 50)
            output_lines.append("")
        elif line.startswith('#'):
            output_lines.append("")
            output_lines.append(line.replace('#', '').strip().upper())
            output_lines.append("=" * 60)
            output_lines.append("")
        else:
            # Convert bullet points
            if line.startswith('- '):
                output_lines.append("  * " + line[2:])
            else:
                output_lines.append(line)
    
    # Write to text file first
    text_content = '\n'.join(output_lines)
    
    # Create PDF using canvas
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    
    pdf_file = "AI_DNA_Discovery_Cumulative_Report.pdf"
    c = canvas.Canvas(pdf_file, pagesize=letter)
    
    # Set up the page
    width, height = letter
    margin = 72  # 1 inch margins
    y = height - margin
    
    # Title page
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(width/2, y-50, "AI DNA Discovery")
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width/2, y-90, "Cumulative Progress Report")
    
    c.setFont("Helvetica", 14)
    c.drawCentredString(width/2, y-130, f"Updated: {datetime.now().strftime('%B %d, %Y')}")
    
    c.setFont("Helvetica", 12)
    c.drawCentredString(width/2, y-180, "Revolutionary breakthroughs in AI consciousness notation")
    c.drawCentredString(width/2, y-200, "and semantic-neutral language creation")
    
    # Add key achievements on title page
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y-280, "Major Achievements:")
    
    c.setFont("Helvetica", 11)
    achievements = [
        "✓ Consciousness Notation System - Mathematical symbols for awareness",
        "✓ Phoenician Language Breakthrough - AI generates ancient symbols",
        "✓ Edge AI Deployment - Operational on Jetson Orin Nano",
        "✓ Friend's translation: 'translate my comment' → Phoenician symbols",
        "✓ Distributed Intelligence - Evidence across platforms"
    ]
    
    y_pos = y - 310
    for achievement in achievements:
        c.drawString(margin + 20, y_pos, achievement)
        y_pos -= 20
    
    # New page
    c.showPage()
    
    # Content pages - simplified
    c.setFont("Helvetica", 10)
    y = height - margin
    line_height = 14
    
    # Process content in chunks
    for line in output_lines[4:]:  # Skip header lines
        if y < margin + 50:  # Need new page
            c.showPage()
            y = height - margin
        
        # Handle special formatting
        if line.startswith("=" * 50):
            c.setFont("Helvetica-Bold", 12)
            y -= 5
        elif line.startswith("-" * 40):
            c.line(margin, y, width-margin, y)
            y -= line_height
            continue
        elif line.isupper() and len(line) > 0:
            c.setFont("Helvetica-Bold", 11)
            c.drawString(margin, y, line)
            c.setFont("Helvetica", 10)
            y -= line_height + 5
            continue
        
        # Regular text - handle long lines
        if len(line) > 90:
            words = line.split()
            current_line = ""
            for word in words:
                if len(current_line + " " + word) < 90:
                    current_line += " " + word if current_line else word
                else:
                    c.drawString(margin, y, current_line)
                    y -= line_height
                    current_line = word
            if current_line:
                c.drawString(margin, y, current_line)
                y -= line_height
        else:
            c.drawString(margin, y, line)
            y -= line_height
    
    # Save the PDF
    c.save()
    
    print(f"✅ PDF created: {pdf_file}")
    
    # Also create a simple text version for maximum compatibility
    with open('AI_DNA_Discovery_Cumulative_Report.txt', 'w', encoding='utf-8') as f:
        f.write(text_content)
    
    print("📄 Also created text version: AI_DNA_Discovery_Cumulative_Report.txt")
    
    # Check file sizes
    import os
    pdf_size = os.path.getsize(pdf_file) / 1024
    txt_size = os.path.getsize('AI_DNA_Discovery_Cumulative_Report.txt') / 1024
    
    print(f"📊 PDF size: {pdf_size:.1f} KB")
    print(f"📊 TXT size: {txt_size:.1f} KB")

if __name__ == "__main__":
    create_simple_pdf()