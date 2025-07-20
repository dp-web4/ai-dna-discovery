#!/bin/bash
# Generate PDF with better font support for Phoenician and symbols

echo "Generating PDF with enhanced font support..."

# Try with multiple font options and fallbacks
pandoc COMPREHENSIVE_REPORT.md \
  -o AI_DNA_Discovery_Comprehensive_Report.pdf \
  --pdf-engine=xelatex \
  -V geometry:margin=1in \
  -V mainfont="DejaVu Sans" \
  -V monofont="DejaVu Sans Mono" \
  -V mathfont="DejaVu Sans" \
  --highlight-style=tango \
  -V documentclass=article \
  -V fontsize=11pt \
  --variable urlcolor=blue

echo "PDF generation complete!"
echo ""
echo "Note: Some ancient scripts like Phoenician may still show as boxes"
echo "due to limited font support. The content is preserved in the document."
echo ""
echo "For full Phoenician support, consider:"
echo "1. Installing fonts-noto-extra package"
echo "2. Using a PDF viewer with better Unicode support"
echo "3. Viewing the HTML or Markdown versions instead"