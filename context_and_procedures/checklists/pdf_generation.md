# PDF Generation Checklist

*Last Updated: 2025-01-24*

## Standard PDF Generation from Markdown

### Pre-Generation Setup
1. [ ] Verify pandoc is installed: `pandoc --version`
2. [ ] Check input markdown file exists
3. [ ] Review markdown for special characters (emojis, Unicode)
4. [ ] Decide on output filename (no colons!)

### Basic PDF Generation
```bash
pandoc input.md -o output.pdf
```

### Professional PDF with Formatting
```bash
pandoc input.md \
  -o output.pdf \
  --pdf-engine=xelatex \
  -V geometry:margin=1in \
  -V mainfont="DejaVu Sans" \
  -V monofont="DejaVu Sans Mono" \
  --highlight-style=tango
```

### For Documents with Images
1. [ ] Ensure images use relative paths
2. [ ] Check image files exist
3. [ ] Add image sizing to pandoc:
   ```bash
   pandoc input.md -o output.pdf \
     --pdf-engine=xelatex \
     -V geometry:margin=1in \
     --highlight-style=tango \
     -V graphics=true
   ```

### For Documents with Phoenician Characters
```bash
cd /home/dp/ai-workspace/ai-agents/ai-dna-discovery
pandoc input.md -o output_with_phoenician.pdf \
  --pdf-engine=xelatex \
  -V geometry:margin=1in \
  -H fonts/fallback.tex \
  --highlight-style=tango
```

## Common Issues and Solutions

### Issue: Blank Pages
- [ ] Check CSS for `page-break-before: always`
- [ ] Change to `page-break-before: avoid` for headers
- [ ] Remove manual page breaks from markdown

### Issue: Images Too Large/Bleeding
- [ ] Set max-width in CSS: `max-width: 600px`
- [ ] Add `page-break-inside: avoid` to image containers
- [ ] Consider reducing image resolution

### Issue: Table of Contents Links
- [ ] For web links: ensure proper HTML generation
- [ ] For PDF: use `--toc` flag for auto-generation
- [ ] For passive TOC: create as plain list without links

### Issue: Missing Characters (Emojis, Special Unicode)
- [ ] Warning is normal - doesn't affect PDF creation
- [ ] Use XeLaTeX engine for better Unicode support
- [ ] Consider replacing with ASCII alternatives if critical

### Issue: File Too Large
- [ ] Check for embedded images (compress first)
- [ ] Remove unnecessary blank pages
- [ ] Consider splitting into multiple PDFs

## Step-by-Step Procedure

### 1. Prepare the Markdown
- [ ] Review content for completeness
- [ ] Check all links are valid
- [ ] Ensure images are properly referenced
- [ ] Remove any HTML that might conflict

### 2. Test Basic Generation
```bash
# Simple test first
pandoc input.md -o test_output.pdf
```
- [ ] Check if PDF generates
- [ ] Review for obvious issues

### 3. Apply Formatting
```bash
# Professional formatting
pandoc input.md \
  -o output.pdf \
  --pdf-engine=xelatex \
  -V geometry:margin=1in \
  -V mainfont="DejaVu Sans" \
  -V monofont="DejaVu Sans Mono" \
  --highlight-style=tango
```

### 4. Review Output
- [ ] Check page breaks are sensible
- [ ] Verify images are properly sized
- [ ] Ensure code blocks are formatted
- [ ] Check headers and footers

### 5. Create Backup
- [ ] Always create .txt backup:
  ```bash
  cp input.md output_backup.txt
  ```
- [ ] Document any special generation flags used

## Advanced Options

### Adding Custom Headers/Footers
```bash
pandoc input.md -o output.pdf \
  --pdf-engine=xelatex \
  -V geometry:margin=1in \
  --template=custom_template.tex
```

### Creating Slide Decks
```bash
pandoc input.md -o slides.pdf \
  -t beamer \
  --pdf-engine=xelatex
```

### Multi-Column Layout
```bash
pandoc input.md -o output.pdf \
  --pdf-engine=xelatex \
  -V classoption=twocolumn
```

## What NOT to Do

- ❌ Don't use wkhtmltopdf (creates huge files)
- ❌ Don't install new Python packages for PDF generation
- ❌ Don't use pdfkit (requires wkhtmltopdf)
- ❌ Don't ignore Unicode warnings (note them, but they're OK)

## Post-Generation Checklist

1. [ ] Open PDF to verify content
2. [ ] Check file size is reasonable
3. [ ] Verify all sections are included
4. [ ] Test links (if any) work
5. [ ] Create backup copy
6. [ ] Document generation command used

---

**Note**: If warnings appear about missing characters, that's normal for emojis and special Unicode. The PDF will still generate correctly.