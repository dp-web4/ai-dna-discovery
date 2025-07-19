# Instructions for Creating PDF from Markdown

## Option 1: Using Pandoc (Recommended)

If you have pandoc installed (or can install it):

```bash
# Install pandoc if needed
sudo apt-get install pandoc texlive-latex-base texlive-fonts-recommended

# Convert to PDF
cd papers
pandoc memory_and_awareness.md -o Memory_Systems_as_Foundation_of_Machine_Awareness.pdf \
  --pdf-engine=pdflatex \
  --variable geometry:margin=1in \
  --variable fontsize=11pt \
  --variable linkcolor=blue \
  --highlight-style=tango \
  --toc
```

## Option 2: Using Online Converters

1. **Markdown to PDF**: https://md2pdf.netlify.app/
   - Copy the content from `memory_and_awareness.md`
   - Paste and convert

2. **HackMD**: https://hackmd.io/
   - Create new note
   - Paste markdown
   - Export as PDF

3. **Dillinger**: https://dillinger.io/
   - Import markdown file
   - Export as PDF

## Option 3: Using VS Code

If you have VS Code:
1. Install "Markdown PDF" extension
2. Open `memory_and_awareness.md`
3. Right-click → "Markdown PDF: Export (pdf)"

## Option 4: Google Docs

1. Copy the markdown content
2. Paste into Google Docs
3. Format headings (Ctrl+Alt+1,2,3)
4. File → Download → PDF

## Option 5: Using Python (Markdown2PDF)

```bash
pip install markdown2pdf
md2pdf memory_and_awareness.md
```

## Formatting Tips for PDF

The paper has been structured to work well with any converter:
- Clear heading hierarchy (# ## ###)
- Code blocks properly fenced with ```
- Lists and bullet points
- No complex tables or special formatting

---

The LinkedIn article is ready to copy/paste: `memory_awareness_linkedin.md`