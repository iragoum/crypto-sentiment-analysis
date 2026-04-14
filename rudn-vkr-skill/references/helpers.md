# Helper Functions Reference

Reusable utility functions for RUDN VKR document generation.

## Quick Reference Table

| Function | Purpose |
|----------|---------|
| `bodyParagraph(text)` | Standard justified paragraph with 1.25cm indent |
| `centeredParagraph(text, opts)` | Centered text (titles, captions) |
| `chapterHeading(num, title)` | Level 1 heading, centered, bold, new page |
| `sectionHeading(num, title)` | Level 2 heading, bold |
| `subsectionHeading(num, title)` | Level 3 heading, bold italic |
| `createListing(num, caption, lines)` | Code listing with caption |
| `figureCaptionParagraph(num, caption)` | "Рисунок X.Y. Caption" |
| `equationFromImage(path, num, w, h)` | LaTeX-rendered equation as image (PREFERRED) |
| `simpleEquationParagraph(text, num)` | Plain text equation (ONLY for trivial w(t) style) |
| `createAbbreviationsTable(items)` | Borderless 2-column abbreviations |
| `emptyLine()` | Blank paragraph with 1.5 spacing |
| `createBibEntry(num, text)` | Numbered bibliography entry |

## Formatting Quick Reference

### DXA Conversion Table

| Measurement | DXA Value |
|-------------|-----------|
| 1 cm | 567 DXA |
| 1 inch | 1440 DXA |
| 1 mm | 56.7 DXA |
| 1.25 cm (indent) | 709 DXA |
| 30 mm (left margin) | 1701 DXA |
| 15 mm (right margin) | 851 DXA |
| 20 mm (top/bottom margin) | 1134 DXA |

### Font Size Conversion (half-points)

| Points | Half-points |
|--------|-------------|
| 10pt | 20 |
| 11pt | 22 |
| 12pt | 24 |
| 13pt | 26 |
| 14pt | 28 |
| 16pt | 32 |
| 18pt | 36 |

### Line Spacing Values

| Spacing | Twips |
|---------|-------|
| Single (1.0) | 240 |
| 1.15 | 276 |
| 1.5 | 360 |
| Double (2.0) | 480 |

## LaTeX Equation Verification Checklist

**Run after EVERY equation insertion:**

```bash
#!/bin/bash
# verify_equations.sh — call after generating the docx
DOCX="$1"

python3 -c "
import zipfile, sys
with zipfile.ZipFile('$DOCX') as z:
    images = [n for n in z.namelist() if n.startswith('word/media/')]
    print(f'Equation images found: {len(images)}')
    errors = 0
    for img in images:
        data = z.read(img)
        if len(data) < 200:
            print(f'  ERROR: {img} is only {len(data)} bytes — likely corrupted')
            errors += 1
        else:
            print(f'  OK: {img} ({len(data)} bytes)')
    if errors:
        print(f'FAIL: {errors} corrupted image(s)')
        sys.exit(1)
    print('PASS: All equation images OK')
"
```

**LaTeX rendering helper function:**

```bash
render_equation() {
  local LATEX_EXPR="$1"
  local OUTPUT_PNG="$2"
  local TMPDIR=$(mktemp -d)
  
  cat > "$TMPDIR/eq.tex" << EOFLATEX
\\documentclass[12pt]{article}
\\usepackage{amsmath,amssymb,amsfonts}
\\pagestyle{empty}
\\begin{document}
\\begin{displaymath}
$LATEX_EXPR
\\end{displaymath}
\\end{document}
EOFLATEX

  cd "$TMPDIR"
  latex -interaction=nonstopmode eq.tex > /dev/null 2>&1
  if [ $? -ne 0 ]; then
    echo "ERROR: LaTeX compilation failed for: $LATEX_EXPR"
    cat eq.log | grep "^!" 
    return 1
  fi
  
  dvipng -D 300 -T tight -bg Transparent -o "$OUTPUT_PNG" eq.dvi > /dev/null 2>&1
  if [ ! -f "$OUTPUT_PNG" ]; then
    echo "ERROR: dvipng failed to create $OUTPUT_PNG"
    return 1
  fi
  
  echo "OK: Rendered $OUTPUT_PNG"
  rm -rf "$TMPDIR"
  return 0
}

# Usage:
# render_equation "\\frac{dE[w(t)]}{dt} = \\frac{1}{T}" "/tmp/eq_2_1.png"
```

## Common Patterns

### Multi-format paragraph (body text with inline formatting)

```javascript
// Example: "В статье [6] описывается принцип функционирования алгоритма RED"
new Paragraph({
  alignment: AlignmentType.JUSTIFIED,
  spacing: { line: 360, after: 0 },
  indent: { firstLine: 709 },
  children: [
    new TextRun({ font: "Times New Roman", size: 28, text: "В статье [6] описывается принцип функционирования алгоритма " }),
    new TextRun({ font: "Times New Roman", size: 28, bold: true, text: "RED" }),
    new TextRun({ font: "Times New Roman", size: 28, text: "." }),
  ]
})
```

### Enumerated list within body text

```javascript
// "Основными задачами данной работы являются:"
// then numbered items with indent
function createNumberedItems(items) {
  return items.map((item, i) => new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { line: 360, after: 0 },
    indent: { firstLine: 709 },
    children: [new TextRun({
      font: "Times New Roman", size: 28,
      text: `${i + 1}. ${item}`
    })]
  }));
}
```

### Bullet list with dash (тире)

```javascript
// Russian academic style uses "⎯" (em-dash) or "—" as bullets
function createDashList(items) {
  return items.map(item => new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { line: 360, after: 0 },
    indent: { firstLine: 709 },
    children: [new TextRun({
      font: "Times New Roman", size: 28,
      text: `⎯ ${item}`
    })]
  }));
}
```

### Table with borders (for data tables in body)

```javascript
function createDataTable(headers, rows, columnWidths) {
  const border = { style: BorderStyle.SINGLE, size: 1, color: "000000" };
  const borders = { top: border, bottom: border, left: border, right: border };

  const headerRow = new TableRow({
    children: headers.map((h, i) => new TableCell({
      borders,
      width: { size: columnWidths[i], type: WidthType.DXA },
      margins: { top: 40, bottom: 40, left: 80, right: 80 },
      verticalAlign: VerticalAlign.CENTER,
      children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { line: 240, after: 0 },
        children: [new TextRun({ font: "Times New Roman", size: 28, bold: true, text: h })]
      })]
    }))
  });

  const dataRows = rows.map(row =>
    new TableRow({
      children: row.map((cell, i) => new TableCell({
        borders,
        width: { size: columnWidths[i], type: WidthType.DXA },
        margins: { top: 40, bottom: 40, left: 80, right: 80 },
        children: [new Paragraph({
          alignment: AlignmentType.LEFT,
          spacing: { line: 240, after: 0 },
          children: [new TextRun({ font: "Times New Roman", size: 28, text: cell })]
        })]
      }))
    })
  );

  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths,
    rows: [headerRow, ...dataRows]
  });
}
```

## Page Number Formatting

To get correct page numbering (title page = page 1 but not displayed):

```javascript
// Title page section: no footer
const titleSection = {
  properties: {
    page: {
      size: { width: 11906, height: 16838 },
      margin: { top: 1134, right: 851, bottom: 1134, left: 1701 },
      pageNumbers: { start: 1 }  // Title page is page 1
    }
  },
  // NO footers defined
  children: [/* title page */]
};

// Main content section: page numbers visible
const mainSection = {
  properties: {
    page: {
      size: { width: 11906, height: 16838 },
      margin: { top: 1134, right: 851, bottom: 1134, left: 1701 },
    },
    type: SectionType.NEXT_PAGE,
  },
  footers: {
    default: new Footer({
      children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({
          font: "Times New Roman", size: 28,
          children: [PageNumber.CURRENT]
        })]
      })]
    })
  },
  children: [/* annotation, TOC, body, bibliography, appendices */]
};
```
