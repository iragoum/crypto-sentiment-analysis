---
name: rudn-vkr
description: "Create Russian academic thesis documents (ВКР / выпускная квалификационная работа) in GOST-compliant .docx format matching RUDN University standards. Use this skill whenever the user asks to create a ВКР, курсовая работа, дипломная работа, academic report, or any Russian university thesis/dissertation document. Also trigger when the user mentions ГОСТ formatting, титульный лист, список литературы in academic context, or asks for a document formatted like a Russian university paper. Covers: title page (титульный лист), annotation, table of contents, abbreviations table, body with numbered chapters/sections, mathematical equations, code listings (листинги), figure captions, bibliography (список литературы), and appendices (приложения). The output exactly matches RUDN Faculty of Physics, Mathematics and Natural Sciences formatting standards."
---

# RUDN VKR Academic Thesis Document Creator

## Overview

This skill creates `.docx` files that exactly replicate the formatting of Russian academic thesis documents (ВКР) following RUDN University and GOST standards. It uses the `docx` npm package (same as the base docx skill).

**IMPORTANT**: Before using this skill, also read `/mnt/skills/public/docx/SKILL.md` for the base docx-js API reference (setup, validation, critical rules). This skill provides the specific formatting parameters and document structure.

## Page Setup (GOST R 7.0.11-2011 compliant)

```
Paper: A4 (11906 x 16838 DXA)
Margins (in DXA, 1cm ≈ 567 DXA):
  Left:   1701 DXA  (30mm — binding margin)
  Right:   851 DXA  (15mm)
  Top:    1134 DXA  (20mm)
  Bottom: 1134 DXA  (20mm)
Content width: 11906 - 1701 - 851 = 9354 DXA
```

## Typography

```
Body text:
  Font: Times New Roman
  Size: 14pt (28 half-points)
  Line spacing: 1.5 (360 twips = line: 360)
  Paragraph indent (first line): 709 DXA (1.25cm)
  Alignment: Justified
  Color: black (000000)

Chapter headings (Глава / numbered like "1.", "2.", "3."):
  Font: Times New Roman, 14pt, Bold
  Alignment: Centered
  Spacing before: 240, after: 240
  No first-line indent
  Start on new page (pageBreakBefore: true)

Section headings (like "1.1", "2.3"):
  Font: Times New Roman, 14pt, Bold
  Alignment: Left (or Justified)
  Spacing before: 240, after: 120
  First-line indent: 709 DXA (1.25cm)

Subsection headings (like "2.3.1"):
  Font: Times New Roman, 14pt, Bold, Italic
  Alignment: Left
  Spacing before: 120, after: 120
  First-line indent: 709 DXA (1.25cm)
```

## Document Structure & Sections

Read `references/document-structure.md` for the complete section-by-section build guide with code examples for:
1. Title page (титульный лист)
2. Annotation page (аннотация)
3. Table of contents (оглавление)
4. Abbreviations table (список сокращений)
5. Introduction (введение)
6. Body chapters (основная часть)
7. Conclusion (заключение)
8. Bibliography (список литературы)
9. Appendices (приложения)

## Critical Formatting Rules

### Page Numbers
- Position: bottom center
- Font: Times New Roman, 14pt
- First page (title page): no number
- Numbering starts from page 2

### Title Page Layout Rule
```
CRITICAL: The title page two-column student/advisor block follows RIGHT-TO-LEFT reading:
  - LEFT column: student info (Группа, Студ. билет №)
  - RIGHT column: advisor info (Руководитель, degree, signature)
  - "Выполнил студент_____ФИО" spans FULL WIDTH above the table
  - "Автор _____" + "(Подпись)" goes at BOTTOM LEFT
  - Advisor signature + "(Подпись)" goes at BOTTOM RIGHT
  
This mirrors the original RUDN format where the reader's eye starts 
from the right column (advisor/approval authority) then moves left to student details.
```

### Annotation Page (АННОТАЦИЯ) — EXACT FORMAT
```
The annotation page MUST follow this exact structure and spacing:

Line 1: "Федеральное государственное автономное образовательное учреждение"
  → Centered, 12pt, regular
Line 2: "высшего образования"
  → Centered, 12pt, regular
Line 3: "«Российский университет дружбы народов»"
  → Centered, 12pt, BOLD
  
[empty line]

Line 4: "АННОТАЦИЯ"
  → Centered, 14pt, BOLD, spacing before: 240
Line 5: "выпускной квалификационной работы"
  → Centered, 14pt, regular

Line 6: " Фамилии Имени Отчества" (student name in GENITIVE case)
  → Centered, 14pt, with underline or preceded by space+underscores
Line 7: "(фамилия, имя, отчество)"
  → Centered, 12pt, regular (label in parentheses)

Line 8: "на тему: Название темы ВКР"
  → Justified, 14pt, first-line indent 1.25cm
  → "на тему:" in regular, topic title in italics

[empty line]

Body paragraphs of annotation:
  → Justified, 14pt, 1.5 spacing, first-line indent 1.25cm
  → Multiple paragraphs describing the work

[bottom of page]

"Автор ВКР"
  → Left-aligned, 14pt
Signature line: "__________________ __________________"
  → Two underscored blocks side by side  
Labels: "(Подпись)                              (ФИО)"
  → 12pt, spaced to align under the signature blocks
```

### Abbreviations Table (Список сокращений)
- Two-column borderless table
- Left column: abbreviation (bold or regular), right column: full name
- No visible borders (BorderStyle.NONE on all sides)
- Column widths: ~2000 DXA for abbreviation, remainder for description

### Figure Captions
```
Format: "Рисунок X.Y. Description text"
  - Centered below the figure
  - Font: Times New Roman, 14pt (same as body, or 12pt)
  - No bold
  - Spacing before: 120, after: 240
  - Example: "Рисунок 2.1. Вид функции сброса в алгоритме RED"
```

### Code Listings (Листинги)
```
Header format: "Листинг X.Y. Description"
  - Centered, Times New Roman, 14pt
  - Spacing before: 120

Code body:
  - Font: Courier New (or Consolas), 10pt (20 half-points)
  - Line spacing: single (240 twips)
  - No first-line indent
  - Left-aligned
  - Optional: light gray background shading
  - Spacing after code block: 120
```

### Mathematical Equations — ALWAYS USE LaTeX RENDERING
```
CRITICAL: All equations and formulas MUST be rendered via LaTeX → PNG → insert as image.
Never attempt to write equations as plain text runs — they will display incorrectly.

Workflow:
1. Write equation in LaTeX syntax
2. Render to PNG using the LaTeX rendering pipeline (see references/document-structure.md §9)
3. VERIFY the rendered image visually before inserting
4. Insert as ImageRun, centered, with equation number right-aligned via tab stop
5. Ensure no body text overlaps or mixes with the equation image

Equation number: right-aligned in parentheses: (2.1), (2.5), etc.
Spacing before/after equation: 120 DXA
The equation image should be vertically centered in the line.

VERIFICATION CHECKLIST (run after every LaTeX insertion):
  ☐ The PNG renders without errors (check exit code of latex/dvipng)
  ☐ No text runs are mixed into the equation paragraph (only ImageRun + tab + number)
  ☐ The image dimensions fit within content width (max ~8000 DXA wide)
  ☐ The equation number "(X.Y)" appears right-aligned on the same line
  ☐ No clipping or overflow in the rendered image
```

### Bibliography (Список литературы)
```
- Numbered list: 1. 2. 3. etc.
- GOST R 7.0.5-2008 citation format
- Font: Times New Roman, 14pt
- Hanging indent: first line flush, continuation indented
- Or: standard numbered list with indent
```

### In-text References
```
Format: [6], [7,8], [19-21]
- Square brackets with reference number
- No superscript
```

## JavaScript Template Structure

```javascript
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat,
  TabStopType, TabStopPosition,
  HeadingLevel, BorderStyle, WidthType, ShadingType,
  VerticalAlign, PageNumber, PageBreak, ImageRun
} = require("docx");

// === CONSTANTS ===
const FONT = "Times New Roman";
const FONT_CODE = "Courier New";
const SIZE_BODY = 28;        // 14pt in half-points
const SIZE_CODE = 20;        // 10pt
const SIZE_SMALL = 24;       // 12pt
const INDENT_FIRST = 709;    // 1.25cm first-line indent
const LINE_SPACING = 360;    // 1.5 line spacing
const PAGE_WIDTH = 11906;    // A4 width
const PAGE_HEIGHT = 16838;   // A4 height
const MARGIN_LEFT = 1701;    // 30mm
const MARGIN_RIGHT = 851;    // 15mm
const MARGIN_TOP = 1134;     // 20mm
const MARGIN_BOTTOM = 1134;  // 20mm
const CONTENT_WIDTH = PAGE_WIDTH - MARGIN_LEFT - MARGIN_RIGHT; // 9354

// === HELPER FUNCTIONS ===
// See references/helpers.md for all helper functions

// === BUILD DOCUMENT ===
const doc = new Document({
  styles: {
    default: {
      document: {
        run: { font: FONT, size: SIZE_BODY, color: "000000" },
        paragraph: {
          spacing: { line: LINE_SPACING },
          alignment: AlignmentType.JUSTIFIED,
        }
      }
    },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1",
        basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: SIZE_BODY, bold: true, font: FONT },
        paragraph: {
          spacing: { before: 240, after: 240, line: LINE_SPACING },
          alignment: AlignmentType.CENTER,
          outlineLevel: 0
        }
      },
      {
        id: "Heading2", name: "Heading 2",
        basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: SIZE_BODY, bold: true, font: FONT },
        paragraph: {
          spacing: { before: 240, after: 120, line: LINE_SPACING },
          alignment: AlignmentType.JUSTIFIED,
          indent: { firstLine: INDENT_FIRST },
          outlineLevel: 1
        }
      },
      {
        id: "Heading3", name: "Heading 3",
        basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: SIZE_BODY, bold: true, italics: true, font: FONT },
        paragraph: {
          spacing: { before: 120, after: 120, line: LINE_SPACING },
          alignment: AlignmentType.JUSTIFIED,
          indent: { firstLine: INDENT_FIRST },
          outlineLevel: 2
        }
      }
    ]
  },
  sections: [
    // Section 1: Title page (no page number)
    titlePageSection,
    // Section 2: Annotation
    annotationSection,
    // Section 3: Main content (with page numbers)
    mainContentSection
  ]
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("/home/claude/output.docx", buf);
});
```

## Validation

After creating, always validate:
```bash
python /mnt/skills/public/docx/scripts/office/validate.py output.docx
```

## Reference Files

- `references/document-structure.md` — Complete code for each document section (title page, annotation, body, etc.)
- `references/helpers.md` — Helper functions for paragraphs, tables, listings, equations, figures
