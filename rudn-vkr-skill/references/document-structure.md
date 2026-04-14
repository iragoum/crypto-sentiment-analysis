# Document Structure Reference

Complete JavaScript code examples for each section of a RUDN VKR document using `docx-js`.

## Table of Contents
1. [Constants & Setup](#constants)
2. [Title Page (Титульный лист)](#title-page)
3. [Annotation Page (Аннотация)](#annotation)
4. [Table of Contents (Оглавление)](#toc)
5. [Abbreviations (Список сокращений)](#abbreviations)
6. [Body Text & Headings](#body)
7. [Code Listings (Листинги)](#listings)
8. [Figure Captions (Рисунки)](#figures)
9. [Equations (Формулы)](#equations)
10. [Bibliography (Список литературы)](#bibliography)
11. [Appendices (Приложения)](#appendices)
12. [Page Numbers & Headers/Footers](#page-numbers)
13. [Full Assembly Example](#assembly)

---

<a id="constants"></a>
## 1. Constants & Setup

```javascript
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat,
  TabStopType, TabStopPosition, PositionalTab,
  PositionalTabAlignment, PositionalTabRelativeTo, PositionalTabLeader,
  HeadingLevel, BorderStyle, WidthType, ShadingType,
  VerticalAlign, PageNumber, PageBreak, ImageRun,
  TableOfContents, SectionType
} = require("docx");

const FONT = "Times New Roman";
const FONT_CODE = "Courier New";
const SIZE_14 = 28;    // 14pt body
const SIZE_13 = 26;    // 13pt (alternate body)
const SIZE_12 = 24;    // 12pt (small)
const SIZE_10 = 20;    // 10pt (code)
const SIZE_16 = 32;    // 16pt (title emphasis)
const INDENT = 709;    // 1.25cm first-line indent
const SPACING_15 = 360; // 1.5 line spacing
const SPACING_1 = 240;  // single line spacing

const PAGE_W = 11906;
const PAGE_H = 16838;
const M_LEFT = 1701;   // 30mm
const M_RIGHT = 851;   // 15mm
const M_TOP = 1134;    // 20mm
const M_BOT = 1134;    // 20mm
const CONTENT_W = PAGE_W - M_LEFT - M_RIGHT; // 9354 DXA

const NO_BORDER = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const NO_BORDERS = { top: NO_BORDER, bottom: NO_BORDER, left: NO_BORDER, right: NO_BORDER };

const pageProps = {
  page: {
    size: { width: PAGE_W, height: PAGE_H },
    margin: { top: M_TOP, right: M_RIGHT, bottom: M_BOT, left: M_LEFT }
  }
};
```

---

<a id="title-page"></a>
## 2. Title Page (Титульный лист)

The title page has NO page number. It is a separate section.

Key layout:
- Ministry name: centered, CAPS, 12-13pt
- University name: centered, CAPS, 12-13pt
- Faculty & department: centered, 13-14pt
- "Допустить к защите" block: right-aligned
- Thesis title: centered, 14pt bold
- Student & advisor info: left-aligned with underscores
- City and year: centered at bottom

```javascript
function createTitlePage(params) {
  // params: { ministry, university, faculty, department, headName, headTitle,
  //           thesisType, direction, topic, studentName, group, studentId,
  //           advisorName, advisorDegree, city, year }

  const centered = (text, opts = {}) => new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { line: SPACING_15, after: 0, before: 0 },
    children: [new TextRun({ font: FONT, size: opts.size || SIZE_14, ...opts, text })]
  });

  const emptyLine = () => new Paragraph({
    spacing: { line: SPACING_15, after: 0, before: 0 },
    children: [new TextRun({ font: FONT, size: SIZE_14, text: "" })]
  });

  // "Допустить к защите" block — right-aligned
  const approvalBlock = [
    new Paragraph({
      alignment: AlignmentType.RIGHT,
      spacing: { line: SPACING_15, after: 0 },
      children: [new TextRun({ font: FONT, size: SIZE_14, text: "«Допустить к защите»" })]
    }),
    new Paragraph({
      alignment: AlignmentType.RIGHT,
      spacing: { line: SPACING_15, after: 0 },
      children: [new TextRun({ font: FONT, size: SIZE_14, text: "Заведующий кафедрой" })]
    }),
    new Paragraph({
      alignment: AlignmentType.RIGHT,
      spacing: { line: SPACING_15, after: 0 },
      children: [new TextRun({ font: FONT, size: SIZE_14, text: params.department || "прикладной информатики" })]
    }),
    new Paragraph({
      alignment: AlignmentType.RIGHT,
      spacing: { line: SPACING_15, after: 0 },
      children: [new TextRun({ font: FONT, size: SIZE_14, text: "и теории вероятностей" })]
    }),
    new Paragraph({
      alignment: AlignmentType.RIGHT,
      spacing: { line: SPACING_15, after: 0 },
      children: [new TextRun({ font: FONT, size: SIZE_14, text: params.headTitle || "д.т.н., профессор" })]
    }),
    new Paragraph({
      alignment: AlignmentType.RIGHT,
      spacing: { line: SPACING_15, after: 0 },
      children: [new TextRun({ font: FONT, size: SIZE_14, text: `____________ ${params.headName}` })]
    }),
    new Paragraph({
      alignment: AlignmentType.RIGHT,
      spacing: { line: SPACING_15, after: 0 },
      children: [new TextRun({ font: FONT, size: SIZE_14, text: "«___» ____________ 20__ г." })]
    }),
  ];

  // Main title block
  const titleBlock = [
    emptyLine(),
    centered("Выпускная квалификационная работа", { size: SIZE_14 }),
    centered(params.thesisType || "бакалавра", { size: SIZE_14 }),
    emptyLine(),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { line: SPACING_15, after: 0 },
      children: [new TextRun({ font: FONT, size: SIZE_14,
        text: `Направление ${params.direction || '02.03.02 «Фундаментальная информатика и информационные технологии»'}`
      })]
    }),
    emptyLine(),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { line: SPACING_15, after: 0 },
      children: [new TextRun({ font: FONT, size: SIZE_14,
        text: `ТЕМА «${params.topic}»`
      })]
    }),
  ];

  // === STUDENT NAME — FULL WIDTH LINE (above the 2-column table) ===
  const studentFullWidth = new Paragraph({
    alignment: AlignmentType.LEFT,
    spacing: { line: SPACING_15, after: 0 },
    children: [new TextRun({ font: FONT, size: SIZE_14, text: `Выполнил студент_____${params.studentName}` })]
  });
  const studentLabel = new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { line: SPACING_15, after: 0 },
    children: [new TextRun({ font: FONT, size: SIZE_12, text: "(Фамилия, имя, отчество)" })]
  });

  // === 2-COLUMN TABLE: LEFT = group/id, RIGHT = advisor ===
  // Layout: RIGHT column has advisor (authority reads first right-to-left)
  //         LEFT column has student group/id details
  const infoTable = new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [Math.floor(CONTENT_W / 2), Math.ceil(CONTENT_W / 2)],
    rows: [
      new TableRow({
        children: [
          // LEFT column: Группа, Студ. билет, Автор + подпись
          new TableCell({
            borders: NO_BORDERS,
            width: { size: Math.floor(CONTENT_W / 2), type: WidthType.DXA },
            children: [
              new Paragraph({
                spacing: { line: SPACING_15, after: 0 },
                children: [new TextRun({ font: FONT, size: SIZE_14, text: `Группа ${params.group}` })]
              }),
              emptyLine(),
              new Paragraph({
                spacing: { line: SPACING_15, after: 0 },
                children: [new TextRun({ font: FONT, size: SIZE_14, text: `Студ. билет № ${params.studentId}` })]
              }),
              emptyLine(),
              emptyLine(),
              emptyLine(),
              emptyLine(),
              new Paragraph({
                spacing: { line: SPACING_15, after: 0 },
                children: [new TextRun({ font: FONT, size: SIZE_14, text: "Автор _________________________" })]
              }),
              new Paragraph({
                alignment: AlignmentType.CENTER,
                spacing: { line: SPACING_15, after: 0 },
                children: [new TextRun({ font: FONT, size: SIZE_12, text: "(Подпись)" })]
              }),
            ]
          }),
          // RIGHT column: Руководитель + advisor details + подпись
          new TableCell({
            borders: NO_BORDERS,
            width: { size: Math.ceil(CONTENT_W / 2), type: WidthType.DXA },
            children: [
              new Paragraph({
                spacing: { line: SPACING_15, after: 0 },
                children: [new TextRun({ font: FONT, size: SIZE_14, text: "Руководитель выпускной" })]
              }),
              new Paragraph({
                spacing: { line: SPACING_15, after: 0 },
                children: [new TextRun({ font: FONT, size: SIZE_14, text: "квалификационной работы" })]
              }),
              new Paragraph({
                spacing: { line: SPACING_15, after: 0 },
                children: [new TextRun({ font: FONT, size: SIZE_14, text: params.advisorName })]
              }),
              new Paragraph({
                spacing: { line: SPACING_15, after: 0 },
                children: [new TextRun({ font: FONT, size: SIZE_12, text: `(${params.advisorDegree})` })]
              }),
              emptyLine(),
              new Paragraph({
                spacing: { line: SPACING_15, after: 0 },
                children: [new TextRun({ font: FONT, size: SIZE_14, text: " _____________________________" })]
              }),
              new Paragraph({
                alignment: AlignmentType.CENTER,
                spacing: { line: SPACING_15, after: 0 },
                children: [new TextRun({ font: FONT, size: SIZE_12, text: "(Подпись)" })]
              }),
            ]
          })
        ]
      })
    ]
  });

  return {
    properties: {
      ...pageProps,
      page: { ...pageProps.page, pageNumbers: { start: 1 } }
    },
    children: [
      // === MINISTRY HEADER ===
      centered("МИНИСТЕРСТВО НАУКИ И ВЫСШЕГО ОБРАЗОВАНИЯ", { size: SIZE_12 }),
      centered("РОССИЙСКОЙ ФЕДЕРАЦИИ", { size: SIZE_12 }),
      centered("ФЕДЕРАЛЬНОЕ ГОСУДАРСТВЕННОЕ АВТОНОМНОЕ ОБРАЗОВАТЕЛЬНОЕ", { size: SIZE_12 }),
      centered("УЧРЕЖДЕНИЕ ВЫСШЕГО ОБРАЗОВАНИЯ", { size: SIZE_12 }),
      centered("«РОССИЙСКИЙ УНИВЕРСИТЕТ ДРУЖБЫ НАРОДОВ»", { size: SIZE_12, bold: true }),
      emptyLine(),
      // === FACULTY & DEPARTMENT ===
      centered(params.faculty || "Факультет физико-математических и естественных наук", { size: SIZE_14 }),
      centered(`Кафедра ${params.department || "прикладной информатики и теории вероятностей"}`, { size: SIZE_14 }),
      emptyLine(),
      // === APPROVAL BLOCK (right-aligned) ===
      ...approvalBlock,
      emptyLine(),
      // === THESIS TYPE & DIRECTION ===
      ...titleBlock,
      emptyLine(),
      // === STUDENT NAME (full-width) ===
      studentFullWidth,
      studentLabel,
      emptyLine(),
      // === 2-COLUMN TABLE: student details LEFT, advisor RIGHT ===
      infoTable,
      emptyLine(),
      emptyLine(),
      // === CITY & YEAR ===
      centered(`г. ${params.city || "Москва"}`, { size: SIZE_14 }),
      centered(`${params.year || new Date().getFullYear()} г.`, { size: SIZE_14 }),
    ]
  };
}
```

---

<a id="annotation"></a>
## 3. Annotation Page (Аннотация) — EXACT FORMAT

The annotation page MUST reproduce this exact structure. Every line matters.

```javascript
function createAnnotationPage(params) {
  // params: {
  //   studentNameGenitive: "Апреутесей Анны Марии Юрьевны" (genitive case!)
  //   topic: "Гибридное моделирование нелинейных систем..."
  //   annotationParagraphs: ["Данная работа посвящена...", "В ходе моделирования..."]
  // }

  const children = [
    // === UNIVERSITY HEADER (same as title page but smaller) ===
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { line: SPACING_15, after: 0, before: 0 },
      children: [new TextRun({ font: FONT, size: SIZE_12, text: "Федеральное государственное автономное образовательное учреждение" })]
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { line: SPACING_15, after: 0 },
      children: [new TextRun({ font: FONT, size: SIZE_12, text: "высшего образования" })]
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { line: SPACING_15, after: 120 },
      children: [new TextRun({ font: FONT, size: SIZE_12, bold: true, text: "«Российский университет дружбы народов»" })]
    }),

    // === "АННОТАЦИЯ" HEADING ===
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { line: SPACING_15, before: 240, after: 0 },
      children: [new TextRun({ font: FONT, size: SIZE_14, bold: true, text: "АННОТАЦИЯ" })]
    }),
    // "выпускной квалификационной работы"
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { line: SPACING_15, after: 0 },
      children: [new TextRun({ font: FONT, size: SIZE_14, text: "выпускной квалификационной работы" })]
    }),

    // === STUDENT NAME (genitive case, with underline space before) ===
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { line: SPACING_15, after: 0 },
      children: [new TextRun({ font: FONT, size: SIZE_14, text: ` ${params.studentNameGenitive}` })]
    }),
    // Label in parentheses
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { line: SPACING_15, after: 120 },
      children: [new TextRun({ font: FONT, size: SIZE_12, text: "(фамилия, имя, отчество)" })]
    }),

    // === TOPIC LINE ===
    // "на тему:" followed by topic in italics
    new Paragraph({
      alignment: AlignmentType.JUSTIFIED,
      spacing: { line: SPACING_15, after: 0 },
      indent: { firstLine: INDENT },
      children: [
        new TextRun({ font: FONT, size: SIZE_14, text: "на тему: " }),
        new TextRun({ font: FONT, size: SIZE_14, italics: true, text: params.topic }),
      ]
    }),

    // === EMPTY LINE BEFORE BODY ===
    emptyLine(),
  ];

  // === ANNOTATION BODY PARAGRAPHS ===
  // Each paragraph: justified, 14pt, 1.5 spacing, 1.25cm indent
  for (const pText of params.annotationParagraphs) {
    children.push(new Paragraph({
      alignment: AlignmentType.JUSTIFIED,
      spacing: { line: SPACING_15, after: 0 },
      indent: { firstLine: INDENT },
      children: [new TextRun({ font: FONT, size: SIZE_14, text: pText.trim() })]
    }));
  }

  // === SIGNATURE BLOCK AT BOTTOM ===
  children.push(
    emptyLine(),
    emptyLine(),
    // "Автор ВКР"
    new Paragraph({
      alignment: AlignmentType.LEFT,
      spacing: { line: SPACING_15, after: 0 },
      children: [new TextRun({ font: FONT, size: SIZE_14, text: "Автор ВКР" })]
    }),
    // Two signature placeholders side by side
    new Paragraph({
      alignment: AlignmentType.LEFT,
      spacing: { line: SPACING_15, after: 0 },
      tabStops: [
        { type: TabStopType.LEFT, position: 4000 }
      ],
      children: [
        new TextRun({ font: FONT, size: SIZE_14, text: "__________________" }),
        new TextRun({ font: FONT, size: SIZE_14, text: "\t__________________" }),
      ]
    }),
    // Labels under signatures
    new Paragraph({
      alignment: AlignmentType.LEFT,
      spacing: { line: SPACING_15, after: 0 },
      tabStops: [
        { type: TabStopType.LEFT, position: 4000 }
      ],
      children: [
        new TextRun({ font: FONT, size: SIZE_12, text: "(Подпись)" }),
        new TextRun({ font: FONT, size: SIZE_12, text: "\t(ФИО)" }),
      ]
    })
  );

  return {
    properties: {
      ...pageProps,
      type: SectionType.NEXT_PAGE,
    },
    children
  };
}
```

### Usage example:
```javascript
createAnnotationPage({
  studentNameGenitive: "Апреутесей Анны Марии Юрьевны",
  topic: "Гибридное моделирование нелинейных систем с управлением в среде OpenModelica",
  annotationParagraphs: [
    "Данная работа посвящена применению непрерывно-дискретного подхода к моделированию нелинейных систем с управлением, в качестве которых выступают системы, состоящие из входящего потока, обрабатываемого согласно протоколу Transmission Control Protocol (TCP), а также маршрутизатора, обрабатывающего трафик по алгоритму типа Random Early Detection (RED).",
    "В ходе моделирования в среде OpenModelica был использован гибридный подход, позволяющий учитывать как непрерывные, так и дискретные элементы системы, например, объект управления с непрерывным характером функционирования и дискретное устройство управления. Для реализации вычислительной и имитационной моделей процесса передачи трафика под управлением классического алгоритма RED, а также некоторых его модификаций, применялся язык Modelica, позволяющий проводить гибридное моделирование относительно простым способом."
  ]
})
```

---

<a id="toc"></a>
## 4. Table of Contents (Оглавление)

```javascript
function createTOCSection() {
  return {
    properties: {
      ...pageProps,
      type: SectionType.NEXT_PAGE,
    },
    children: [
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { line: SPACING_15, before: 0, after: 240 },
        children: [new TextRun({ font: FONT, size: SIZE_14, bold: true, text: "Оглавление" })]
      }),
      new TableOfContents("Оглавление", {
        hyperlink: true,
        headingStyleRange: "1-3",
      }),
    ]
  };
}
```

**Note**: The TOC will only populate when opened in Word and "Update Fields" is used. This is a limitation of the docx format.

---

<a id="abbreviations"></a>
## 5. Abbreviations Table (Список сокращений)

This is a **borderless two-column table** — abbreviation on left, definition on right.

```javascript
function createAbbreviationsTable(abbreviations) {
  // abbreviations: array of { abbr: "TCP", full: "Transmission Control Protocol" }

  const rows = abbreviations.map(item =>
    new TableRow({
      children: [
        new TableCell({
          borders: NO_BORDERS,
          width: { size: 2000, type: WidthType.DXA },
          margins: { top: 40, bottom: 40, left: 0, right: 120 },
          children: [new Paragraph({
            spacing: { line: SPACING_15, after: 0 },
            children: [new TextRun({ font: FONT, size: SIZE_14, text: item.abbr })]
          })]
        }),
        new TableCell({
          borders: NO_BORDERS,
          width: { size: CONTENT_W - 2000, type: WidthType.DXA },
          margins: { top: 40, bottom: 40, left: 120, right: 0 },
          children: [new Paragraph({
            spacing: { line: SPACING_15, after: 0 },
            children: [new TextRun({ font: FONT, size: SIZE_14, text: item.full })]
          })]
        })
      ]
    })
  );

  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { line: SPACING_15, before: 0, after: 240 },
      children: [new TextRun({ font: FONT, size: SIZE_14, bold: true, text: "Список используемых сокращений" })]
    }),
    new Table({
      width: { size: CONTENT_W, type: WidthType.DXA },
      columnWidths: [2000, CONTENT_W - 2000],
      rows
    })
  ];
}
```

---

<a id="body"></a>
## 6. Body Text & Headings

### Regular paragraph
```javascript
function bodyParagraph(text) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { line: SPACING_15, after: 0 },
    indent: { firstLine: INDENT },
    children: [new TextRun({ font: FONT, size: SIZE_14, text })]
  });
}
```

### Paragraph with mixed formatting (bold, italic, references)
```javascript
function mixedParagraph(runs) {
  // runs: array of { text, bold?, italics?, superScript? }
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { line: SPACING_15, after: 0 },
    indent: { firstLine: INDENT },
    children: runs.map(r => new TextRun({
      font: FONT, size: SIZE_14,
      text: r.text,
      bold: r.bold || false,
      italics: r.italics || false,
      superScript: r.superScript || false,
    }))
  });
}
```

### Chapter heading (level 1) — "1. Методы и анализ моделирования"
```javascript
function chapterHeading(number, title) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    alignment: AlignmentType.CENTER,
    spacing: { line: SPACING_15, before: 240, after: 240 },
    pageBreakBefore: true,
    children: [new TextRun({
      font: FONT, size: SIZE_14, bold: true,
      text: `${number}. ${title}`
    })]
  });
}
```

### Section heading (level 2) — "1.1 Обзор исследований"
```javascript
function sectionHeading(number, title) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    alignment: AlignmentType.JUSTIFIED,
    spacing: { line: SPACING_15, before: 240, after: 120 },
    indent: { firstLine: INDENT },
    children: [new TextRun({
      font: FONT, size: SIZE_14, bold: true,
      text: `${number} ${title}`
    })]
  });
}
```

### Subsection heading (level 3) — "2.3.1 Алгоритм Enhanced RED"
```javascript
function subsectionHeading(number, title) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    alignment: AlignmentType.JUSTIFIED,
    spacing: { line: SPACING_15, before: 120, after: 120 },
    indent: { firstLine: INDENT },
    children: [new TextRun({
      font: FONT, size: SIZE_14, bold: true, italics: true,
      text: `${number} ${title}`
    })]
  });
}
```

---

<a id="listings"></a>
## 7. Code Listings (Листинги)

Each listing has a centered caption followed by monospaced code lines.

```javascript
function createListing(number, caption, codeLines) {
  // number: "3.1", caption: "Уравнение для изменения размера окна в RED"
  // codeLines: array of strings

  const elements = [
    // Caption
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { line: SPACING_1, before: 120, after: 60 },
      children: [new TextRun({
        font: FONT, size: SIZE_14,
        text: `Листинг ${number}. ${caption}`
      })]
    }),
  ];

  // Code lines
  for (const line of codeLines) {
    elements.push(new Paragraph({
      alignment: AlignmentType.LEFT,
      spacing: { line: SPACING_1, after: 0 },
      indent: { firstLine: 0 },
      children: [new TextRun({
        font: FONT_CODE, size: SIZE_10,
        text: line || " " // empty lines need a space
      })]
    }));
  }

  // Space after listing
  elements.push(new Paragraph({
    spacing: { line: SPACING_15, after: 0 },
    children: []
  }));

  return elements;
}
```

---

<a id="figures"></a>
## 8. Figure Captions (Рисунки)

```javascript
function figureCaptionParagraph(number, caption) {
  // number: "2.1", caption: "Вид функции сброса в алгоритме RED"
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { line: SPACING_15, before: 120, after: 240 },
    children: [new TextRun({
      font: FONT, size: SIZE_14,
      text: `Рисунок ${number}. ${caption}`
    })]
  });
}

// If you have an actual image file:
function figureWithImage(imagePath, number, caption, widthPx, heightPx) {
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { line: SPACING_15, before: 120, after: 0 },
      children: [new ImageRun({
        type: imagePath.endsWith(".png") ? "png" : "jpg",
        data: fs.readFileSync(imagePath),
        transformation: { width: widthPx, height: heightPx },
        altText: { title: caption, description: caption, name: `fig_${number}` }
      })]
    }),
    figureCaptionParagraph(number, caption)
  ];
}
```

---

<a id="equations"></a>
## 9. Equations (Формулы) — LaTeX Rendering Pipeline

**CRITICAL**: All equations MUST be rendered via LaTeX → PNG → ImageRun. Never write equations as plain TextRun — they will display incorrectly with missing symbols, wrong spacing, and broken formatting.

### Step 1: Render LaTeX to PNG

```bash
# Install LaTeX tools if not present
apt-get install -y texlive-base texlive-latex-extra dvipng 2>/dev/null

# Create a minimal LaTeX file for each equation
cat > /tmp/eq.tex << 'EOF'
\documentclass[12pt]{article}
\usepackage{amsmath,amssymb}
\pagestyle{empty}
\begin{document}
\begin{displaymath}
YOUR_EQUATION_HERE
\end{displaymath}
\end{document}
EOF

# Render to PNG (300 DPI for crisp output)
cd /tmp
latex -interaction=nonstopmode eq.tex
dvipng -D 300 -T tight -bg Transparent -o eq.png eq.dvi

# VERIFY: Check exit code and file exists
if [ $? -ne 0 ] || [ ! -f /tmp/eq.png ]; then
  echo "ERROR: LaTeX rendering failed!"
  exit 1
fi
echo "OK: Equation rendered successfully"
```

### Step 2: Get image dimensions and insert

```javascript
const sharp = require("sharp"); // or use image-size package

// Get dimensions for proper sizing
// Target: equation should fit within ~80% of content width
// At 300 DPI: 1 inch = 300px, 1 DXA = 300/1440 px
async function getEquationSize(pngPath) {
  const sizeOf = require("image-size");
  const dims = sizeOf(pngPath);
  // Scale to fit: max width ~7000 DXA (~4.86 inches)
  const maxWidthPx = 7000 * 300 / 1440; // ~1458 px
  let scale = 1;
  if (dims.width > maxWidthPx) {
    scale = maxWidthPx / dims.width;
  }
  return {
    width: Math.round(dims.width * scale * 1440 / 300 * 0.35),
    height: Math.round(dims.height * scale * 1440 / 300 * 0.35)
  };
}
```

### Step 3: Insert equation with right-aligned number

```javascript
function equationFromImage(pngPath, number, widthPx, heightPx) {
  // CRITICAL: The paragraph must contain ONLY:
  //   1. Tab to center
  //   2. ImageRun (the equation)
  //   3. Tab to right
  //   4. Equation number "(X.Y)"
  // NO other TextRun or content — this prevents text mixing with equation
  
  return new Paragraph({
    alignment: AlignmentType.LEFT,
    spacing: { line: SPACING_15, before: 120, after: 120 },
    indent: { firstLine: 0 },
    tabStops: [
      { type: TabStopType.CENTER, position: Math.floor(CONTENT_W / 2) },
      { type: TabStopType.RIGHT, position: CONTENT_W }
    ],
    children: [
      new TextRun({ font: FONT, size: SIZE_14, text: "\t" }),
      new ImageRun({
        type: "png",
        data: fs.readFileSync(pngPath),
        transformation: { width: widthPx, height: heightPx },
        altText: { title: `eq_${number}`, description: `Equation ${number}`, name: `eq_${number}` }
      }),
      new TextRun({ font: FONT, size: SIZE_14, text: `\t(${number})` }),
    ]
  });
}
```

### Step 4: MANDATORY Verification

After inserting every equation, run this check:

```bash
# Verify the docx contains the equation image
python3 -c "
import zipfile
with zipfile.ZipFile('output.docx') as z:
    images = [n for n in z.namelist() if n.startswith('word/media/')]
    print(f'Images in document: {len(images)}')
    for img in images:
        data = z.read(img)
        print(f'  {img}: {len(data)} bytes')
        if len(data) < 100:
            print('  WARNING: Image too small, may be corrupted!')
"
```

### Fallback: Simple text equations (ONLY for trivial cases)

For very simple expressions like `w(t)` or `q > 0` that need no special symbols, you may use text, but NEVER for:
- Fractions, summations, integrals, matrices
- Greek letters (α, β, τ, etc.)
- Subscripts/superscripts combined with other notation
- Multi-line equation systems
- Any notation involving special mathematical layout

```javascript
// ONLY for trivial inline references like "w(t)" or "q(t) > 0"
function simpleEquationParagraph(equationText, number) {
  return new Paragraph({
    alignment: AlignmentType.LEFT,
    spacing: { line: SPACING_15, before: 120, after: 120 },
    indent: { firstLine: 0 },
    tabStops: [
      { type: TabStopType.CENTER, position: Math.floor(CONTENT_W / 2) },
      { type: TabStopType.RIGHT, position: CONTENT_W }
    ],
    children: [
      new TextRun({ font: FONT, size: SIZE_14, text: "\t" }),
      new TextRun({ font: "Cambria Math", size: SIZE_14, text: equationText }),
      new TextRun({ font: FONT, size: SIZE_14, text: `\t(${number})` }),
    ]
  });
}
```

---

<a id="bibliography"></a>
## 10. Bibliography (Список литературы)

Uses a numbered list format. Each entry is a separate paragraph with a number.

```javascript
function createBibliography(entries) {
  // entries: array of strings, each is a full GOST-formatted reference
  const children = [
    new Paragraph({
      heading: HeadingLevel.HEADING_1,
      alignment: AlignmentType.CENTER,
      spacing: { line: SPACING_15, before: 240, after: 240 },
      pageBreakBefore: true,
      children: [new TextRun({
        font: FONT, size: SIZE_14, bold: true,
        text: "Список литературы"
      })]
    })
  ];

  entries.forEach((entry, i) => {
    children.push(new Paragraph({
      alignment: AlignmentType.JUSTIFIED,
      spacing: { line: SPACING_15, after: 0 },
      indent: { firstLine: INDENT },
      children: [new TextRun({
        font: FONT, size: SIZE_14,
        text: `${i + 1}. ${entry}`
      })]
    }));
  });

  return children;
}
```

---

<a id="appendices"></a>
## 11. Appendices (Приложения)

Each appendix starts on a new page with a centered title.

```javascript
function createAppendixHeader(number, title) {
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { line: SPACING_15, before: 240, after: 120 },
      pageBreakBefore: true,
      children: [new TextRun({
        font: FONT, size: SIZE_14, bold: true,
        text: `Приложение ${number}`
      })]
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { line: SPACING_15, before: 0, after: 240 },
      children: [new TextRun({
        font: FONT, size: SIZE_14, bold: true,
        text: title
      })]
    }),
  ];
}

// Appendix with full code listing
function createAppendixCode(number, title, subtitle, codeLines) {
  const elements = createAppendixHeader(number, title);

  if (subtitle) {
    elements.push(new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { line: SPACING_15, after: 120 },
      children: [new TextRun({ font: FONT, size: SIZE_14, text: subtitle })]
    }));
  }

  // Full code listing
  elements.push(new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { line: SPACING_1, before: 120, after: 60 },
    children: [new TextRun({ font: FONT, size: SIZE_14, text: "Полный листинг программы:" })]
  }));

  for (const line of codeLines) {
    elements.push(new Paragraph({
      alignment: AlignmentType.LEFT,
      spacing: { line: SPACING_1, after: 0 },
      indent: { firstLine: 0 },
      children: [new TextRun({
        font: FONT_CODE, size: SIZE_10,
        text: line || " "
      })]
    }));
  }

  return elements;
}
```

---

<a id="page-numbers"></a>
## 12. Page Numbers & Headers/Footers

Page numbers: bottom center, starting from page 2 of the main content.

```javascript
// For the title page section — NO headers/footers
const titleSection = {
  properties: {
    ...pageProps,
    // No headers/footers defined = no page number
  },
  children: [/* title page content */]
};

// For the main content section — with page numbers
const mainSection = {
  properties: {
    ...pageProps,
    type: SectionType.NEXT_PAGE,
  },
  headers: {
    default: new Header({
      children: [new Paragraph({ children: [] })] // empty header
    })
  },
  footers: {
    default: new Footer({
      children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({
          font: FONT, size: SIZE_14,
          children: [PageNumber.CURRENT]
        })]
      })]
    })
  },
  children: [/* all main content */]
};
```

---

<a id="assembly"></a>
## 13. Full Assembly Example

```javascript
// After defining all helper functions above:

const doc = new Document({
  styles: {
    default: {
      document: {
        run: { font: FONT, size: SIZE_14, color: "000000" },
        paragraph: {
          spacing: { line: SPACING_15 },
          alignment: AlignmentType.JUSTIFIED,
        }
      }
    },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1",
        basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: SIZE_14, bold: true, font: FONT },
        paragraph: { spacing: { before: 240, after: 240, line: SPACING_15 }, alignment: AlignmentType.CENTER, outlineLevel: 0 }
      },
      {
        id: "Heading2", name: "Heading 2",
        basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: SIZE_14, bold: true, font: FONT },
        paragraph: { spacing: { before: 240, after: 120, line: SPACING_15 }, indent: { firstLine: INDENT }, outlineLevel: 1 }
      },
      {
        id: "Heading3", name: "Heading 3",
        basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: SIZE_14, bold: true, italics: true, font: FONT },
        paragraph: { spacing: { before: 120, after: 120, line: SPACING_15 }, indent: { firstLine: INDENT }, outlineLevel: 2 }
      }
    ]
  },
  sections: [
    // 1. Title page (no page number)
    createTitlePage({
      faculty: "Факультет физико-математических и естественных наук",
      department: "прикладной информатики и теории вероятностей",
      headName: "К.Е. Самуйлов",
      headTitle: "д.т.н., профессор",
      thesisType: "бакалавра",
      direction: '02.03.02 «Фундаментальная информатика и информационные технологии»',
      topic: "Your thesis topic here",
      studentName: "Фамилия Имя Отчество",
      group: "НФИбд-01-15",
      studentId: "1032152610",
      advisorName: "Королькова А.В., к.ф.-м.н., доцент",
      advisorDegree: "Ф.И.О., степень, звание, должность",
      city: "Москва",
      year: "2025"
    }),

    // 2. Annotation
    createAnnotationPage({
      studentName: "Фамилии Имени Отчества",
      topic: "Your thesis topic here",
      annotationText: "Текст аннотации..."
    }),

    // 3. Main content (TOC + body + bibliography + appendices)
    {
      properties: {
        ...pageProps,
        type: SectionType.NEXT_PAGE,
      },
      footers: {
        default: new Footer({
          children: [new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [new TextRun({ font: FONT, size: SIZE_14, children: [PageNumber.CURRENT] })]
          })]
        })
      },
      children: [
        // Table of contents
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { line: SPACING_15, after: 240 },
          children: [new TextRun({ font: FONT, size: SIZE_14, bold: true, text: "Оглавление" })]
        }),
        new TableOfContents("Оглавление", { hyperlink: true, headingStyleRange: "1-3" }),

        // Abbreviations (new page)
        new Paragraph({ children: [new PageBreak()] }),
        ...createAbbreviationsTable([
          { abbr: "TCP", full: "Transmission Control Protocol" },
          { abbr: "RED", full: "Random Early Detection" },
          // ... more abbreviations
        ]),

        // Introduction
        new Paragraph({ children: [new PageBreak()] }),
        chapterHeading("", "Введение"),
        bodyParagraph("Текст введения..."),

        // Chapter 1
        chapterHeading("1", "Название главы"),
        sectionHeading("1.1", "Название раздела"),
        bodyParagraph("Текст раздела..."),

        // Code listing example
        ...createListing("3.1", "Уравнение для изменения размера окна", [
          "equation",
          "  der(w) = wAdd(w, wmax, T) - w * delay(w, T);",
          "end equation;",
        ]),

        // Figure caption example
        figureCaptionParagraph("2.1", "Вид функции сброса в алгоритме RED"),

        // Equation example
        equationParagraph("w(t) = 1/T(q,t)", "2.2"),

        // Conclusion
        chapterHeading("", "Заключение"),
        bodyParagraph("Текст заключения..."),

        // Bibliography
        ...createBibliography([
          "Автор А.Б. Название статьи // Журнал. — 2019. — с. 1-10.",
          "Author A.B. Article title // Journal. — 2020. — p. 1-20.",
        ]),

        // Appendix
        ...createAppendixCode("1", "Программа для ЭВМ «Название»", "Описание авторов.", [
          "function example()",
          "  return 42;",
          "end function;",
        ]),
      ]
    }
  ]
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("/home/claude/vkr_output.docx", buf);
  console.log("Document created successfully!");
});
```
