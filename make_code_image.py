"""Render lines of game_of_life.py as an editor-style image for the Code Example slide.

The text comes straight from the file, so the image always shows the real code
with its real line numbers. Long lines wrap the way an editor's word wrap does.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pygments import lex
from pygments.lexers import PythonLexer
from pygments.token import Comment, Keyword, Name, Number, Operator, String

HERE = Path(__file__).parent
SOURCE = HERE / "game_of_life.py"
OUT_PATH = HERE / "slide_images" / "code" / "code_screenshot.png"

FIRST_LINE, LAST_LINE = 68, 89      # countNeighbors, stepForward, conwayStep
WIDTH, HEIGHT = 1672, 2000          # same shape as the slide's image box
FONT_PATH = "/System/Library/Fonts/Menlo.ttc"
LINE_SPACING = 1.42
TITLE_BAR = 96
PADDING = 36
GUTTER_COLUMNS = 4
WRAP_INDENT = 4

BACKGROUND = "#1e1e1e"
TITLE_BACKGROUND = "#2d2d2d"
TAB_TEXT = "#ffffff"
LINE_NUMBER = "#858585"
PLAIN = "#d4d4d4"
VARIABLE = "#9cdcfe"
COMMENT = "#6a9955"
STRING = "#ce9178"
KEYWORD = "#569cd6"
FUNCTION = "#dcdcaa"
CLASS = "#4ec9b0"
NUMBER = "#b5cea8"
WINDOW_DOTS = ("#ff5f57", "#febc2e", "#28c840")


def tokenColor(token, text):
    if token in Comment:
        return COMMENT
    if token in String:
        return STRING
    if token in Number:
        return NUMBER
    if token in Keyword or token in Operator.Word:
        return KEYWORD
    if token in Name.Function or token in Name.Decorator or token in Name.Builtin:
        return FUNCTION
    if token in Name.Class or text == "LifeGrid":
        return CLASS
    if token in Name:
        return VARIABLE
    return PLAIN


def coloredLines():
    """Every line of the source file as a list of (character, color) pairs."""
    lines = [[]]
    for token, text in lex(SOURCE.read_text(), PythonLexer()):
        color = tokenColor(token, text)
        for ch in text:
            if ch == "\n":
                lines.append([])
            else:
                lines[-1].append((ch, color))
    return lines


def wrapLine(line, columns):
    """Split one line into rows of at most `columns`, breaking at spaces."""
    indent = len(line) - len("".join(ch for ch, _ in line).lstrip())
    rows = []
    while len(line) > columns:
        breakAt = max(i for i in range(columns + 1) if i == 0 or line[i - 1][0] == " ")
        if breakAt <= indent + WRAP_INDENT:
            breakAt = columns
        rows.append(line[:breakAt])
        line = [(" ", PLAIN)] * (indent + WRAP_INDENT) + line[breakAt:]
    rows.append(line)
    return rows


def layout(lines):
    """Find the largest font where the wrapped code still fits in the image."""
    for size in range(60, 15, -1):
        font = ImageFont.truetype(FONT_PATH, size, index=0)
        charWidth = font.getlength("M")
        lineHeight = round(size * LINE_SPACING)
        columns = int((WIDTH - 2 * PADDING) / charWidth) - GUTTER_COLUMNS
        wrapped = [wrapLine(line, columns) for line in lines]
        rowCount = sum(len(rows) for rows in wrapped)
        if TITLE_BAR + 2 * PADDING + rowCount * lineHeight <= HEIGHT:
            return font, charWidth, lineHeight, wrapped
    raise ValueError("code does not fit")


if __name__ == "__main__":
    allLines = coloredLines()
    shown = allLines[FIRST_LINE - 1:LAST_LINE]
    assert "countNeighbors" in "".join(ch for ch, _ in allLines[FIRST_LINE])
    assert "stepForward" in "".join(ch for ch, _ in allLines[LAST_LINE - 1])
    font, charWidth, lineHeight, wrapped = layout(shown)

    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)

    draw.rectangle([0, 0, WIDTH, TITLE_BAR], fill=TITLE_BACKGROUND)
    for i, color in enumerate(WINDOW_DOTS):
        x = 44 + i * 40
        draw.ellipse([x - 12, TITLE_BAR // 2 - 12, x + 12, TITLE_BAR // 2 + 12], fill=color)
    tabFont = ImageFont.truetype(FONT_PATH, 32, index=0)
    draw.rectangle([170, 14, 170 + 56 + tabFont.getlength(SOURCE.name), TITLE_BAR], fill=BACKGROUND)
    draw.text((198, TITLE_BAR // 2 + 7), SOURCE.name, font=tabFont, fill=TAB_TEXT, anchor="lm")

    y = TITLE_BAR + PADDING
    for offset, rows in enumerate(wrapped):
        number = str(FIRST_LINE + offset).rjust(GUTTER_COLUMNS - 2)
        draw.text((PADDING, y), number, font=font, fill=LINE_NUMBER)
        for row in rows:
            for column, (ch, color) in enumerate(row):
                if ch != " ":
                    draw.text((PADDING + (GUTTER_COLUMNS + column) * charWidth, y), ch, font=font, fill=color)
            y += lineHeight

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUT_PATH)
    print(f"font size {font.size}, rows {sum(len(rows) for rows in wrapped)}, bottom of text {y} of {HEIGHT}")
