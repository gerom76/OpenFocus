"""Write a spreadsheet Excel opens, without a spreadsheet library.

An .xlsx is a zip of XML parts, and the subset needed for "a table of numbers
with a header row" is small enough to emit directly. That is worth doing here
because the alternative is a dependency: nothing else in this project reads or
writes spreadsheets, and a report tool that produces its main artefact only when
an optional package happens to be installed produces it unreliably.

CSV was the other option and is not the same thing: a sweep's table is read by
sorting and filtering it, which is what the frozen header, the autofilter and
the per-column widths set up here are for.

    from tests import xlsx
    xlsx.write("results.xlsx", [
        ("Results", [["Variant", "Score"], ["k51_sel100", 0.647]]),
    ])

The first row of each sheet is the header: bold, frozen, and the range the
autofilter is applied to. Values are written as numbers when they are numbers
and as text otherwise; a NaN or an infinity is written as text, since a
spreadsheet has no numeric spelling for either.
"""

import zipfile
from typing import Any, Sequence, Tuple

# The two content types every part below is one of, plus the three relationship
# types the package needs to be navigable.
_CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/>'
    '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
    '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
    "{sheets}"
    "</Types>"
)

_ROOT_RELS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" '
    'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
    'Target="xl/workbook.xml"/>'
    "</Relationships>"
)

# Two cell formats: 0 is plain, 1 is bold, which is what the header row uses.
# The fills are not decorative - Excel rejects a stylesheet whose first two
# fills are not `none` and `gray125`.
_STYLES = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
    '<fonts count="2">'
    '<font><sz val="11"/><name val="Calibri"/></font>'
    '<font><b/><sz val="11"/><name val="Calibri"/></font>'
    "</fonts>"
    '<fills count="2">'
    '<fill><patternFill patternType="none"/></fill>'
    '<fill><patternFill patternType="gray125"/></fill>'
    "</fills>"
    '<borders count="1"><border/></borders>'
    '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
    '<cellXfs count="2">'
    '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
    '<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/>'
    "</cellXfs>"
    "</styleSheet>"
)

# Widths are in character units and clamped: a column holding one long note
# should not push every other column off the screen.
_MIN_WIDTH = 8
_MAX_WIDTH = 48


def column_letter(index: int) -> str:
    """1 -> A, 26 -> Z, 27 -> AA."""
    letters = ""
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        letters = chr(ord("A") + remainder) + letters
    return letters


def _escape(text: str) -> str:
    """XML text, with the characters XML 1.0 cannot carry removed.

    A metric formatted into a string never contains a control character, but a
    caption or an exception message can, and one of those in the stream is the
    difference between a spreadsheet and a file Excel refuses to open.
    """
    text = "".join(ch for ch in text
                   if ch >= " " or ch in "\t\n\r")
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _is_number(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if not isinstance(value, (int, float)):
        return False
    # NaN and the infinities have no numeric spelling in a worksheet; they go
    # through as text so the cell says what happened rather than being empty.
    return value == value and value not in (float("inf"), float("-inf"))


def _cell(reference: str, value: Any, style: int) -> str:
    attributes = f'r="{reference}"' + (f' s="{style}"' if style else "")
    if value is None or value == "":
        return f"<c {attributes}/>"
    if _is_number(value):
        return f"<c {attributes}><v>{value!r}</v></c>"
    return (f'<c {attributes} t="inlineStr"><is><t xml:space="preserve">'
            f"{_escape(str(value))}</t></is></c>")


def _sheet(rows: Sequence[Sequence[Any]]) -> str:
    """One worksheet part: a frozen bold header, the data, and an autofilter."""
    rows = [list(row) for row in rows] or [[]]
    columns = max(len(row) for row in rows)
    span = f"A1:{column_letter(max(columns, 1))}{max(len(rows), 1)}"

    widths = []
    for index in range(columns):
        longest = max((len(str(row[index])) for row in rows if index < len(row)
                       and row[index] is not None), default=0)
        widths.append(min(_MAX_WIDTH, max(_MIN_WIDTH, longest + 2)))
    columns_xml = "".join(
        f'<col min="{i + 1}" max="{i + 1}" width="{width}" customWidth="1"/>'
        for i, width in enumerate(widths))

    body = []
    for row_index, row in enumerate(rows, start=1):
        style = 1 if row_index == 1 else 0
        cells = "".join(
            _cell(f"{column_letter(column_index)}{row_index}", value, style)
            for column_index, value in enumerate(row, start=1))
        body.append(f'<row r="{row_index}">{cells}</row>')

    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<dimension ref="{span}"/>'
        '<sheetViews><sheetView workbookViewId="0">'
        '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
        "</sheetView></sheetViews>"
        '<sheetFormatPr defaultRowHeight="15"/>'
        f"<cols>{columns_xml}</cols>"
        f'<sheetData>{"".join(body)}</sheetData>'
        f'<autoFilter ref="{span}"/>'
        "</worksheet>"
    )


def write(path: str, sheets: Sequence[Tuple[str, Sequence[Sequence[Any]]]]) -> str:
    """Write `sheets` as an .xlsx at `path`, and return the path.

    Each sheet is (name, rows); the first row is its header. Sheet names are
    truncated to the 31 characters Excel allows and stripped of the characters
    it forbids in one.
    """
    if not sheets:
        raise ValueError("a workbook needs at least one sheet")

    names = []
    for index, (name, _) in enumerate(sheets, start=1):
        cleaned = "".join(ch for ch in str(name) if ch not in r"[]:*?/\'")[:31]
        names.append(cleaned or f"Sheet{index}")

    overrides = "".join(
        f'<Override PartName="/xl/worksheets/sheet{i}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for i in range(1, len(sheets) + 1))

    sheet_tags = "".join(
        f'<sheet name="{_escape(name)}" sheetId="{i}" r:id="rId{i}"/>'
        for i, name in enumerate(names, start=1))
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheets>{sheet_tags}</sheets></workbook>"
    )

    # The styles part takes the id after the last sheet, so the two never clash
    # however many sheets there are.
    relationships = "".join(
        f'<Relationship Id="rId{i}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        f'Target="worksheets/sheet{i}.xml"/>'
        for i in range(1, len(sheets) + 1))
    relationships += (
        f'<Relationship Id="rId{len(sheets) + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>')
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f"{relationships}</Relationships>"
    )

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml",
                         _CONTENT_TYPES.format(sheets=overrides))
        archive.writestr("_rels/.rels", _ROOT_RELS)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        archive.writestr("xl/styles.xml", _STYLES)
        for index, (_, rows) in enumerate(sheets, start=1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", _sheet(rows))
    return path
