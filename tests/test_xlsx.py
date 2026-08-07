"""The workbook writer emits a package Excel will accept.

tests/xlsx.py builds an .xlsx by hand rather than through a library, which buys
the report tools a spreadsheet with no dependency and costs them a format that
can go subtly wrong without anything raising: a stray control character, a
column past Z, or a NaN written where a number is expected all produce a file
that zips and parses and that Excel then refuses to open. Nothing downstream
would notice, because the run has already succeeded by the time the workbook is
written.

So these check the parts of the spec that a wrong result would still look valid
without: that every part parses, that values land in the right kind of cell,
and that the one row treated differently - the header - actually is.
"""

import xml.etree.ElementTree as ET
import zipfile

import pytest

from tests import xlsx

MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def read_sheet(path, index=1):
    """Return one sheet as a list of rows of (value, is_number, style)."""
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read(f"xl/worksheets/sheet{index}.xml"))

    rows = []
    for row in root.find(MAIN + "sheetData"):
        cells = []
        for cell in row:
            number = cell.find(MAIN + "v")
            inline = cell.find(MAIN + "is")
            if number is not None:
                cells.append((number.text, True, cell.get("s")))
            elif inline is not None:
                cells.append((inline.find(MAIN + "t").text, False, cell.get("s")))
            else:
                cells.append((None, False, cell.get("s")))
        rows.append(cells)
    return rows


@pytest.fixture
def workbook(tmp_path):
    def build(sheets):
        path = tmp_path / "book.xlsx"
        return xlsx.write(str(path), sheets)
    return build


def test_every_part_is_well_formed_xml(workbook):
    """A part that does not parse is a file Excel opens with an error dialog."""
    path = workbook([("Results", [["Variant", "Score"], ["k51", 0.5]])])

    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        for name in names:
            ET.fromstring(archive.read(name))     # raises if malformed

    # The five parts a reader walks from the root relationship down.
    for required in ("[Content_Types].xml", "_rels/.rels", "xl/workbook.xml",
                     "xl/_rels/workbook.xml.rels", "xl/worksheets/sheet1.xml"):
        assert required in names


def test_numbers_are_numeric_cells_and_text_is_not(workbook):
    """A number written as text sorts alphabetically, which is the whole point lost."""
    path = workbook([("Results", [["Variant", "Score", "Note"],
                                  ["k51_sel100", 0.6875, "ok"],
                                  ["k15_sel50", 12, ""]])])
    rows = read_sheet(path)

    assert [cell[1] for cell in rows[1]] == [False, True, False]
    assert rows[1][1][0] == "0.6875"
    assert [cell[1] for cell in rows[2][:2]] == [False, True]
    assert rows[2][1][0] == "12"


def test_header_row_is_bold_and_data_rows_are_not(workbook):
    path = workbook([("Results", [["Variant", "Score"], ["k51", 0.5]])])
    rows = read_sheet(path)

    assert all(cell[2] == "1" for cell in rows[0])
    assert all(cell[2] in (None, "0") for cell in rows[1])


def test_non_finite_values_are_written_as_text(workbook):
    """PSNR comes back as inf on an exact match, and a worksheet cannot hold one."""
    path = workbook([("Results", [["Metric"],
                                  [float("inf")],
                                  [float("nan")],
                                  [float("-inf")]])])
    rows = read_sheet(path)

    assert [cell[1] for cell in (rows[1][0], rows[2][0], rows[3][0])] == [False] * 3
    assert rows[1][0][0] == "inf"


def test_columns_continue_past_z(workbook):
    """A sweep over a method with many dials runs off the single letters."""
    assert xlsx.column_letter(1) == "A"
    assert xlsx.column_letter(26) == "Z"
    assert xlsx.column_letter(27) == "AA"
    assert xlsx.column_letter(52) == "AZ"
    assert xlsx.column_letter(53) == "BA"

    path = workbook([("Wide", [[f"c{i}" for i in range(30)],
                               list(range(30))])])
    with zipfile.ZipFile(path) as archive:
        sheet = archive.read("xl/worksheets/sheet1.xml").decode()
    assert 'r="AD1"' in sheet and 'ref="A1:AD2"' in sheet


def test_control_characters_do_not_corrupt_the_package(workbook):
    """An exception message reaches the Status column verbatim."""
    path = workbook([("Results", [["Status"], ["broke\x00 at \x07tile <3> & up"]])])
    rows = read_sheet(path)

    assert rows[1][0][0] == "broke at tile <3> & up"


def test_ragged_rows_keep_their_place(workbook):
    """A failed candidate has no metric cells, and must still be a row.

    Short rows are written short rather than padded - a cell carries its own
    reference, so the trailing ones are simply absent - but the sheet dimension
    still has to span the widest row or a reader can clip the table.
    """
    path = workbook([("Results", [["Variant", "Score", "Status"],
                                  ["k51"],
                                  ["k15", 0.5, "ok"]])])
    rows = read_sheet(path)

    assert len(rows) == 3
    assert rows[1][0][0] == "k51"
    with zipfile.ZipFile(path) as archive:
        assert 'ref="A1:C3"' in archive.read("xl/worksheets/sheet1.xml").decode()


def test_header_is_frozen_and_filterable(workbook):
    """What makes the table usable rather than merely present."""
    path = workbook([("Results", [["Variant", "Score"], ["k51", 0.5]])])
    with zipfile.ZipFile(path) as archive:
        sheet = archive.read("xl/worksheets/sheet1.xml").decode()

    assert 'state="frozen"' in sheet and 'ySplit="1"' in sheet
    assert '<autoFilter ref="A1:B2"/>' in sheet


def test_sheet_names_are_sanitised_and_truncated(workbook):
    """Excel rejects a workbook naming a sheet in a way it forbids."""
    path = workbook([("Depth Map (Average) - kernel x selectivity x halo", [["a"]]),
                     ("bad/name:here", [["b"]])])
    with zipfile.ZipFile(path) as archive:
        workbook_xml = archive.read("xl/workbook.xml").decode()

    root = ET.fromstring(workbook_xml)
    names = [sheet.get("name") for sheet in root.find(MAIN + "sheets")]
    assert all(len(name) <= 31 for name in names)
    assert not set("".join(names)) & set(r"[]:*?/\'")


def test_a_workbook_needs_a_sheet(workbook):
    with pytest.raises(ValueError):
        workbook([])
