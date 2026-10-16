import csv
import json

from shapely.geometry import LineString

from floorplan_parser.model import Element, Extraction, elements_from_lines


def _extraction():
    return Extraction(
        drawing_id="sample_a",
        page=1,
        extractor="vector",
        elements=[
            Element("ln_1", "line", [(10.0, 20.0), (110.0, 20.0)], "vector", (10.0, 20.0, 110.0, 20.0)),
            Element("ln_2", "polyline", [(0.0, 0.0), (3.0, 4.0), (3.0, 10.0)], "vector", (0.0, 0.0, 3.0, 10.0)),
        ],
        page_size=(800.0, 800.0),
    )


def test_json_roundtrip(tmp_path):
    ex = _extraction()
    path = tmp_path / "elements.json"
    ex.to_json(path)
    data = json.loads(path.read_text())
    assert data["coordinate_system"] == "pdf_points"
    assert data["elements"][0]["geometry"] == [[10.0, 20.0], [110.0, 20.0]]
    assert Extraction.from_json(path) == ex


def test_csv_has_one_row_per_element_with_length(tmp_path):
    ex = Extraction("d", 1, "raster", elements_from_lines([LineString([(0, 0), (3, 4)])], "raster"))
    path = tmp_path / "elements.csv"
    ex.to_csv(path)
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 1
    assert rows[0]["id"] == "ln_1"
    assert float(rows[0]["length"]) == 5.0
    assert rows[0]["kind"] == "line"
