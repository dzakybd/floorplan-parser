import hashlib

from eval.synth import make_synthetic
from floorplan_parser.groundtruth import load_ground_truth

NAMES = [
    "synth-1-walls",
    "synth-2-diagonal",
    "synth-3-doors",
    "synth-4-stair",
    "synth-5-near-parallel",
    "synth-text-paths",
]


def test_makes_six_svgs(tmp_path):
    paths = make_synthetic(tmp_path)
    assert [p.stem for p in paths] == NAMES
    assert (tmp_path / "synth-text-paths.gt.svg").exists()


def test_deterministic(tmp_path):
    a = [hashlib.sha256(p.read_bytes()).hexdigest() for p in make_synthetic(tmp_path / "a")]
    b = [hashlib.sha256(p.read_bytes()).hexdigest() for p in make_synthetic(tmp_path / "b")]
    assert a == b


def test_ground_truth_loads_for_each(tmp_path):
    for p in make_synthetic(tmp_path):
        gt = load_ground_truth(p)
        assert len(gt.lines) >= 10, p.stem
        assert "wall" in gt.labels, p.stem


def test_text_glyphs_only_in_drawing(tmp_path):
    make_synthetic(tmp_path)
    full = load_ground_truth(tmp_path / "synth-text-paths.svg")
    gt = load_ground_truth(tmp_path / "synth-text-paths.gt.svg")
    assert len(full.lines) > len(gt.lines)
