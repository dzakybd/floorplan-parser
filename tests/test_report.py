from eval.report import distribution


def test_distribution_keeps_missing_matches_out_of_denominator():
    assert distribution([None, 1, 3]) == {"n": 2, "mean": 2, "median": 2, "min": 1, "max": 3}
    assert distribution([None])["mean"] is None
