import numpy as np
import pytest

from roofphoto.stats import compare, sign_test_one_sided, sign_test_two_sided


def test_sign_test_known_values():
    # 204 of 351: the Stage 4 result. scipy.stats.binomtest(204, 351) gives 0.0014 / 0.0027.
    assert sign_test_one_sided(204, 351) == pytest.approx(0.00137, abs=5e-5)
    assert sign_test_two_sided(204, 351) == pytest.approx(0.00274, abs=5e-5)
    # Symmetric, and 1 at the centre.
    assert sign_test_two_sided(147, 351) == pytest.approx(sign_test_two_sided(204, 351))
    assert sign_test_two_sided(5, 10) == 1.0
    # Small case by hand: 9 of 10 -> one-sided 11/1024, two-sided 22/1024.
    assert sign_test_one_sided(9, 10) == pytest.approx(11 / 1024)
    assert sign_test_two_sided(9, 10) == pytest.approx(22 / 1024)


def test_compare_reports_both_sides():
    rng = np.random.default_rng(1)
    base = rng.uniform(0.05, 0.3, 200)
    model = base * rng.uniform(0.7, 1.1, 200)
    r = compare(model, base, n_boot=200)
    assert r["sign_test_p_two_sided"] >= r["sign_test_p_one_sided"]
