import numpy as np
import pytest

from roofphoto.baseline import AGE_BANDS, age_band, errors, fit


def test_age_bands_cover_every_year_once():
    assert [age_band(y) for y in (1850, 1929, 1930, 1969, 1970, 1999, 2000, 2025)] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert len(AGE_BANDS) == 4


def test_fit_recovers_factors_and_falls_back_for_thin_bands():
    fp = np.full(30, 100.0)
    built = np.array([1920] * 20 + [1980] * 8 + [2010] * 2)
    truth = np.array([160.0] * 20 + [120.0] * 8 + [110.0] * 2)
    b = fit(fp, truth, built)
    assert b.k_all == pytest.approx(1.6)  # median ratio
    assert b.k_band == {0: pytest.approx(1.6)}  # the 1980 and 2010 bands have < 10 houses
    np.testing.assert_allclose(b.b2([100, 100], [1920, 1980]), [160, 160])  # 1980 falls back to k_all
    np.testing.assert_allclose(b.b1([50]), [80])


def test_errors_are_relative_and_signed_for_bias():
    out = errors([110, 90, 100, 150], [100, 100, 100, 100])
    assert out["median_abs_pct"] == pytest.approx(10.0)
    assert out["bias_median_pct"] == pytest.approx(5.0)
    assert out["within_10pct"] == pytest.approx(0.75)
