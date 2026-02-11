import numpy as np
import pytest

import HGD.operators

try:
    from HGD.motion import d2q4_cpp
except ImportError:  # pragma: no cover
    d2q4_cpp = None


pytestmark = pytest.mark.skipif(d2q4_cpp is None, reason="d2q4_cpp extension is not available")


def _make_test_array(nx=10, ny=10, nm=5):
    rng = np.random.default_rng(1234)
    s = rng.random((nx, ny, nm))
    s[s < 0.2] = np.nan
    return s


def test_compute_solid_fraction_matches_numpy():
    s = _make_test_array()
    nu_cpp = np.array(d2q4_cpp.compute_solid_fraction(s)).reshape(s.shape[0], s.shape[1])
    nu_np = HGD.operators.get_solid_fraction(s)
    assert np.allclose(nu_cpp, nu_np, atol=1e-6, equal_nan=True)


def test_compute_s_inv_bar_matches_numpy():
    s = _make_test_array()
    s_inv_bar_cpp = np.array(d2q4_cpp.compute_s_inv_bar(s)).reshape(s.shape[0], s.shape[1])
    s_inv_bar_np = HGD.operators.get_hyperbolic_average(s)
    assert np.allclose(s_inv_bar_cpp, s_inv_bar_np, atol=1e-6, equal_nan=True)


def test_compute_mean_matches_numpy():
    s = _make_test_array()
    s_bar_cpp = np.array(d2q4_cpp.compute_mean(s)).reshape(s.shape[0], s.shape[1])
    s_bar_np = HGD.operators.get_average(s)
    assert np.allclose(s_bar_cpp, s_bar_np, atol=1e-6, equal_nan=True)
