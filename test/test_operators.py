import numpy as np
from HGD import operators, params


def test_simple_swap():
    src = [0, 0, 0]
    dst = [1, 1, 0]
    s = np.array([[[1], [2]], [[3], [4]]])
    arrays = [s, None]
    nu = np.zeros((2, 2))
    p = params.dict_to_class({"nm": 2})

    arrays, nu = operators.swap(src, dst, arrays, nu, p)

    assert arrays[0][0, 0, 0] == 4
    assert arrays[1] is None
    assert np.allclose(nu, np.array([[0.5, 0.0], [0.0, -0.5]]))


def test_simple_solid_fraction():
    s = np.array([[[2, np.nan], [3, 1]], [[1, 1], [4, np.nan]]])
    nu = operators.get_solid_fraction(s, [1, 1])
    assert nu == 0.5


def test_simple_get_average():
    s = np.array([[[2, np.nan, 1], [3, 3, 3]], [[1, 1, np.nan], [4, np.nan, np.nan]]])
    s_mean = operators.get_average(s)
    assert s_mean[0, 0] == 1.5
    assert operators.get_average(s, [0, 0]) == 1.5


def test_get_hyperbolic_average():
    s = np.array([[[2, np.nan], [3, 1]], [[1, 1], [4, np.nan]]])
    s_inv_bar = operators.get_hyperbolic_average(s)
    assert s_inv_bar[0][0] == 2
    assert s_inv_bar[1][1] == 4
