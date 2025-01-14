import unittest
from HGD import operators, params
import numpy as np


class TestOperators(unittest.TestCase):
    def test_simple_swap(self):
        """
        This test to check for the swapping of cell positions and update the corresponding solid fraction (nu)
        Args
            src: has the old indices of a cell [i,j,k]
            dst: has the new indices of a cell [i,j,k]
            s: 3D array containing the local grainsizes at each cell located at [i,j,k]
            arrays: can have different arrays including s,c and T;
            c: tracks the motion of differently labelled cells; it can be an array or None
            T: temperature field; it can be array or None
            nu: solid fraction defined at [i,j] location

        returns
            the updated swapped or unswapped arrays
            the updated solid fraction nu after swapping

        """
        src = [0, 0, 0]
        dst = [1, 1, 0]
        s = np.array([[[1], [2]], [[3], [4]]])
        arrays = [s, None]
        nu = np.zeros((2, 2))
        p = params.dict_to_class({"nm": 2})

        arrays, nu = operators.swap(src, dst, arrays, nu, p)

        self.assertEqual(arrays[0][0, 0, 0], 4)
        self.assertEqual(arrays[1], None)
        self.assertTrue(np.allclose(nu, np.array([[0.5, 0.0], [0.0, -0.5]])))

    def test_simple_solid_fraction(self):
        """
        This test is to check the calculation of solid fraction
        Args
            s: 3D array containing the local grainsizes at each cell located at [i,j,k]
            i: row index of a cell
            j: column index of a cell

        returns
            the solid fraction at i,j location
            = 1 - (no. of NaN cells along heterarchical coordinate / total no. of heterarchical cells)

        """

        s = np.array([[[2, np.nan], [3, 1]], [[1, 1], [4, np.nan]]])

        nu = operators.get_solid_fraction(s, [1, 1])

        self.assertEqual(nu, 0.5)

    def test_simple_get_average(self):
        """
        This test to check the average grainsize along each row
        along heterarchical coordinate - sum of grainsize at every non_NaN cell/no. of non-NaN cells
        and then average over the columns

        Args
            s: 3D array containing the local grainsizes at each cell located at [i,j,k]

        returns
            the average grainsize

        """
        s = np.array([[[2, np.nan, 1], [3, 3, 3]], [[1, 1, np.nan], [4, np.nan, np.nan]]])

        s_mean = operators.get_average(s)

        self.assertEqual(s_mean[0, 0], 1.5)
        self.assertEqual(operators.get_average(s, [0, 0]), 1.5)

    def test_get_hyperbolic_average(self):
        """
        This is to check the hyperbolic mean of s along the heterarchical coordinate
        along heterarchical coordinate - 1 / (sum of (1 / grainsize) at every non_NaN cell/no. of non-NaN cells)
        Args
            s: 3D array containing the local grainsizes at each cell located at [i,j,k]

        returns
            s_inv_bar: a 2D array with hyperbolic mean at every [i,j]

        """
        s = np.array([[[2, np.nan], [3, 1]], [[1, 1], [4, np.nan]]])

        s_inv_bar = operators.get_hyperbolic_average(s)
        self.assertEqual(s_inv_bar[0][0], 2)
        self.assertEqual(s_inv_bar[1][1], 4)


if __name__ == "__main__":
    unittest.main()
