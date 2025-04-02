import numpy as np
import HGD.operators
from HGD.motion import d2q4_cpp  # Import your compiled C++ module

# Create a random test array with NaNs (similar to real data)
nx, ny, nm = 10, 10, 5  # Small test size for speed
s = np.random.rand(nx, ny, nm)
s[s < 0.2] = np.nan  # Introduce NaNs to simulate missing values


# Compute using C++ functions
nu_cpp = np.array(d2q4_cpp.compute_solid_fraction(s)).reshape(nx, ny)
s_inv_bar_cpp = np.array(d2q4_cpp.compute_s_inv_bar(s)).reshape(nx, ny)
s_bar_cpp = np.array(d2q4_cpp.compute_s_bar(s)).reshape(nx, ny)

# Compute reference values using NumPy
nu_np = HGD.operators.get_solid_fraction(s)
s_inv_bar_np = HGD.operators.get_hyperbolic_average(s)
s_bar_np = HGD.operators.get_average(s)


# Check if they match (allowing small floating-point errors)
def compare_results(name, cpp_result, np_result):
    # print(f"Comparing {name}")
    # print(f"NumPy: {np_result}")
    # print(f"C++: {cpp_result}")
    diff = np.abs(cpp_result - np_result)
    max_diff = np.nanmax(diff)
    print(f"{name} - Max Difference: {max_diff}")
    assert np.allclose(cpp_result, np_result, atol=1e-6, equal_nan=True), f"{name} does not match!"


# Compare results
compare_results("Solid Fraction", nu_cpp, nu_np)
compare_results("Hyperbolic size", s_inv_bar_cpp, s_inv_bar_np)
compare_results("Mean size", s_bar_cpp, s_bar_np)

print("All functions match the original versions!")
