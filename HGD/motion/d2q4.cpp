// move_voids.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <cstdio>
#include <vector>

namespace py = pybind11;

double nanmean(std::vector<double> data)
{
    int count = 0;
    double sum = 0;

    for (int i = 0; i < data.size(); i++)
    {
        if (!std::isnan(data[i]))
        {
            sum += data[i];
            count++;
        }
    }
    double mean = (count > 0) ? sum / static_cast<double>(count) : 0.0;
    return mean;
}

std::vector<double> compute_solid_fraction(py::array_t<double> s)
{
    auto s_buf = s.unchecked<3>(); // Read-only access
    int nx = s.shape(0);
    int ny = s.shape(1);
    int nm = s.shape(2);

    std::vector<double> nu(nx * ny, 0.0); // Flattened 2D array

    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            int count = 0; // Total number of entries in third dimension
            int valid = 0; // Non-NaN count

            for (int k = 0; k < nm; k++)
            {
                if (!std::isnan(s_buf(i, j, k)))
                {
                    valid++; // Count valid (non-NaN) entries
                }
                count++;
            }

            nu[i * ny + j] = (count > 0) ? static_cast<double>(valid) / count : 0.0;
        }
    }

    return nu; // Returns a C++ std::vector<double>
}

std::vector<double> compute_s_inv_bar(py::array_t<double> s)
{
    auto s_buf = s.unchecked<3>(); // Read-only access
    int nx = s.shape(0);
    int ny = s.shape(1);
    int nm = s.shape(2);

    std::vector<double> s_inv_bar(nx * ny, 0.0); // Flattened 2D array

    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            double sum = 0.0;
            int count = 0;
            for (int k = 0; k < nm; k++)
            {
                if (!std::isnan(s_buf(i, j, k)) && s_buf(i, j, k) > 0)
                {
                    sum += 1.0 / s_buf(i, j, k);
                    count++;
                }
            }
            s_inv_bar[i * s.shape(1) + j] = (count > 0) ? count / sum : 0.0;
        }
    }

    return s_inv_bar;
}

std::vector<double> compute_s_bar(py::array_t<double> s)
{
    auto s_buf = s.unchecked<3>(); // Read-only access
    int nx = s.shape(0);
    int ny = s.shape(1);
    int nm = s.shape(2);

    std::vector<double> s_bar(nx * ny, 0.0); // Flattened 2D array

    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            double sum = 0.0;
            int count = 0;
            for (int k = 0; k < nm; k++)
            {
                if (!std::isnan(s_buf(i, j, k)) && s_buf(i, j, k) > 0)
                {
                    sum += s_buf(i, j, k);
                    count++;
                }
            }
            s_bar[i * s.shape(1) + j] = (count > 0) ? sum / count : 0.0;
        }
    }

    return s_bar;
}

std::vector<bool> compute_some_particles(std::vector<double> nu, int nx, int ny)
{

    std::vector<bool> some_particles(nx * ny, true); // Flattened 2D array

    for (int i = 1; i < nx - 1; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            some_particles[i * ny + j] = (nu[i * ny + j] + nu[(i - 1) * ny + j] + nu[(i + 1) * ny + j] + nu[i * ny + j + 1]) > 0.0;
        }
    }

    return some_particles;
}

std::vector<bool> compute_locally_fluid(std::vector<double> nu, int nx, int ny, double nu_cs)
{

    std::vector<bool> locally_fluid(nx * ny, true); // Flattened 2D array

    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            locally_fluid[i * ny + j] = nu[i * ny + j] < nu_cs;
        }
    }

    return locally_fluid;
}

py::tuple move_voids(
    py::array_t<double> u,
    py::array_t<double> v,
    py::array_t<double> s,
    py::object p,
    int dummy,
    py::object c,
    py::object T,
    py::object chi,
    py::object last_swap)
{
    double g = p.attr("g").cast<double>();
    double dt = p.attr("dt").cast<double>();
    double dy = p.attr("dy").cast<double>();
    double alpha = p.attr("alpha").cast<double>();
    double nu_cs = p.attr("nu_cs").cast<double>();
    bool cyclic_BC = p.attr("cyclic_BC").cast<bool>();

    auto u_buf = u.mutable_unchecked<3>();
    auto v_buf = v.mutable_unchecked<3>();
    auto s_buf = s.mutable_unchecked<3>();

    int nx = p.attr("nx").cast<int>();
    int ny = p.attr("ny").cast<int>();
    int nm = p.attr("nm").cast<int>();

    // py::object operators = py::module_::import("HGD.operators");

    // py::array_t<double> N_swap_arr({s.shape(0), s.shape(1)});
    // N_swap_arr.attr("fill")(1.0);
    std::vector<double> N_swap_arr(s.shape(0) * s.shape(1), 1.0);

    auto np = py::module_::import("numpy");
    auto nu_py = np.attr("mean")(np.attr("isnan")(s), 2);
    auto nu_2 = np.attr("subtract")(1.0, nu_py).cast<py::array_t<double>>();
    auto nu_buf = nu_2.mutable_unchecked<2>();

    std::vector<double> nu = compute_solid_fraction(s);
    std::vector<double> s_bar = compute_s_bar(s);
    std::vector<double> s_inv_bar = compute_s_inv_bar(s);
    double s_bar_mean = nanmean(s_bar);
    std::vector<bool> some_particles = compute_some_particles(nu, nx, ny);
    // std::vector<bool> locally_fluid = compute_locally_fluid(nu, nx, ny, nu_cs);
    // auto skip = operators.attr("empty_nearby")(nu_2, p);
    // auto s_bar = operators.attr("get_average")(s);
    // auto s_inv_bar = operators.attr("get_hyperbolic_average")(s);

    double v_y;
    std::string advection_model = p.attr("advection_model").cast<std::string>();

    if (advection_model == "average_size")
    {
        v_y = std::sqrt(g * s_bar_mean);
    }
    else if (advection_model == "freefall")
    {
        v_y = std::sqrt(2 * g * dy);
    }
    else
    {
        throw std::runtime_error("Stress model not implemented");
    }

    double P_u_ref = v_y * dt / dy;
    double P_lr_ref = alpha * v_y * s_bar_mean * dt / dy / dy;
    auto delta_nu = p.attr("delta_limit").cast<double>();

    auto indices = p.attr("indices").cast<std::vector<int>>();

    std::array<int, 3> dest = {0, 0, 0};

    for (auto index : indices)
    {
        int i = index / ((ny - 1) * nm);
        int j = (index / nm) % (ny - 1);
        int k = index % nm;

        if (some_particles[i * ny + j])
        {
            if (std::isnan(s_buf(i, j, k)))
            {
                // if (locally_fluid[i * ny + j])
                if ( nu[i * ny + j] < nu_cs )
                {
                    double P_u = (!std::isnan(s_buf(i, j + 1, k))) ? P_u_ref * (s_inv_bar[i * ny + j + 1] / s_buf(i, j + 1, k)) : 0;

                    int l = (i == 0) ? (cyclic_BC ? nx - 1 : 0) : i - 1;
                    int r = (i == nx - 1) ? (cyclic_BC ? 0 : nx - 1) : i + 1;

                    double nu_here = nu[i * ny + j];
                    double nu_left = nu[l * ny + j];
                    double nu_right = nu[r * ny + j];
                    bool stable_left = nu_left - nu_here <= delta_nu;
                    bool stable_right = nu_right - nu_here <= delta_nu;

                    double P_l = (!std::isnan(s_buf(l, j, k)) && !stable_left) ? P_lr_ref * (s_buf(l, j, k) / s_bar[l * ny + j]) : 0;

                    double P_r = (!std::isnan(s_buf(r, j, k)) && !stable_right) ? P_lr_ref * (s_buf(r, j, k) / s_bar[r * ny + j]) : 0;

                    double P_tot = P_u + P_l + P_r;
                    
                    if (P_tot > 1) {
                        throw std::runtime_error("P_tot > 1");
                    }

                    if (P_tot > 0)
                    {
                        double rand_val = static_cast<double>(std::rand()) / RAND_MAX;

                        bool found = false;

                        if (rand_val < P_u && P_u > 0)
                        {
                            dest = {i, j + 1, k};
                            found = true;
                            v_buf(i, j, k) += 1;
                        }
                        else if (rand_val < (P_l + P_u))
                        {
                            dest = {l, j, k};
                            found = true;
                            u_buf(i, j, k) += 1;
                            v_buf(i, j, k) += 1;
                        }
                        else if (rand_val < P_tot)
                        {
                            dest = {r, j, k};
                            found = true;
                            u_buf(i, j, k) -= 1;
                            v_buf(i, j, k) += 1;
                        }

                        if (found)
                        {
                            float tmp = s_buf(i, j, k);
                            s_buf(i, j, k) = s_buf(dest[0], dest[1], dest[2]);
                            s_buf(dest[0], dest[1], dest[2]) = tmp;

                            nu[i * ny + j] += 1.0 / nm;
                            nu[dest[0] * ny + dest[1]] -= 1.0 / nm;

                            N_swap_arr[i * ny + j] += 1;
                            N_swap_arr[dest[0] * ny + dest[1]] += 1;
                        }
                    }
                }
            }
        }
    }

    for (size_t i = 0; i < N_swap_arr.size(); i++)
    {
        N_swap_arr[i] /= nm;
    }
    py::array_t<double> chi_out({s.shape(0), s.shape(1)}, N_swap_arr.data());

    return py::make_tuple(u, v, s, c, T, chi_out, last_swap);
}

// PYBIND11_MODULE(d2q4_cpp, m) {
//     m.def("move_voids", &move_voids);
// }

PYBIND11_MODULE(d2q4_cpp, m)
{
    m.def("move_voids", &move_voids, "Moves voids in the system",
          py::arg("u"), py::arg("v"), py::arg("s"), py::arg("p"), py::arg("dummy"),
          py::arg("c") = py::none(), py::arg("T") = py::none(),
          py::arg("chi") = py::none(), py::arg("last_swap") = py::none());
}