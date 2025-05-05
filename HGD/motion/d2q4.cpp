#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <omp.h>
#include <random>
#include <cmath>
#include <vector>
#include "helpers.h"

namespace py = pybind11;
using Eigen::ArrayXXd;

py::array_t<double> stream(
    std::vector<double> u_mean,
    std::vector<double> v_mean,
    py::array_t<double> s,
    py::array_t<double> mask,
    std::vector<double> nu,
    int nx, int ny, int nm, double nu_cs,
    int cyclic_BC_y_offset, bool cyclic_BC,
    double dx, double dy, double dt)
{

    // Access mutable NumPy arrays
    // auto u_buf = u.mutable_unchecked<2>();
    // auto v_buf = v.mutable_unchecked<2>();
    auto s_buf = s.mutable_unchecked<3>();
    auto mask_buf = mask.unchecked<2>();

    // storage arrays
    std::array<int, 3> dest = {0, 0, 0};
    std::vector<double> u_new(nx * ny * nm, 0.0);
    std::vector<double> v_new(nx * ny * nm, 0.0);

    double inverse_nm = 1.0 / nm;

    int l, r, j_l, j_r;

    {
        // Thread-safe PRNG (One per thread)
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> rand_dist(0.0, 1.0);

        // #pragma omp for
        for (int i=0; i < nx; i++) {
            for (int j=0; j < ny; j++) {
                std::tie(l, r, j_l, j_r) = get_lr(i, j, nx, ny, cyclic_BC_y_offset, cyclic_BC);

                double P_u = mask_buf(i, j + 1) ? v_mean[i*ny + j+1]*dy/dt : 0;
                double u_left = (u_mean[i * ny + j_l] < 0) ? u_mean[i * ny + j_l] : 0;
                double P_l = mask_buf(l, j_l) ? u_left * dy / dt : 0;
                double u_right = (u_mean[i * ny + j_r] > 0) ? u_mean[i * ny + j_r] : 0;
                double P_r = mask_buf(r, j_r) ? u_right * dy / dt : 0;

                int N_u = static_cast<int>(nu[i * ny + j] * P_u);
                int N_l = static_cast<int>(nu[i * ny + j] * P_l);
                int N_r = static_cast<int>(nu[i * ny + j] * P_r);    
                
                std::vector<int> N_arr = {N_u, N_l, N_r};
                std::vector<int> dests = {i, j + 1, l, j_l, r, j_r};
                int k = 0;
                for (int d = 0; d < 3; d += 1) {
                    dest = {dests[2*d], dests[2*d + 1]};
                        // get indices in 3rd dimension of s that can be swapped
                        // Eigen::VectorXi non_nan_indices(nm);
                        // int count = 0;
                        // for (int kk = 0; kk < nm; kk++) {
                        //     if (!std::isnan(s_buf(i, j, kk))) {
                        //         count++;
                        //     }
                        // }
                        // non_nan_indices.conservativeResize(count);
                        // shuffle the indices
                        // std::shuffle(non_nan_indices.data(), non_nan_indices.data() + count, gen);
                    int N_added = 0;
                    while (N_added < N_arr[d] && k < nm) {
                        if (!std::isnan(s_buf(i, j, k)) && !std::isnan(s_buf(dest[0], dest[1], k))) {
                            // check if the destination is not a solid
                            if (nu[dest[0] * ny + dest[1]] < nu_cs) {
                                float tmp = s_buf(i, j, k);
                                s_buf(i, j, k) = s_buf(dest[0], dest[1], k);
                                s_buf(dest[0], dest[1], k) = tmp;

                                nu[i * ny + j] += inverse_nm;
                                nu[dest[0] * ny + dest[1]] -= inverse_nm;
                                
                                N_arr[d] -= 1;
                                k += 1;

                                if (k==nm) {
                                    break; // exit the loop if we have swapped all particles
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return s;
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
    // Retrieve parameters from Python object
    double g = p.attr("g").cast<double>();
    double dt = p.attr("dt").cast<double>();
    double dx = p.attr("dx").cast<double>();
    double dy = p.attr("dy").cast<double>();
    double alpha = p.attr("alpha").cast<double>();
    double nu_cs = p.attr("nu_cs").cast<double>();
    double P_stab = p.attr("P_stab").cast<double>();
    bool cyclic_BC = p.attr("cyclic_BC").cast<bool>();
    int cyclic_BC_y_offset = p.attr("cyclic_BC_y_offset").cast<int>();
    bool inertia = p.attr("inertia").cast<bool>();

    // Access mutable NumPy arrays
    auto u_buf = u.mutable_unchecked<3>();
    auto v_buf = v.mutable_unchecked<3>();
    auto s_buf = s.mutable_unchecked<3>();
    auto mask = p.attr("boundary_mask").cast<py::array_t<bool>>();
    auto mask_buf = mask.unchecked<2>();

    int nx = p.attr("nx").cast<int>();
    int ny = p.attr("ny").cast<int>();
    int nm = p.attr("nm").cast<int>();

    // Precompute useful quantities
    std::vector<double> N_swap_arr(s.shape(0) * s.shape(1), 0.0);
    std::vector<double> nu = compute_solid_fraction(s);
    std::vector<double> s_bar = compute_mean(s);
    std::vector<double> s_inv_bar = compute_s_inv_bar(s);
    std::vector<bool> some_particles = compute_some_particles(nu, mask, nx, ny);

    double v_y = std::sqrt(g * dy);
    double P_u_bar = v_y * dt / dy;  // P_u = P_u_bar * (s_inv_bar/s)
    double P_lr_ref = alpha * v_y * dt / dx / dx;
    auto delta_nu_limit = p.attr("delta_limit").cast<double>();
    double beta = std::exp(-P_stab * dt / (dx / v_y));

    auto indices = p.attr("indices").cast<std::vector<int>>();
    double inverse_nm = 1.0 / nm;

    // storage arrays
    std::array<int, 3> dest = {0, 0, 0};
    std::vector<double> u_new(nx * ny * nm, 0.0);
    std::vector<double> v_new(nx * ny * nm, 0.0);

    // Parallelize the loop using OpenMP
    // #pragma omp parallel
    {
        // Thread-safe PRNG (One per thread)
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> rand_dist(0.0, 1.0);

        // #pragma omp for
        for (size_t index = 0; index < indices.size(); index++)
        {
            int i = indices[index] / ((ny - 1) * nm);
            int j = (indices[index] / nm) % (ny - 1);
            int k = indices[index] % nm;
            int l,r,j_l, j_r;

            if (some_particles[i * ny + j])
            {
                if (std::isnan(s_buf(i, j, k)))
                {
                    if (nu[i * ny + j] < nu_cs)
                    {
                        if (mask_buf(i, j)) {
                            // Skip this cell if it is masked
                            continue;
                        }

                        double P_u = std::isnan(s_buf(i, j + 1, k)) ? 0 : P_u_bar * (s_inv_bar[i * ny + j + 1] / s_buf(i, j + 1, k));

                        std::tie(l, r, j_l, j_r) = get_lr(i, j, nx, ny, cyclic_BC_y_offset, cyclic_BC);
                        
                        double nu_here = nu[i * ny + j];
                        double nu_left = nu[l * ny + j_l];
                        double nu_right = nu[r * ny + j_r];
                        bool unstable_left = nu_left - nu_here > delta_nu_limit;
                        bool unstable_right = nu_right - nu_here > delta_nu_limit;

                        // double alpha_0 = 0.01; // alpha at nu = 0 (gaseous)
                        // double this_alpha = alpha_0*std::pow(alpha/alpha_0, nu_here/nu_cs);
                        
                        // P_lr_ref = this_alpha * v_y * dt / dx / dx;

                        double P_l = (!std::isnan(s_buf(l, j_l, k)) && unstable_left) ? P_lr_ref * s_buf(l, j_l, k) : 0;
                        double P_r = (!std::isnan(s_buf(r, j_r, k)) && unstable_right) ? P_lr_ref * s_buf(r, j_r, k) : 0;


                        if (mask_buf(i, j + 1)) {
                            P_u = 0; // Prevent upward movement into a masked cell
                        }

                        if (mask_buf(l, j_l)) {
                            P_l = 0; // Prevent leftward movement into a masked cell
                        }

                        if (mask_buf(r, j_r)) {
                            P_r = 0; // Prevent rightward movement into a masked cell
                        }

                        double P_tot = P_u + P_l + P_r;

                        // if (P_tot > 1)
                        // {
                        //     throw std::runtime_error("P_tot > 1");
                        // }

                        if (P_tot > 0)
                        {
                            double rand_val = rand_dist(gen);

                            bool found = false;

                            if (rand_val < P_u && P_u > 0)
                            {
                                dest = {i, j + 1, k};
                                found = true;
                                // if (nu[i * ny + j-1] < nu_cs) { // in the next cell in the direction of motion of the SOLID
                                    v_buf(i, j, k) += dy / dt; // where the particle will be
                                // } else {
                                    // v_buf(i, j, k) = 0; // collision
                                // }
                                // v_buf(i, j+1, k) = 0; // where the new void is
                            }
                            else if (rand_val < (P_l + P_u))
                            {
                                dest = {l, j_l, k};
                                found = true;
                                // if (nu[r * ny + j_l] < nu_cs) {
                                u_buf(i, j, k) += dx / dt; // move velocity with the particle
                                // } else {
                                    // u_buf(i, j, k) = 0; // collision
                                // }
                                // u_buf(l, j_l, k) = 0;
                            }
                            else if (rand_val < P_tot)
                            {
                                dest = {r, j_r, k};
                                found = true;
                                // if (nu[l * ny + j_r] < nu_cs) {
                                    u_buf(i, j, k) -= dx / dt; // where the particle will be
                                // } else {
                                    // u_buf(i, j, k) = 0; // collision
                                // }
                                // u_buf(r, j_r, k) = 0; // set to zero at void
                            }

                            if (found)
                            {
                                // CRITICAL SECTION 
                                // #pragma omp critical
                                {
                                    float tmp = s_buf(i, j, k);
                                    s_buf(i, j, k) = s_buf(dest[0], dest[1], dest[2]);
                                    s_buf(dest[0], dest[1], dest[2]) = tmp;

                                    nu[i * ny + j] += inverse_nm;
                                    nu[dest[0] * ny + dest[1]] -= inverse_nm;

                                    N_swap_arr[i * ny + j] += 1;
                                    N_swap_arr[dest[0] * ny + dest[1]] += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for (size_t i = 0; i < N_swap_arr.size(); i++)
    {
        N_swap_arr[i] /= nm * P_stab;
    }
    py::array_t<double> chi_out({s.shape(0), s.shape(1)}, N_swap_arr.data());

    std::vector<double> u_mean = compute_mean(u);
    std::vector<double> v_mean = compute_mean(v);
    // s = stream(u_mean, v_mean, s, mask, nu, nx, ny, nm, nu_cs, cyclic_BC_y_offset, cyclic_BC, dx, dy, dt);
    return py::make_tuple(u, v, s, c, T, chi_out, last_swap);
}

// Register module with Pybind11
PYBIND11_MODULE(d2q4_cpp, m)
{
    // m.def("stream", &stream, "Stream mass based on inertia",
    //     std::vector<double> u_mean, std::vector<double> v_mean, py::array_t<double> s,
    //     py::array_t<double> mask, std::vector<double> nu,
    //     int nx, int ny, int nm, double nu_cs,
    //     int cyclic_BC_y_offset, bool cyclic_BC, double dx, double dy, double dt);
    m.def("compute_solid_fraction", &compute_solid_fraction, "Compute solid fraction");
    m.def("compute_s_inv_bar", &compute_s_inv_bar, "Compute inverse solid fraction");
    m.def("compute_mean", &compute_mean, "Compute mean over last dimension of a 3D array");
    m.def("compute_some_particles", &compute_some_particles, "Compute some particles mask");
    m.def("compute_locally_fluid", &compute_locally_fluid, "Compute locally fluid mask");
    m.def("move_voids", &move_voids, "Moves voids in the system",
          py::arg("u"), py::arg("v"), py::arg("s"), py::arg("p"), py::arg("dummy"),
          py::arg("c") = py::none(), py::arg("T") = py::none(),
          py::arg("chi") = py::none(), py::arg("last_swap") = py::none());
}
