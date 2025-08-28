#include "core.h"
#include <random>
#include <cmath>
#include <algorithm>
#include <array>

std::vector<double> compute_solid_fraction_core(const View3<const double>& s) {
    int nx = s.nx, ny = s.ny, nm = s.nm;
    std::vector<double> nu(nx * ny);

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            int nan_count = 0;
            for (int k = 0; k < nm; k++) {
                if (std::isnan(s(i, j, k))) {
                    nan_count++;
                }
            }
            nu[i * ny + j] = 1.0 - (static_cast<double>(nan_count) / nm);
        }
    }

    return nu;
}

std::vector<double> compute_mean_core(const View3<const double>& a) {
    int nx = a.nx, ny = a.ny, nm = a.nm;
    std::vector<double> mean_vals(nx * ny);

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            double sum = 0.0;
            int count = 0;
            
            for (int k = 0; k < nm; k++) {
                double val = a(i, j, k);
                if (!std::isnan(val)) {
                    sum += val;
                    count++;
                }
            }
            
            mean_vals[i * ny + j] = (count > 0) ? sum / count : 0.0;
        }
    }

    return mean_vals;
}

std::vector<double> compute_s_inv_bar_core(const View3<const double>& s) {
    int nx = s.nx, ny = s.ny, nm = s.nm;
    std::vector<double> s_inv_bar(nx * ny);

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            double sum = 0.0;
            int count = 0;
            
            for (int k = 0; k < nm; k++) {
                double val = s(i, j, k);
                if (!std::isnan(val) && val > 0.0) {
                    sum += 1.0 / val;
                    count++;
                }
            }
            
            s_inv_bar[i * ny + j] = (count > 0) ? count / sum : 0.0;
        }
    }

    return s_inv_bar;
}

std::vector<bool> compute_some_particles_core(const std::vector<double>& nu,
                                             const View2<const uint8_t>& mask, int nx, int ny) {
    std::vector<bool> some_particles(nx * ny, true);

    for (int i = 1; i < nx - 1; i++) {
        for (int j = 0; j < ny; j++) {
            some_particles[i * ny + j] = (nu[i * ny + j] + 
                                        nu[(i - 1) * ny + j] + 
                                        nu[(i + 1) * ny + j] + 
                                        nu[i * ny + j + 1]) > 0.0;
        }
    }

    return some_particles;
}

std::vector<bool> compute_locally_fluid_core(const std::vector<double>& nu, int nx, int ny, double nu_cs) {
    std::vector<bool> locally_fluid(nx * ny);

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            locally_fluid[i * ny + j] = nu[i * ny + j] < nu_cs;
        }
    }

    return locally_fluid;
}

std::tuple<int, int, int, int> get_lr_core(int i, int j, int nx, int ny, int cyclic_BC_y_offset, bool cyclic_BC) {
    int l = (i == 0) ? (cyclic_BC ? nx - 1 : 0) : i - 1;
    int r = (i == nx - 1) ? (cyclic_BC ? 0 : nx - 1) : i + 1;
    int j_l, j_r;

    if (cyclic_BC_y_offset > 0 && i == nx - 1) {
        j_r = j + cyclic_BC_y_offset; // Reference cells upwards at the right boundary
        if (j_r >= ny - 1) j_r = ny - 1;
    } else {
        j_r = j;
    }
    if (cyclic_BC_y_offset > 0 && i == 0) {
        j_l = j - cyclic_BC_y_offset; // Reference cells downwards at the left boundary
        if (j_l < 0) j_l = 0;
    } else {
        j_l = j;
    }
    return std::make_tuple(l, r, j_l, j_r);
}

void move_voids_core(View3<double> u, View3<double> v, View3<double> s,
                     const View2<const uint8_t>& mask,
                     Params p,
                     std::vector<double>& nu,
                     std::vector<double>& chi_out) {
    
    int nx = p.nx, ny = p.ny, nm = p.nm;
    
    // Precompute useful quantities
    std::vector<double> s_bar = compute_mean_core(View3<const double>{s.data, nx, ny, nm, s.sx, s.sy, s.sz});
    std::vector<double> s_inv_bar = compute_s_inv_bar_core(View3<const double>{s.data, nx, ny, nm, s.sx, s.sy, s.sz});
    std::vector<bool> some_particles = compute_some_particles_core(nu, mask, nx, ny);

    double v_y = std::sqrt(p.g * p.dy);
    double P_u_bar = v_y * p.dt / p.dy;  // P_u = P_u_bar * (s_inv_bar/s)
    double P_lr_ref = p.alpha * v_y * p.dt / p.dx / p.dx;
    double delta_nu_limit = p.delta_limit;
    double beta = std::exp(-p.P_stab * p.dt / (p.dx / v_y));

    double inverse_nm = 1.0 / nm;

    // storage arrays
    std::array<int, 3> dest = {0, 0, 0};
    std::vector<double> N_swap_arr(nx * ny, 0.0);

    // Generate indices for randomized iteration - mimic what was done in d2q4.cpp
    std::vector<int> indices;
    for (int i = 1; i < nx - 1; i++) {
        for (int j = 0; j < ny - 1; j++) {
            for (int k = 0; k < nm; k++) {
                indices.push_back(i * (ny - 1) * nm + j * nm + k);
            }
        }
    }
    
    // Thread-safe PRNG
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> rand_dist(0.0, 1.0);
    
    // Shuffle indices for random processing order
    std::shuffle(indices.begin(), indices.end(), gen);

    for (size_t index = 0; index < indices.size(); index++) {
        int i = indices[index] / ((ny - 1) * nm);
        int j = (indices[index] / nm) % (ny - 1);
        int k = indices[index] % nm;
        int l, r, j_l, j_r;

        if (some_particles[i * ny + j]) {
            if (std::isnan(s(i, j, k))) {
                if (nu[i * ny + j] < p.nu_cs) {
                    if (mask(i, j)) {
                        // Skip this cell if it is masked
                        continue;
                    }

                    double P_u = std::isnan(s(i, j + 1, k)) ? 0 : P_u_bar * (s_inv_bar[i * ny + j + 1] / s(i, j + 1, k));

                    std::tie(l, r, j_l, j_r) = get_lr_core(i, j, nx, ny, p.cyclic_BC_y_offset, p.cyclic_BC);
                    
                    double nu_here = nu[i * ny + j];
                    double nu_left = nu[l * ny + j_l];
                    double nu_right = nu[r * ny + j_r];
                    bool unstable_left = nu_left - nu_here > delta_nu_limit;
                    bool unstable_right = nu_right - nu_here > delta_nu_limit;

                    double P_l = (!std::isnan(s(l, j_l, k)) && unstable_left) ? P_lr_ref * s(l, j_l, k) : 0;
                    double P_r = (!std::isnan(s(r, j_r, k)) && unstable_right) ? P_lr_ref * s(r, j_r, k) : 0;

                    if (mask(i, j + 1)) {
                        P_u = 0; // Prevent upward movement into a masked cell
                    }

                    if (mask(l, j_l)) {
                        P_l = 0; // Prevent leftward movement into a masked cell
                    }

                    if (mask(r, j_r)) {
                        P_r = 0; // Prevent rightward movement into a masked cell
                    }

                    double P_tot = P_u + P_l + P_r;

                    if (P_tot > 0) {
                        double rand_val = rand_dist(gen);

                        bool found = false;

                        if (rand_val < P_u && P_u > 0) {
                            dest = {i, j + 1, k};
                            found = true;
                            v(i, j, k) += p.dy / p.dt;
                        }
                        else if (rand_val < (P_l + P_u)) {
                            dest = {l, j_l, k};
                            found = true;
                            u(i, j, k) += p.dx / p.dt;
                        }
                        else if (rand_val < P_tot) {
                            dest = {r, j_r, k};
                            found = true;
                            u(i, j, k) -= p.dx / p.dt;
                        }

                        if (found) {
                            double tmp = s(i, j, k);
                            s(i, j, k) = s(dest[0], dest[1], dest[2]);
                            s(dest[0], dest[1], dest[2]) = tmp;

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

    // Normalize chi_out
    chi_out.resize(nx * ny);
    for (size_t i = 0; i < N_swap_arr.size(); i++) {
        chi_out[i] = N_swap_arr[i] / (nm * p.P_stab);
    }
}

void stream_core(const std::vector<double>& u_mean,
                 const std::vector<double>& v_mean,
                 View3<double> s,
                 const View2<const uint8_t>& mask,
                 std::vector<double>& nu,
                 const Params& p) {
    
    int nx = p.nx, ny = p.ny, nm = p.nm;
    double inverse_nm = 1.0 / nm;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> rand_dist(0.0, 1.0);

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            int l, r, j_l, j_r;
            std::tie(l, r, j_l, j_r) = get_lr_core(i, j, nx, ny, p.cyclic_BC_y_offset, p.cyclic_BC);

            double P_u = mask(i, j + 1) ? v_mean[i * ny + j + 1] * p.dy / p.dt : 0;
            double u_left = (u_mean[i * ny + j_l] < 0) ? u_mean[i * ny + j_l] : 0;
            double P_l = mask(l, j_l) ? u_left * p.dy / p.dt : 0;
            double u_right = (u_mean[i * ny + j_r] > 0) ? u_mean[i * ny + j_r] : 0;
            double P_r = mask(r, j_r) ? u_right * p.dy / p.dt : 0;

            int N_u = static_cast<int>(nu[i * ny + j] * P_u);
            int N_l = static_cast<int>(nu[i * ny + j] * P_l);
            int N_r = static_cast<int>(nu[i * ny + j] * P_r);

            std::vector<int> N_arr = {N_u, N_l, N_r};
            std::vector<int> dests = {i, j + 1, l, j_l, r, j_r};
            int k = 0;
            
            for (int d = 0; d < 3; d += 1) {
                std::array<int, 2> dest = {dests[2 * d], dests[2 * d + 1]};
                int N_added = 0;
                
                while (N_added < N_arr[d] && k < nm) {
                    if (!std::isnan(s(i, j, k)) && !std::isnan(s(dest[0], dest[1], k))) {
                        // check if the destination is not a solid
                        if (nu[dest[0] * ny + dest[1]] < p.nu_cs) {
                            double tmp = s(i, j, k);
                            s(i, j, k) = s(dest[0], dest[1], k);
                            s(dest[0], dest[1], k) = tmp;

                            nu[i * ny + j] += inverse_nm;
                            nu[dest[0] * ny + dest[1]] -= inverse_nm;
                            
                            N_added += 1;

                            if (k == nm) {
                                break; // exit the loop if we have swapped all particles
                            }
                        }
                    }
                    k += 1;
                }
            }
        }
    }
}
