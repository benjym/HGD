#include "core.h"
#include <random>
#include <cmath>
#include <algorithm>
#include <array>
#include <thread>
#include <vector>
#include <limits>

// Local copy of NeighborIndices helper
struct NeighborIndicesLocal {
    std::vector<int> left;
    std::vector<int> right;
    std::vector<int> j_left;
    std::vector<int> j_right;
    std::vector<int> idx_left;
    std::vector<int> idx_right;

    NeighborIndicesLocal(int nx, int ny, int cyclic_BC_y_offset, bool cyclic_BC)
        : left(nx * ny), right(nx * ny), j_left(nx * ny), j_right(nx * ny),
          idx_left(nx * ny), idx_right(nx * ny) {
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                int idx = i * ny + j;
                int l, r, j_l, j_r;
                std::tie(l, r, j_l, j_r) = get_lr_core(i, j, nx, ny, cyclic_BC_y_offset, cyclic_BC);
                left[idx] = l;
                right[idx] = r;
                j_left[idx] = j_l;
                j_right[idx] = j_r;
                idx_left[idx] = l * ny + j_l;
                idx_right[idx] = r * ny + j_r;
            }
        }
    }
};

static inline double compute_pore_size_local(View3<double>& s, int i, int j, int k, int k_range,
                                             double void_ratio, double beta_on_6, int nm) {
    double numerator = 0.0;
    double denominator = 0.0;
    for (int offset = -k_range; offset <= k_range; offset++) {
        int k_idx = (k + offset + nm) % nm;
        if (!std::isnan(s(i, j, k_idx))) {
            numerator += std::pow(s(i, j, k_idx), 3);
            denominator += std::pow(s(i, j, k_idx), 2);
        }
    }

    if (denominator <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }

    double sauter_mean_diameter = numerator / denominator;
    if (std::isnan(sauter_mean_diameter)) {
        return std::numeric_limits<double>::infinity();
    }

    return beta_on_6 * void_ratio * sauter_mean_diameter;
}

struct Tile {
    int i0, i1; // [i0, i1)
    int j0, j1; // [j0, j1)
};

void move_particles_core_tiled(View3<double> u, View3<double> v, View3<double> s,
                               const View2<const uint8_t>& mask,
                               Params p,
                               std::vector<double>& nu,
                               std::vector<double>& chi_out) {

    int nx = p.nx, ny = p.ny, nm = p.nm;
    double seg_exponent = p.seg_exponent;

    NeighborIndicesLocal neighbors(nx, ny, p.cyclic_BC_y_offset, p.cyclic_BC);

    std::vector<double> s_inv_bar = compute_s_inv_bar_core(View3<const double>{s.data, nx, ny, nm, s.sx, s.sy, s.sz});
    std::vector<bool> some_particles = compute_some_particles_core(nu, mask, nx, ny);

    double v_y = std::sqrt(p.g * p.dy);
    double P_ud_bar = v_y * p.dt / p.dy;
    double P_lr_ref = p.alpha * v_y * p.dt / p.dx / p.dx;
    double delta_nu_limit = p.delta_limit;

    double inverse_nm = 1.0 / nm;
    double dy_over_dt = p.dy / p.dt;
    double dx_over_dt = p.dx / p.dt;
    double beta_on_6 = p.beta / 6.0;

    std::vector<double> N_swap_arr(nx * ny, 0.0);

    // Tile layout
    const int tile_i = 32;
    const int tile_j = 32;
    int tiles_i = (nx + tile_i - 1) / tile_i;
    int tiles_j = (ny + tile_j - 1) / tile_j;

    std::vector<Tile> tiles;
    tiles.reserve(tiles_i * tiles_j);
    for (int ti = 0; ti < tiles_i; ++ti) {
        for (int tj = 0; tj < tiles_j; ++tj) {
            Tile t;
            t.i0 = ti * tile_i;
            t.i1 = std::min(nx, (ti + 1) * tile_i);
            t.j0 = tj * tile_j;
            t.j1 = std::min(ny, (tj + 1) * tile_j);
            tiles.push_back(t);
        }
    }

    // Precompute tile id per cell
    std::vector<int> tile_id(nx * ny, -1);
    for (int ti = 0; ti < tiles_i; ++ti) {
        for (int tj = 0; tj < tiles_j; ++tj) {
            int id = ti * tiles_j + tj;
            int i0 = ti * tile_i;
            int i1 = std::min(nx, (ti + 1) * tile_i);
            int j0 = tj * tile_j;
            int j1 = std::min(ny, (tj + 1) * tile_j);
            for (int i = i0; i < i1; ++i) {
                for (int j = j0; j < j1; ++j) {
                    tile_id[i * ny + j] = id;
                }
            }
        }
    }

    unsigned int hw = std::max(1u, std::thread::hardware_concurrency());
    unsigned int requested = (p.max_threads > 0) ? static_cast<unsigned int>(p.max_threads) : hw;
    unsigned int thread_count = std::min(hw, requested);

    static bool printed = false;
    if (!printed) {
        printed = true;
        std::printf("[particle_tiled] threads=%u (hw=%u, max=%u), tiles=%zu (%dx%d)\n",
                    thread_count, hw, requested, tiles.size(), tile_i, tile_j);
    }
    std::vector<std::thread> workers;
    workers.reserve(thread_count);

    auto worker = [&](unsigned int thread_id) {
        std::mt19937 gen(static_cast<uint32_t>(std::random_device{}()) + thread_id);
        std::uniform_real_distribution<double> rand_dist(0.0, 1.0);

        for (size_t t = thread_id; t < tiles.size(); t += thread_count) {
            const Tile& tile = tiles[t];
            for (int k = 0; k < nm; k++) {
                for (int i = tile.i0; i < tile.i1; i++) {
                    for (int j = std::max(tile.j0, 1); j < tile.j1; j++) {
                        int idx = i * ny + j;

                        if (!some_particles[idx]) {
                            continue;
                        }
                        if (std::isnan(s(i, j, k))) {
                            continue;
                        }
                        if (mask(i, j)) {
                            continue;
                        }

                        int l = neighbors.left[idx];
                        int r = neighbors.right[idx];
                        int j_l = neighbors.j_left[idx];
                        int j_r = neighbors.j_right[idx];
                        int idx_l = neighbors.idx_left[idx];
                        int idx_r = neighbors.idx_right[idx];
                        int idx_down = idx - 1;

                        // Tile boundary check: only allow moves within same tile
                        bool down_in_tile = (j - 1 >= 0) && (tile_id[idx_down] == tile_id[idx]);
                        bool left_in_tile = (tile_id[idx_l] == tile_id[idx]);
                        bool right_in_tile = (tile_id[idx_r] == tile_id[idx]);

                        double s_here = s(i, j, k);

                        double P_d = 0.0;
                        double P_l = 0.0;
                        double P_r = 0.0;

                        int k_range = 1;

                        if (down_in_tile) {
                            double s_down = s(i, j - 1, k);
                            double void_ratio_down = (1.0 - nu[idx_down])/(nu[idx_down] + 1e-10);
                            double d_pore_down = compute_pore_size_local(s, i, j - 1, k, k_range, void_ratio_down, beta_on_6, nm);
                            if (std::isnan(s_down) && s_here <= d_pore_down) {
                                P_d = P_ud_bar * std::pow(s_inv_bar[idx] / s_here, seg_exponent);
                            }
                            if (mask(i, j - 1)) {
                                P_d = 0;
                            }
                        }

                        if (left_in_tile) {
                            double s_left = s(l, j_l, k);
                            double void_ratio_left = (1.0 - nu[idx_l]) / (nu[idx_l] + 1e-10);
                            double d_pore_left = compute_pore_size_local(s, l, j_l, k, k_range, void_ratio_left, beta_on_6, nm);
                            double nu_here = nu[idx];
                            double nu_left = nu[idx_l];
                            bool unstable_left = (std::abs(nu_here - nu_left) > delta_nu_limit);
                            if (std::isnan(s_left) && unstable_left && s_here <= d_pore_left) {
                                P_l = P_lr_ref * s_here;
                            }
                            if (mask(l, j_l)) {
                                P_l = 0;
                            }
                        }

                        if (right_in_tile) {
                            double s_right = s(r, j_r, k);
                            double void_ratio_right = (1.0 - nu[idx_r]) / (nu[idx_r] + 1e-10);
                            double d_pore_right = compute_pore_size_local(s, r, j_r, k, k_range, void_ratio_right, beta_on_6, nm);
                            double nu_here = nu[idx];
                            double nu_right = nu[idx_r];
                            bool unstable_right = (std::abs(nu_here - nu_right) > delta_nu_limit);
                            if (std::isnan(s_right) && unstable_right && s_here <= d_pore_right) {
                                P_r = P_lr_ref * s_here;
                            }
                            if (mask(r, j_r)) {
                                P_r = 0;
                            }
                        }

                        double P_tot = P_d + P_l + P_r;
                        if (P_tot <= 0) {
                            continue;
                        }

                        double rand_val = rand_dist(gen);
                        bool found = false;
                        int dest_idx = -1;
                        std::array<int, 3> dest = {0, 0, 0};

                        if (rand_val < P_d && P_d > 0) {
                            dest = {i, j - 1, k};
                            dest_idx = idx_down;
                            found = true;
                            v(i, j, k) -= dy_over_dt;
                        } else if (rand_val < (P_l + P_d)) {
                            dest = {l, j_l, k};
                            dest_idx = idx_l;
                            found = true;
                            u(i, j, k) -= dx_over_dt;
                        } else if (rand_val < P_tot) {
                            dest = {r, j_r, k};
                            dest_idx = idx_r;
                            found = true;
                            u(i, j, k) += dx_over_dt;
                        }

                        if (found) {
                            double tmp = s(i, j, k);
                            s(i, j, k) = s(dest[0], dest[1], dest[2]);
                            s(dest[0], dest[1], dest[2]) = tmp;

                            nu[idx] -= inverse_nm;
                            nu[dest_idx] += inverse_nm;

                            N_swap_arr[idx] += 1;
                            N_swap_arr[dest_idx] += 1;
                        }
                    }
                }
            }
        }
    };

    for (unsigned int t = 0; t < thread_count; ++t) {
        workers.emplace_back(worker, t);
    }

    for (auto& w : workers) {
        w.join();
    }

    // Boundary pass for cross-tile moves (single-thread, simple)
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> rand_dist(0.0, 1.0);

    for (int k = 0; k < nm; k++) {
        for (int i = 0; i < nx; i++) {
            for (int j = 1; j < ny; j++) {
                int idx = i * ny + j;
                if (!some_particles[idx] || std::isnan(s(i, j, k)) || mask(i, j)) {
                    continue;
                }

                int l = neighbors.left[idx];
                int r = neighbors.right[idx];
                int j_l = neighbors.j_left[idx];
                int j_r = neighbors.j_right[idx];
                int idx_l = neighbors.idx_left[idx];
                int idx_r = neighbors.idx_right[idx];
                int idx_down = idx - 1;

                bool down_cross = (j - 1 >= 0) && (tile_id[idx_down] != tile_id[idx]);
                bool left_cross = (tile_id[idx_l] != tile_id[idx]);
                bool right_cross = (tile_id[idx_r] != tile_id[idx]);

                double s_here = s(i, j, k);
                double s_down = s(i, j - 1, k);
                double s_right = s(r, j_r, k);
                double s_left = s(l, j_l, k);

                double void_ratio_down = (1.0 - nu[idx_down])/(nu[idx_down] + 1e-10);
                double void_ratio_right = (1.0 - nu[idx_r]) / (nu[idx_r] + 1e-10);
                double void_ratio_left = (1.0 - nu[idx_l]) / (nu[idx_l] + 1e-10);

                int k_range = 1;
                double d_pore_down = compute_pore_size_local(s, i, j - 1, k, k_range, void_ratio_down, beta_on_6, nm);
                double d_pore_right = compute_pore_size_local(s, r, j_r, k, k_range, void_ratio_right, beta_on_6, nm);
                double d_pore_left = compute_pore_size_local(s, l, j_l, k, k_range, void_ratio_left, beta_on_6, nm);

                double P_d = (down_cross && std::isnan(s_down) && s_here <= d_pore_down)
                                 ? P_ud_bar * std::pow(s_inv_bar[idx] / s_here, seg_exponent)
                                 : 0.0;

                double nu_here = nu[idx];
                double nu_left = nu[idx_l];
                double nu_right = nu[idx_r];
                bool unstable_left = (std::abs(nu_here - nu_left) > delta_nu_limit);
                bool unstable_right = (std::abs(nu_here - nu_right) > delta_nu_limit);

                double P_l = (left_cross && std::isnan(s_left) && unstable_left && s_here <= d_pore_left) ? P_lr_ref * s_here : 0.0;
                double P_r = (right_cross && std::isnan(s_right) && unstable_right && s_here <= d_pore_right) ? P_lr_ref * s_here : 0.0;

                if (down_cross && mask(i, j - 1)) {
                    P_d = 0;
                }
                if (left_cross && mask(l, j_l)) {
                    P_l = 0;
                }
                if (right_cross && mask(r, j_r)) {
                    P_r = 0;
                }

                double P_tot = P_d + P_l + P_r;
                if (P_tot <= 0) {
                    continue;
                }

                double rand_val = rand_dist(gen);
                bool found = false;
                int dest_idx = -1;
                std::array<int, 3> dest = {0, 0, 0};

                if (rand_val < P_d && P_d > 0) {
                    dest = {i, j - 1, k};
                    dest_idx = idx_down;
                    found = true;
                    v(i, j, k) -= dy_over_dt;
                } else if (rand_val < (P_l + P_d)) {
                    dest = {l, j_l, k};
                    dest_idx = idx_l;
                    found = true;
                    u(i, j, k) -= dx_over_dt;
                } else if (rand_val < P_tot) {
                    dest = {r, j_r, k};
                    dest_idx = idx_r;
                    found = true;
                    u(i, j, k) += dx_over_dt;
                }

                if (found) {
                    double tmp = s(i, j, k);
                    s(i, j, k) = s(dest[0], dest[1], dest[2]);
                    s(dest[0], dest[1], dest[2]) = tmp;

                    nu[idx] -= inverse_nm;
                    nu[dest_idx] += inverse_nm;

                    N_swap_arr[idx] += 1;
                    N_swap_arr[dest_idx] += 1;
                }
            }
        }
    }

    chi_out.resize(nx * ny);
    for (size_t i = 0; i < N_swap_arr.size(); i++) {
        chi_out[i] = N_swap_arr[i] / (nm * p.P_stab);
    }
}
