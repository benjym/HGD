#include "core.h"
#include <random>
#include <cmath>
#include <algorithm>
#include <array>

double inf = std::numeric_limits<double>::infinity();

// Helper struct to store precomputed neighbor indices
struct NeighborIndices {
    std::vector<int> left;      // left neighbor i-index for each (i,j)
    std::vector<int> right;     // right neighbor i-index for each (i,j)
    std::vector<int> j_left;    // j-index when accessing left neighbor
    std::vector<int> j_right;   // j-index when accessing right neighbor
    std::vector<int> idx_left;  // flattened index for left neighbor
    std::vector<int> idx_right; // flattened index for right neighbor
    
    NeighborIndices(int nx, int ny, int cyclic_BC_y_offset, bool cyclic_BC) 
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

// static inline double compute_pore_size(double s_a, double s_b, double s_c,
                                    //    double void_ratio, double beta_on_6) {
static inline double compute_pore_size(View3<double>& s, int i, int j, int k, int k_range, double void_ratio, double beta_on_6, int nm) {
    double numerator = 0.0;
    double denominator = 0.0;
    for (int offset = -k_range; offset <= k_range; offset++) {
        int k_idx = (k + offset + nm) % nm; // wrap around
        if (!std::isnan(s(i, j, k_idx))) {
            numerator += std::pow(s(i, j, k_idx), 3);
            denominator += std::pow(s(i, j, k_idx), 2);    
        }
    }

    if (denominator <= 0.0) {
        return inf;
    }

    double sauter_mean_diameter = numerator / denominator;
    if (std::isnan(sauter_mean_diameter)) {
        return inf;
    }

    return beta_on_6 * void_ratio * sauter_mean_diameter;
}

enum class SpaceCriterion {
    NuCs,
    PoreSize
};

static inline SpaceCriterion parse_space_criterion(const Params& p) {
    if (p.space_criterion == "nu_cs" || p.space_criterion == "nu") {
        return SpaceCriterion::NuCs;
    }
    if (p.space_criterion == "pore_size" || p.space_criterion == "d_pore" || p.space_criterion == "pore") {
        return SpaceCriterion::PoreSize;
    }
    throw std::invalid_argument("Invalid space_criterion: " + p.space_criterion + ". Use 'nu_cs' or 'pore_size'.");
}

static inline bool has_space_nu_cs(const std::vector<double>& nu, int dest_idx, double inverse_nm, double nu_cs) {
    return (nu[dest_idx] + inverse_nm) <= (nu_cs + 1e-12);
}

static inline bool has_space_pore_size(View3<double>& s,
                                       int dest_i,
                                       int dest_j,
                                       int k,
                                       double s_here,
                                       const std::vector<double>& nu,
                                       int dest_idx,
                                       double beta_on_6,
                                       int nm) {
    double void_ratio_dest = (1.0 - nu[dest_idx]) / (nu[dest_idx] + 1e-10);
    double d_pore_dest = compute_pore_size(s, dest_i, dest_j, k, 1, void_ratio_dest, beta_on_6, nm);
    return s_here <= d_pore_dest;
}

static inline bool has_space_for_particle(View3<double>& s,
                                          int dest_i,
                                          int dest_j,
                                          int k,
                                          double s_here,
                                          const std::vector<double>& nu,
                                          int dest_idx,
                                          double inverse_nm,
                                          double nu_cs,
                                          double beta_on_6,
                                          int nm,
                                          SpaceCriterion criterion) {
    if (criterion == SpaceCriterion::NuCs) {
        return has_space_nu_cs(nu, dest_idx, inverse_nm, nu_cs);
    }
    return has_space_pore_size(s, dest_i, dest_j, k, s_here, nu, dest_idx, beta_on_6, nm);
}

static inline bool should_relax_particle(View3<double>& s,
                                         int i,
                                         int j,
                                         int k,
                                         const std::vector<double>& nu,
                                         int idx,
                                         double nu_cs,
                                         double beta_on_6,
                                         int nm,
                                         SpaceCriterion criterion) {
    if (criterion == SpaceCriterion::NuCs) {
        return nu[idx] >= (nu_cs - 1e-12);
    }

    double s_here = s(i, j, k);
    if (std::isnan(s_here)) {
        return false;
    }

    double void_ratio_here = (1.0 - nu[idx]) / (nu[idx] + 1e-10);
    double d_pore_here = compute_pore_size(s, i, j, k, 1, void_ratio_here, beta_on_6, nm);
    return s_here > d_pore_here;
}

static inline int wrap_index(int idx, int n) {
    if (n <= 0) {
        return 0;
    }
    int wrapped = idx % n;
    return (wrapped < 0) ? (wrapped + n) : wrapped;
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

void move_core(View3<double> u, View3<double> v, View3<double> s,
                     const View2<const uint8_t>& mask,
                     Params p,
                     std::vector<double>& nu,
                     std::vector<double>& chi_out) {

    if (p.move_type == "void") {
        move_voids_core(u, v, s, mask, p, nu, chi_out);
    } else if (p.move_type == "particle") {
        move_particles_core(u, v, s, mask, p, nu, chi_out);
    } else if (p.move_type == "particle_tiled") {
        move_particles_core_tiled(u, v, s, mask, p, nu, chi_out);
    } else {
        throw std::invalid_argument("Invalid move_type: " + p.move_type);
    }
}

void move_particles_core(View3<double> u, View3<double> v, View3<double> s,
                     const View2<const uint8_t>& mask,
                     Params p,
                     std::vector<double>& nu,
                     std::vector<double>& chi_out) {

    int nx = p.nx, ny = p.ny, nm = p.nm;
    double seg_exponent = p.seg_exponent;
    
    // Precompute neighbor indices once
    NeighborIndices neighbors(nx, ny, p.cyclic_BC_y_offset, p.cyclic_BC);
    
    // Precompute useful quantities
    std::vector<double> s_bar = compute_mean_core(View3<const double>{s.data, nx, ny, nm, s.sx, s.sy, s.sz});
    std::vector<double> s_inv_bar = compute_s_inv_bar_core(View3<const double>{s.data, nx, ny, nm, s.sx, s.sy, s.sz});
    std::vector<bool> some_particles = compute_some_particles_core(nu, mask, nx, ny);

    double v_y = std::sqrt(p.g * p.dy);
    double P_ud_bar = v_y * p.dt / p.dy;  // P_u = P_u_bar * (s_inv_bar/s)
    double P_lr_ref = p.alpha * v_y * p.dt / p.dx / p.dx;
    double delta_nu_limit = p.delta_limit;

    double inverse_nm = 1.0 / nm;
    double dy_over_dt = p.dy / p.dt;
    double dx_over_dt = p.dx / p.dt;
    double beta_on_6 = p.beta / 6.0;
    const SpaceCriterion space_criterion = parse_space_criterion(p);

    // storage arrays
    std::array<int, 3> dest = {0, 0, 0};
    std::vector<double> N_swap_arr(nx * ny, 0.0);
    
    // Thread-safe PRNG
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> rand_dist(0.0, 1.0);
    
    for (int k = 0; k < nm; k++) {
        for (int i = 0; i < nx; i++) {
            for (int j = 1; j < ny; j++) {
                int idx = i * ny + j;

                if (some_particles[idx]) {
                    if (!std::isnan(s(i, j, k))) { // particle present

                        if (mask(i, j)) {
                            continue;
                        }

                        // Use precomputed neighbor indices
                        int l = neighbors.left[idx];
                        int r = neighbors.right[idx];
                        int j_l = neighbors.j_left[idx];
                        int j_r = neighbors.j_right[idx];
                        int idx_l = neighbors.idx_left[idx];
                        int idx_r = neighbors.idx_right[idx];
                        // int idx_up = idx + 1;
                        int idx_down = idx - 1;

                        double s_here = s(i, j, k);
                        double s_down = s(i, j - 1, k);
                        double s_right = s(r, j_r, k);
                        double s_left = s(l, j_l, k);

                        bool down_void = std::isnan(s_down);
                        bool left_void = std::isnan(s_left);
                        bool right_void = std::isnan(s_right);

                        bool space_down = down_void && has_space_for_particle(
                            s, i, j - 1, k, s_here, nu, idx_down, inverse_nm, p.nu_cs, beta_on_6, nm, space_criterion);
                        bool space_left = left_void && has_space_for_particle(
                            s, l, j_l, k, s_here, nu, idx_l, inverse_nm, p.nu_cs, beta_on_6, nm, space_criterion);
                        bool space_right = right_void && has_space_for_particle(
                            s, r, j_r, k, s_here, nu, idx_r, inverse_nm, p.nu_cs, beta_on_6, nm, space_criterion);

                        double P_d = (down_void && space_down)
                                         ? P_ud_bar * std::pow(s_inv_bar[idx] / s_here, seg_exponent)
                                         : 0.0;
                        
                        double nu_here = nu[idx];
                        double nu_left = nu[idx_l];
                        double nu_right = nu[idx_r];
                        // Determine if left/right moves are unstable based on delta_nu_limit
                        // Limits slopes via geometrical constraint on density gradients

                        bool unstable_left = (std::abs(nu_here - nu_left) > delta_nu_limit);
                        bool unstable_right = (std::abs(nu_here - nu_right) > delta_nu_limit);

                        // 
                        double P_l = (left_void && unstable_left && space_left) ? P_lr_ref * s_here : 0.0;
                        double P_r = (right_void && unstable_right && space_right) ? P_lr_ref * s_here : 0.0;

                        if (mask(i, j - 1)) {
                            P_d = 0; // Prevent downward movement into a masked cell
                        }

                        if (mask(l, j_l)) {
                            P_l = 0; // Prevent leftward movement into a masked cell
                        }

                        if (mask(r, j_r)) {
                            P_r = 0; // Prevent rightward movement into a masked cell
                        }

                        double P_tot = P_d + P_l + P_r;

                        if (P_tot > 0) {
                            double rand_val = rand_dist(gen);

                            bool found = false;
                            int dest_idx = -1;

                            if (rand_val < P_d && P_d > 0) {
                                dest = {i, j - 1, k};
                                dest_idx = idx_down;
                                found = true;
                                v(i, j, k) -= P_d * dy_over_dt;
                            }
                            else if (rand_val < (P_l + P_d)) {
                                dest = {l, j_l, k};
                                dest_idx = idx_l;
                                found = true;
                                u(i, j, k) -= P_l * dx_over_dt;
                            }
                            else if (rand_val < P_tot) {
                                dest = {r, j_r, k};
                                dest_idx = idx_r;
                                found = true;
                                u(i, j, k) += P_r * dx_over_dt;
                            }

                            if (found) {
                                double tmp = s(i, j, k);
                                s(i, j, k) = s(dest[0], dest[1], dest[2]);
                                s(dest[0], dest[1], dest[2]) = tmp;

                                // Stream velocity with the moved particle.
                                double u_tmp = u(i, j, k);
                                u(i, j, k) = u(dest[0], dest[1], dest[2]);
                                u(dest[0], dest[1], dest[2]) = u_tmp;

                                double v_tmp = v(i, j, k);
                                v(i, j, k) = v(dest[0], dest[1], dest[2]);
                                v(dest[0], dest[1], dest[2]) = v_tmp;

                                nu[idx] -= inverse_nm;
                                nu[dest_idx] += inverse_nm;

                                N_swap_arr[idx] += 1;
                                N_swap_arr[dest_idx] += 1;
                            }
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

void move_voids_core(View3<double> u, View3<double> v, View3<double> s,
                     const View2<const uint8_t>& mask,
                     Params p,
                     std::vector<double>& nu,
                     std::vector<double>& chi_out) {
    
    int nx = p.nx, ny = p.ny, nm = p.nm;
    double seg_exponent = p.seg_exponent;
    
    // Precompute neighbor indices once
    NeighborIndices neighbors(nx, ny, p.cyclic_BC_y_offset, p.cyclic_BC);
    
    // Precompute useful quantities
    std::vector<double> s_bar = compute_mean_core(View3<const double>{s.data, nx, ny, nm, s.sx, s.sy, s.sz});
    std::vector<double> s_inv_bar = compute_s_inv_bar_core(View3<const double>{s.data, nx, ny, nm, s.sx, s.sy, s.sz});
    std::vector<bool> some_particles = compute_some_particles_core(nu, mask, nx, ny);

    double v_y = std::sqrt(p.g * p.dy);
    double P_u_bar = v_y * p.dt / p.dy;  // P_u = P_u_bar * (s_inv_bar/s)
    double P_lr_ref = p.alpha * v_y * p.dt / p.dx / p.dx;
    double delta_nu_limit = p.delta_limit;

    double inverse_nm = 1.0 / nm;
    double dy_over_dt = p.dy / p.dt;
    double dx_over_dt = p.dx / p.dt;
    double beta_on_6 = p.beta / 6.0;

    // storage arrays
    std::array<int, 3> dest = {0, 0, 0};
    std::vector<double> N_swap_arr(nx * ny, 0.0);

    // Generate indices for randomized iteration - mimic what was done in d2q4.cpp
    // Store i, j, k directly and the 2D flattened index for efficiency
    std::vector<int> i_cache, j_cache, k_cache, idx_cache;
    i_cache.reserve(nx * (ny - 1) * nm);
    j_cache.reserve(nx * (ny - 1) * nm);
    k_cache.reserve(nx * (ny - 1) * nm);
    idx_cache.reserve(nx * (ny - 1) * nm);
    
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny - 1; j++) {
            int idx = i * ny + j;
            for (int k = 0; k < nm; k++) {
                i_cache.push_back(i);
                j_cache.push_back(j);
                k_cache.push_back(k);
                idx_cache.push_back(idx);
            }
        }
    }
    
    // Thread-safe PRNG
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> rand_dist(0.0, 1.0);
    
    // Shuffle indices for random processing order
    std::vector<size_t> shuffle_indices(i_cache.size());
    for (size_t i = 0; i < shuffle_indices.size(); i++) {
        shuffle_indices[i] = i;
    }
    std::shuffle(shuffle_indices.begin(), shuffle_indices.end(), gen);

    for (size_t idx_pos = 0; idx_pos < shuffle_indices.size(); idx_pos++) {
        size_t index = shuffle_indices[idx_pos];
        int i = i_cache[index];
        int j = j_cache[index];
        int k = k_cache[index];
        int idx = idx_cache[index];
        int idx_up = idx + 1;  // i * ny + j + 1

        if (some_particles[idx]) {
            if (std::isnan(s(i, j, k))) {
                // if (nu[idx] < p.nu_cs) {
                    if (mask(i, j)) {
                        continue;
                    }

                    // Use precomputed neighbor indices
                    int l = neighbors.left[idx];
                    int r = neighbors.right[idx];
                    int j_l = neighbors.j_left[idx];
                    int j_r = neighbors.j_right[idx];
                    int idx_l = neighbors.idx_left[idx];
                    int idx_r = neighbors.idx_right[idx];
                    int idx_up = idx + 1;

                    double void_ratio_here = (1.0 - nu[idx]) / (nu[idx] + 1e-10);
                    double void_ratio_up = (1.0-nu[idx_up])/(nu[idx_up] + 1e-10);
                    double void_ratio_right = (1.0 - nu[idx_r]) / (nu[idx_r] + 1e-10);
                    double void_ratio_left = (1.0 - nu[idx_l]) / (nu[idx_l] + 1e-10);

                    int l_up, r_up, j_l_up, j_r_up;
                    std::tie(l_up, r_up, j_l_up, j_r_up) = get_lr_core(i, j + 1, nx, ny, p.cyclic_BC_y_offset, p.cyclic_BC);

                    int l_down, r_down, j_l_down, j_r_down;
                    std::tie(l_down, r_down, j_l_down, j_r_down) = get_lr_core(i, j - 1, nx, ny, p.cyclic_BC_y_offset, p.cyclic_BC);

                    double s_up = s(i, j + 1, k);
                    double s_right = s(r, j_r, k);
                    double s_left = s(l, j_l, k);

                    int k_range = 1; // consider particles k-1, k, k+1 for pore size calculation
                    double d_pore_here = compute_pore_size(s, i, j, k, k_range, void_ratio_here, beta_on_6, nm);
                    double d_pore_up = compute_pore_size(s, i, j+1, k, k_range, void_ratio_here, beta_on_6, nm);
                    double d_pore_right = compute_pore_size(s, r, j_r, k, k_range, void_ratio_right, beta_on_6, nm);
                    double d_pore_left = compute_pore_size(s, l, j_l, k, k_range, void_ratio_left, beta_on_6, nm);

                    double P_u = std::isnan(s_up) ? 0 : P_u_bar * std::pow(s_inv_bar[idx_up] / s_up, seg_exponent);

                    // double d_pore_up_eff = (d_pore_here > 0.0) ? d_pore_here : d_pore_up;
                    // double d_pore_right_eff = (d_pore_here > 0.0) ? d_pore_here : d_pore_right;
                    // double d_pore_left_eff = (d_pore_here > 0.0) ? d_pore_here : d_pore_left;

                    d_pore_up = std::min(d_pore_here, d_pore_up);
                    d_pore_right = std::min(d_pore_here, d_pore_right);
                    d_pore_left = std::min(d_pore_here, d_pore_left);

                    if (s_up > d_pore_up) P_u = 0; // Prevent upward movement of void if particle is larger than pore size
                    
                    double nu_here = nu[idx];
                    double nu_left = nu[idx_l];
                    double nu_right = nu[idx_r];
                    bool unstable_left = nu_left - nu_here > delta_nu_limit;
                    bool unstable_right = nu_right - nu_here > delta_nu_limit;

                    double P_l = (!std::isnan(s_left) && unstable_left) ? P_lr_ref * s_left : 0;
                    double P_r = (!std::isnan(s_right) && unstable_right) ? P_lr_ref * s_right : 0;

                    if (!unstable_right && s_right > d_pore_right) P_r = 0; // Prevent rightward movement of void if particle is larger than pore size
                    if (!unstable_left && s_left > d_pore_left) P_l = 0;   // Prevent leftward movement of void if particle is larger than pore size

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
                        int dest_idx = -1;

                        if (rand_val < P_u && P_u > 0) {
                            dest = {i, j + 1, k};
                            dest_idx = idx_up;
                            found = true;
                            v(i, j, k) += dy_over_dt;
                        }
                        else if (rand_val < (P_l + P_u)) {
                            dest = {l, j_l, k};
                            dest_idx = idx_l;
                            found = true;
                            u(i, j, k) += dx_over_dt;
                        }
                        else if (rand_val < P_tot) {
                            dest = {r, j_r, k};
                            dest_idx = idx_r;
                            found = true;
                            u(i, j, k) -= dx_over_dt;
                        }

                        if (found) {
                            double tmp = s(i, j, k);
                            s(i, j, k) = s(dest[0], dest[1], dest[2]);
                            s(dest[0], dest[1], dest[2]) = tmp;

                            nu[idx] += inverse_nm;
                            nu[dest_idx] -= inverse_nm;

                            N_swap_arr[idx] += 1;
                            N_swap_arr[dest_idx] += 1;
                        }
                    }
                // }
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
    double dy_over_dt = p.dy / p.dt;
    
    // Precompute neighbor indices once
    NeighborIndices neighbors(nx, ny, p.cyclic_BC_y_offset, p.cyclic_BC);

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            int idx = i * ny + j;
            int idx_up = idx + 1;
            
            // Use precomputed neighbor indices
            int l = neighbors.left[idx];
            int r = neighbors.right[idx];
            int j_l = neighbors.j_left[idx];
            int j_r = neighbors.j_right[idx];
            int idx_l = neighbors.idx_left[idx];
            int idx_r = neighbors.idx_right[idx];

            double v_up = (v_mean[idx_up] < 0) ? -v_mean[idx_up] : 0;
            double P_u = mask(i, j + 1) ? v_up * dy_over_dt : 0;
            double v_down = (v_mean[idx] > 0) ? v_mean[idx] : 0;
            double P_d = mask(i, j) ? v_down * dy_over_dt : 0;
            double u_left = (u_mean[idx_l] < 0) ? u_mean[idx_l] : 0;
            double P_l = mask(l, j_l) ? u_left * dy_over_dt : 0;
            double u_right = (u_mean[idx_r] > 0) ? u_mean[idx_r] : 0;
            double P_r = mask(r, j_r) ? u_right * dy_over_dt : 0;

            int N_u = static_cast<int>(nu[idx] * P_u);
            int N_l = static_cast<int>(nu[idx] * P_l);
            int N_r = static_cast<int>(nu[idx] * P_r);

            std::vector<int> N_arr = {N_u, N_l, N_r};
            std::vector<int> dests = {i, j + 1, l, j_l, r, j_r};
            int k = 0;

            // printf("At cell (%d, %d): N_u=%d, N_l=%d, N_r=%d\n", i, j, N_u, N_l, N_r);
            
            for (int d = 0; d < 3; d += 1) {
                std::array<int, 2> dest = {dests[2 * d], dests[2 * d + 1]};
                int dest_idx = dest[0] * ny + dest[1];
                int N_added = 0;
                
                while (N_added < N_arr[d] && k < nm) {
                    if (!std::isnan(s(i, j, k)) && std::isnan(s(dest[0], dest[1], k))) {
                        // check if the destination is not a solid
                        // if (nu[dest_idx] < p.nu_cs) {
                        double d_pore_dest = compute_pore_size(s, dest[0], dest[1], k, 1, (1.0 - nu[dest_idx])/(nu[dest_idx] + 1e-10), p.beta / 6.0, nm);

                        printf("d_pore_dest: %f, s: %f\n", d_pore_dest, s(i, j, k));
                        if (s(i, j, k) < d_pore_dest) {
                            double tmp = s(i, j, k);
                            s(i, j, k) = s(dest[0], dest[1], k);
                            s(dest[0], dest[1], k) = tmp;

                            nu[idx] += inverse_nm;
                            nu[dest_idx] -= inverse_nm;
                            
                            N_added += 1;
                        }
                    }
                    k += 1;
                }
            }
        }
    }
}

void stream_core_lbm_zero_eq(const std::vector<double>& u_mean,
                             const std::vector<double>& v_mean,
                             View3<double> u,
                             View3<double> v,
                             View3<double> s,
                             const View2<const uint8_t>& mask,
                             std::vector<double>& nu,
                             const Params& p) {
    const int nx = p.nx;
    const int ny = p.ny;
    const int nm = p.nm;
    const double inverse_nm = 1.0 / static_cast<double>(nm);
    const double beta_on_6 = p.beta / 6.0;
    const SpaceCriterion space_criterion = parse_space_criterion(p);

    // Precompute lateral neighbors once.
    NeighborIndices neighbors(nx, ny, p.cyclic_BC_y_offset, p.cyclic_BC);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> unit01(0.0, 1.0);

    auto sample_count = [&](double prob, int n_solid) -> int {
        if (prob <= 0.0 || n_solid <= 0) {
            return 0;
        }
        double expected = prob * static_cast<double>(n_solid);
        int count = static_cast<int>(std::floor(expected));
        double frac = expected - static_cast<double>(count);
        if (unit01(gen) < frac) {
            count += 1;
        }
        return std::min(count, n_solid);
    };

    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            const int idx = i * ny + j;
            if (mask(i, j)) {
                continue;
            }

            std::vector<int> solid_indices;
            solid_indices.reserve(nm);
            for (int k = 0; k < nm; ++k) {
                if (!std::isnan(s(i, j, k))) {
                    solid_indices.push_back(k);
                }
            }

            const int n_solid = static_cast<int>(solid_indices.size());
            if (n_solid == 0) {
                continue;
            }

            const int l = neighbors.left[idx];
            const int r = neighbors.right[idx];
            const int j_l = neighbors.j_left[idx];
            const int j_r = neighbors.j_right[idx];

            // LBM-inspired pure streaming (f_eq = 0, no forcing):
            // directional transport probability comes directly from local velocity.
            double P_u = 0.0;
            double P_d = 0.0;
            double P_l = 0.0;
            double P_r = 0.0;

            const double u_here = u_mean[idx];
            const double v_here = v_mean[idx];
            const double cfl_x = (p.dx > 0.0) ? (p.dt / p.dx) : 0.0;
            const double cfl_y = (p.dy > 0.0) ? (p.dt / p.dy) : 0.0;

            if (u_here > 0.0) {
                P_r = std::min(1.0, u_here * cfl_x);
            } else if (u_here < 0.0) {
                P_l = std::min(1.0, -u_here * cfl_x);
            }

            if (v_here > 0.0) {
                P_u = std::min(1.0, v_here * cfl_y);
            } else if (v_here < 0.0) {
                P_d = std::min(1.0, -v_here * cfl_y);
            }

            // Gate moves that cross the boundary or masked cells.
            if (j >= ny - 1 || mask(i, j + 1)) {
                P_u = 0.0;
            }
            if (j <= 0 || mask(i, j - 1)) {
                P_d = 0.0;
            }
            if (mask(l, j_l)) {
                P_l = 0.0;
            }
            if (mask(r, j_r)) {
                P_r = 0.0;
            }

            double P_tot = P_u + P_d + P_l + P_r;
            if (P_tot <= 0.0) {
                continue;
            }

            // A particle can stream in only one direction per sub-step.
            if (P_tot > 1.0) {
                const double inv = 1.0 / P_tot;
                P_u *= inv;
                P_d *= inv;
                P_l *= inv;
                P_r *= inv;
            }

            std::array<int, 4> N_req = {
                sample_count(P_u, n_solid),
                sample_count(P_d, n_solid),
                sample_count(P_l, n_solid),
                sample_count(P_r, n_solid),
            };

            std::array<std::array<int, 2>, 4> dests = {{
                {i, j + 1},   // up
                {i, j - 1},   // down
                {l, j_l},     // left
                {r, j_r},     // right
            }};

            std::array<int, 4> dir_order = {0, 1, 2, 3};
            std::shuffle(dir_order.begin(), dir_order.end(), gen);
            std::shuffle(solid_indices.begin(), solid_indices.end(), gen);

            int solid_cursor = 0;
            for (int ord = 0; ord < 4 && solid_cursor < n_solid; ++ord) {
                int d = dir_order[ord];
                int dest_i = dests[d][0];
                int dest_j = dests[d][1];
                int dest_idx = dest_i * ny + dest_j;

                int moved = 0;
                while (moved < N_req[d] && solid_cursor < n_solid) {
                    int k = solid_indices[solid_cursor++];
                    if (std::isnan(s(i, j, k))) {
                        continue;
                    }
                    if (mask(dest_i, dest_j)) {
                        continue;
                    }
                    if (!std::isnan(s(dest_i, dest_j, k))) {
                        continue;
                    }
                    if (!has_space_for_particle(
                            s,
                            dest_i,
                            dest_j,
                            k,
                            s(i, j, k),
                            nu,
                            dest_idx,
                            inverse_nm,
                            p.nu_cs,
                            beta_on_6,
                            nm,
                            space_criterion)) {
                        continue;
                    }

                    double tmp = s(i, j, k);
                    s(i, j, k) = s(dest_i, dest_j, k);
                    s(dest_i, dest_j, k) = tmp;

                    // Stream particle momentum with the particle and clear the vacated site.
                    u(dest_i, dest_j, k) = u(i, j, k);
                    v(dest_i, dest_j, k) = v(i, j, k);
                    u(i, j, k) = 0.0;
                    v(i, j, k) = 0.0;

                    nu[idx] -= inverse_nm;
                    nu[dest_idx] += inverse_nm;
                    moved += 1;
                }
            }
        }
    }

    // Relax particle velocities toward equilibrium (u=v=0) using the selected
    // space criterion. Dilute/gas-like states are left unrelaxed.
    const bool relax_to_zero = (p.tau > 0.0);
    const double relax_factor = relax_to_zero ? std::exp(-p.dt / p.tau) : 1.0;

    // Remove stale velocity in voids (and in masked cells), then selectively relax.
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            int idx = i * ny + j;
            for (int k = 0; k < nm; ++k) {
                if (mask(i, j) || std::isnan(s(i, j, k))) {
                    u(i, j, k) = 0.0;
                    v(i, j, k) = 0.0;
                } else if (relax_to_zero && should_relax_particle(
                        s, i, j, k, nu, idx, p.nu_cs, beta_on_6, nm, space_criterion)) {
                    u(i, j, k) *= relax_factor;
                    v(i, j, k) *= relax_factor;
                }
            }
        }
    }
}
