#include "core.h"
#include <random>
#include <cmath>
#include <algorithm>
#include <array>
#include <numeric>
#include <unordered_map>
#include <mutex>
#include <memory>

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

namespace {

struct ParticleOrderKey {
    int nx;
    int ny;
    int nm;

    bool operator==(const ParticleOrderKey& other) const {
        return nx == other.nx && ny == other.ny && nm == other.nm;
    }
};

struct ParticleOrderKeyHash {
    size_t operator()(const ParticleOrderKey& key) const {
        size_t h = static_cast<size_t>(key.nx);
        h = h * 1315423911u + static_cast<size_t>(key.ny);
        h = h * 1315423911u + static_cast<size_t>(key.nm);
        return h;
    }
};

struct CellOrderKey {
    int nx;
    int ny;

    bool operator==(const CellOrderKey& other) const {
        return nx == other.nx && ny == other.ny;
    }
};

struct CellOrderKeyHash {
    size_t operator()(const CellOrderKey& key) const {
        size_t h = static_cast<size_t>(key.nx);
        h = h * 1315423911u + static_cast<size_t>(key.ny);
        return h;
    }
};

std::shared_ptr<const std::vector<int>> get_cached_particle_iteration_order(int nx, int ny, int nm) {
    static std::unordered_map<ParticleOrderKey, std::shared_ptr<const std::vector<int>>, ParticleOrderKeyHash> cache;
    static std::mutex cache_mutex;

    const ParticleOrderKey key{nx, ny, nm};
    std::lock_guard<std::mutex> lock(cache_mutex);

    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }

    const int active_ny = std::max(0, ny - 1);
    const size_t total_count = static_cast<size_t>(nx) * static_cast<size_t>(active_ny) * static_cast<size_t>(nm);
    auto order = std::make_shared<std::vector<int>>(total_count);
    std::iota(order->begin(), order->end(), 0);

    std::mt19937 order_gen(std::random_device{}());
    std::shuffle(order->begin(), order->end(), order_gen);

    it = cache.emplace(key, order).first;
    return it->second;
}

std::shared_ptr<const std::vector<int>> get_cached_cell_iteration_order(int nx, int ny) {
    static std::unordered_map<CellOrderKey, std::shared_ptr<const std::vector<int>>, CellOrderKeyHash> cache;
    static std::mutex cache_mutex;

    const CellOrderKey key{nx, ny};
    std::lock_guard<std::mutex> lock(cache_mutex);

    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }

    const int active_ny = std::max(0, ny - 1);
    const size_t total_count = static_cast<size_t>(nx) * static_cast<size_t>(active_ny);
    auto order = std::make_shared<std::vector<int>>(total_count);
    std::iota(order->begin(), order->end(), 0);

    std::mt19937 order_gen(std::random_device{}());
    std::shuffle(order->begin(), order->end(), order_gen);

    it = cache.emplace(key, order).first;
    return it->second;
}

} // namespace

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
            const int idx = i * ny + j;
            const double nu_here = nu[idx];
            const double nu_left = nu[(i - 1) * ny + j];
            const double nu_right = nu[(i + 1) * ny + j];
            const double nu_up = (j + 1 < ny) ? nu[idx + 1] : 0.0;

            some_particles[idx] = (nu_here + nu_left + nu_right + nu_up) > 0.0;
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

static inline double get_above_density(const std::vector<double>& nu,
                                       const View2<const uint8_t>& mask,
                                       int i,
                                       int j,
                                       int ny,
                                       double nu_here) {
    if (j + 1 < ny) {
        return mask(i, j + 1) ? nu_here : nu[i * ny + (j + 1)];
    }
    // Open top boundary behaves as void.
    return 0.0;
}

static inline bool lateral_unstable_with_surface_gate(double nu_here,
                                                      double nu_dest,
                                                      double nu_up_here,
                                                      double nu_up_dest,
                                                      double delta_nu_limit_h,
                                                      double nu_surface_limit,
                                                      bool bulk_unstable) {
    const bool near_surface = (std::min(nu_up_here, nu_up_dest) < nu_surface_limit);
    if (near_surface) {
        // Near a free surface, enforce slope stability.
        return (nu_here - nu_dest) > delta_nu_limit_h;
    }
    // Deep in the bulk, use the space/packing criterion.
    return bulk_unstable;
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

    if (p.move_type == "particle") {
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
    std::vector<double> s_inv_bar = compute_s_inv_bar_core(View3<const double>{s.data, nx, ny, nm, s.sx, s.sy, s.sz});
    std::vector<bool> some_particles = compute_some_particles_core(nu, mask, nx, ny);

    double v_y = std::sqrt(p.g * p.dy);
    double P_ud_bar = v_y * p.dt / p.dy;  // P_u = P_u_bar * (s_inv_bar/s)
    double P_lr_ref = p.alpha * v_y * p.dt / p.dx / p.dx;
    const double delta_nu_limit_h = p.delta_limit;
    // Horizontal threshold is Eq. (356):
    //   Δν_h = ν_c / ( (Δy/Δx)(1/μ) + 1 )
    // The vertical equivalent (swap Δx and Δy) is:
    //   Δν_v = ν_c / ( (Δx/Δy)μ + 1 ) = ν_c - Δν_h
    double inverse_nm = 1.0 / nm;
    const double delta_nu_limit_v = std::clamp(p.nu_cs - delta_nu_limit_h, 0.0, p.nu_cs);
    const double nu_surface_limit = std::min(p.nu_cs, delta_nu_limit_v + inverse_nm);

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
    
    // Reuse one shuffled traversal order per grid shape.
    std::shared_ptr<const std::vector<int>> shuffled_indices = get_cached_particle_iteration_order(nx, ny, nm);
    const int plane_stride = (ny - 1) * nm;

    for (int linear_index : *shuffled_indices) {
        int i = linear_index / plane_stride;
        int rem = linear_index % plane_stride;
        int j = (rem / nm) + 1;
        int k = rem % nm;
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

                double P_d = 0;
                if (down_void && space_down) {
                    if (space_criterion == SpaceCriterion::NuCs) {
                        P_d = P_ud_bar * std::pow(s_inv_bar[idx] / s_here, seg_exponent);
                    } else {
                        P_d = P_ud_bar; // P_ud_bar already accounts for pore size via has_space_for_particle
                    }
                }
                
                double nu_here = nu[idx];
                double nu_left = nu[idx_l];
                double nu_right = nu[idx_r];
                const double nu_up_here = get_above_density(nu, mask, i, j, ny, nu_here);
                const double nu_up_left = get_above_density(nu, mask, l, j_l, ny, nu_left);
                const double nu_up_right = get_above_density(nu, mask, r, j_r, ny, nu_right);

                const bool bulk_unstable_left = space_left;
                const bool bulk_unstable_right = space_right;

                bool unstable_left = lateral_unstable_with_surface_gate(
                    nu_here, nu_left, nu_up_here, nu_up_left, delta_nu_limit_h, nu_surface_limit, bulk_unstable_left);
                bool unstable_right = lateral_unstable_with_surface_gate(
                    nu_here, nu_right, nu_up_here, nu_up_right, delta_nu_limit_h, nu_surface_limit, bulk_unstable_right);

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

                if (P_tot > 1.0) {
                    printf("Warning: At (i,j,k)=(%d,%d,%d), total move probability P_tot=%.3f > 1. Individual probabilities before normalization: P_d=%.3f, P_l=%.3f, P_r=%.3f. Normalizing probabilities.\n",
                           i, j, k, P_tot, P_d, P_l, P_r);
                    double inv = 1.0 / P_tot;
                    P_d *= inv;
                    P_l *= inv;
                    P_r *= inv;
                    P_tot = 1.0;
                }

                if (p.inertia) {
                    u(i, j, k) -= P_l * dx_over_dt;
                    u(i, j, k) += P_r * dx_over_dt;
                    v(i, j, k) -= P_d * dy_over_dt;

                } else {

                    if (P_tot > 0.0) {
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

    // Normalize chi_out
    chi_out.resize(nx * ny);
    for (size_t i = 0; i < N_swap_arr.size(); i++) {
        chi_out[i] = N_swap_arr[i] / (nm * p.P_stab);
    }
}

void stream_core_lbm_zero_eq(View3<double> u,
                             View3<double> v,
                             View3<double> s,
                             const View2<const uint8_t>& mask,
                             std::vector<double>& nu,
                             const Params& p) {
    const int nx = p.nx;
    const int ny = p.ny;
    const int nm = p.nm;
    const double inverse_nm = 1.0 / static_cast<double>(nm);
    const double delta_nu_limit_h = p.delta_limit;
    const double delta_nu_limit_v = std::clamp(p.nu_cs - delta_nu_limit_h, 0.0, p.nu_cs);
    const double nu_surface_limit = std::min(p.nu_cs, delta_nu_limit_v + inverse_nm);
    const double beta_on_6 = p.beta / 6.0;
    const SpaceCriterion space_criterion = parse_space_criterion(p);

    // Precompute lateral neighbors once.
    NeighborIndices neighbors(nx, ny, p.cyclic_BC_y_offset, p.cyclic_BC);

    const double cfl_x = (p.dx > 0.0) ? (p.dt / p.dx) : 0.0;
    const double cfl_y = (p.dy > 0.0) ? (p.dt / p.dy) : 0.0;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> unit01(0.0, 1.0);

    // Reuse one shuffled traversal order per grid shape.
    std::shared_ptr<const std::vector<int>> shuffled_indices = get_cached_cell_iteration_order(nx, ny);
    const int row_stride = (ny - 1);
    std::vector<int> traversal_indices(*shuffled_indices);
    std::shuffle(traversal_indices.begin(), traversal_indices.end(), gen);

    // Enforce one-hop streaming per particle per timestep even with in-place updates.
    std::vector<uint8_t> streamed_this_step(static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nm), 0u);

    for (int linear_index : traversal_indices) {
        const int i = linear_index / row_stride;
        const int j = (linear_index % row_stride) + 1;
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

        if (solid_indices.empty()) {
            continue;
        }

        const int l = neighbors.left[idx];
        const int r = neighbors.right[idx];
        const int j_l = neighbors.j_left[idx];
        const int j_r = neighbors.j_right[idx];
        const int idx_l = neighbors.idx_left[idx];
        const int idx_r = neighbors.idx_right[idx];
        bool allow_left = true;
        bool allow_right = true;
        if (p.gate_inertia_lateral) {
            const double nu_here = nu[idx];
            const double nu_left = nu[idx_l];
            const double nu_right = nu[idx_r];
            const double nu_up_here = get_above_density(nu, mask, i, j, ny, nu_here);
            const double nu_up_left = get_above_density(nu, mask, l, j_l, ny, nu_left);
            const double nu_up_right = get_above_density(nu, mask, r, j_r, ny, nu_right);

            const bool bulk_unstable_left =
                (space_criterion == SpaceCriterion::NuCs) ? (nu_left < p.nu_cs) : true;
            const bool bulk_unstable_right =
                (space_criterion == SpaceCriterion::NuCs) ? (nu_right < p.nu_cs) : true;

            allow_left = lateral_unstable_with_surface_gate(
                nu_here, nu_left, nu_up_here, nu_up_left, delta_nu_limit_h, nu_surface_limit, bulk_unstable_left);
            allow_right = lateral_unstable_with_surface_gate(
                nu_here, nu_right, nu_up_here, nu_up_right, delta_nu_limit_h, nu_surface_limit, bulk_unstable_right);
        }

        std::shuffle(solid_indices.begin(), solid_indices.end(), gen);
        for (int k : solid_indices) {
            if (std::isnan(s(i, j, k))) {
                continue;
            }
            const int src_slot = (i * ny + j) * nm + k;
            if (streamed_this_step[src_slot] != 0) {
                continue;
            }

            double P_u = 0.0;
            double P_d = 0.0;
            double P_l = 0.0;
            double P_r = 0.0;

            const double u_here = u(i, j, k);
            const double v_here = v(i, j, k);

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

            if (j >= ny - 1 || mask(i, j + 1)) {
                P_u = 0.0;
            }
            if (j <= 0 || mask(i, j - 1)) {
                P_d = 0.0;
            }
            if (mask(l, j_l) || !allow_left) {
                P_l = 0.0;
            }
            if (mask(r, j_r) || !allow_right) {
                P_r = 0.0;
            }

            double P_tot = P_u + P_d + P_l + P_r;
            if (P_tot <= 0.0) {
                u(i,j,k) = 0.0;
                v(i,j,k) = 0.0;
                continue;
            }
            if (P_tot > 1.0) {
                const double inv = 1.0 / P_tot;
                P_u *= inv;
                P_d *= inv;
                P_l *= inv;
                P_r *= inv;
                // printf("Warning: At (i,j,k)=(%d,%d,%d), total move probability P_tot=%.3f > 1. Individual probabilities before normalization: P_u=%.3f, P_d=%.3f, P_l=%.3f, P_r=%.3f. Normalizing probabilities.\n",
                //        i, j, k, P_tot, P_u, P_d, P_l, P_r);
            }

            int dest_i = i;
            int dest_j = j;
            const double rand_val = unit01(gen);
            if (rand_val < P_u) {
                dest_j = j + 1; // up
            } else if (rand_val < P_u + P_d) {
                dest_j = j - 1; // down
            } else if (rand_val < P_u + P_d + P_l) {
                dest_i = l;
                dest_j = j_l; // left
            } else if (rand_val < P_u + P_d + P_l + P_r) {
                dest_i = r;
                dest_j = j_r; // right
            } else {
                continue; // No movement
            }

            const int dest_idx = dest_i * ny + dest_j;
            const int dest_slot = (dest_i * ny + dest_j) * nm + k;

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
            streamed_this_step[dest_slot] = 1;

            nu[idx] -= inverse_nm;
            nu[dest_idx] += inverse_nm;
        }
    }

    // Relax particle velocities toward equilibrium (u=v=0) using the selected
    // space criterion. Dilute/gas-like states are left unrelaxed.
    const bool relax_to_zero = (p.tau > 0.0);
    const double relax_factor = relax_to_zero ? std::exp(-p.dt / p.tau) : 1.0;
    
    if (relax_to_zero) {
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                int idx = i * ny + j;
                for (int k = 0; k < nm; ++k) {
                    if (should_relax_particle(
                            s, i, j, k, nu, idx, p.nu_cs, beta_on_6, nm, space_criterion)) {
                        u(i, j, k) *= relax_factor;
                        v(i, j, k) *= relax_factor;
                    }
                }
            }
        }
    }
}
