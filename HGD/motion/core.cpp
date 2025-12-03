#include "core.h"
#include "stress.h"
#include <random>
#include <cmath>
#include <algorithm>
#include <array>

// NeighborIndices constructor implementation
NeighborIndices::NeighborIndices(int nx, int ny, int cyclic_BC_y_offset, bool cyclic_BC) 
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

// Cached singleton accessor for NeighborIndices
const NeighborIndices& get_cached_neighbours(int nx, int ny, int cyclic_BC_y_offset, bool cyclic_BC) {
    static int cached_nx = -1;
    static int cached_ny = -1;
    static int cached_offset = -1;
    static bool cached_cyclic = false;
    static NeighborIndices* cached_neighbours = nullptr;
    
    if (cached_neighbours == nullptr || 
        cached_nx != nx || cached_ny != ny || 
        cached_offset != cyclic_BC_y_offset || cached_cyclic != cyclic_BC) {
        
        delete cached_neighbours;
        cached_neighbours = new NeighborIndices(nx, ny, cyclic_BC_y_offset, cyclic_BC);
        cached_nx = nx;
        cached_ny = ny;
        cached_offset = cyclic_BC_y_offset;
        cached_cyclic = cyclic_BC;
    }
    
    return *cached_neighbours;
}

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

// Cache structure for shuffled indices
struct ShuffledIndicesCache {
    std::vector<int> i_cache;
    std::vector<int> j_cache;
    std::vector<int> k_cache;
    std::vector<int> idx_cache;
    std::vector<size_t> shuffle_indices;
    int cached_nx = -1;
    int cached_ny = -1;
    int cached_nm = -1;
    
    void initialize(int nx, int ny, int nm, std::mt19937& gen) {
        // Only regenerate if dimensions changed
        if (cached_nx == nx && cached_ny == ny && cached_nm == nm) {
            // Just reshuffle the existing indices
            // std::shuffle(shuffle_indices.begin(), shuffle_indices.end(), gen);
            return;
        }
        
        // Dimensions changed, regenerate everything
        cached_nx = nx;
        cached_ny = ny;
        cached_nm = nm;
        
        size_t total_size = nx * (ny - 1) * nm;
        i_cache.clear();
        j_cache.clear();
        k_cache.clear();
        idx_cache.clear();
        i_cache.reserve(total_size);
        j_cache.reserve(total_size);
        k_cache.reserve(total_size);
        idx_cache.reserve(total_size);
        
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
        
        shuffle_indices.resize(total_size);
        for (size_t i = 0; i < total_size; i++) {
            shuffle_indices[i] = i;
        }
        std::shuffle(shuffle_indices.begin(), shuffle_indices.end(), gen);
    }
};

void move_voids_core(View3<double> u, View3<double> v, View3<double> s,
                     const View2<const uint8_t>& mask,
                     Params p,
                     std::vector<double>& nu,
                     std::vector<double>& chi_out) {
    
    int nx = p.nx, ny = p.ny, nm = p.nm;
    double seg_exponent = p.seg_exponent;
    
    // Precompute neighbor indices once
    NeighborIndices neighbours(nx, ny, p.cyclic_BC_y_offset, p.cyclic_BC);
    
    // Precompute useful quantities
    std::vector<double> s_bar = compute_mean_core(View3<const double>{s.data, nx, ny, nm, s.sx, s.sy, s.sz});
    std::vector<double> s_inv_bar = compute_s_inv_bar_core(View3<const double>{s.data, nx, ny, nm, s.sx, s.sy, s.sz});
    std::vector<bool> some_particles = compute_some_particles_core(nu, mask, nx, ny);

    // FROM PYTHON VERSION:
    // if p.advection_model == "freefall":
    //     u_here = U_dest = np.sqrt(p.g * p.dy)
    // elif p.advection_model == "stress":
    //     sigma = stress.calculate_stress(s, last_swap, p)
    //     pressure = np.abs(
    //         stress.get_pressure(sigma, p)
    //     )  # HACK: PRESSURE SHOULD BE POSITIVE BUT I HAVE ISSUES WITH THE STRESS MODEL
    //     u_here = np.sqrt(2 * pressure / p.solid_density)
    //     U = np.repeat(u_here[:, :, np.newaxis], p.nm, axis=2)
    //     U_dest = np.roll(
    //         U, d, axis=axis
    //     )  # NEED TO TAKE DESTINATION VALUE

    double v_y = std::sqrt(p.g * p.dy);
    double v_x = v_y;  // assuming isotropic velocity fluctuations
    double P_u_bar = v_y * p.dt / p.dy;  // P_u = P_u_bar * (s_inv_bar/s)
    double P_lr_ref = p.alpha * v_y * p.dt / p.dx / p.dx;
    double delta_nu_limit = p.delta_limit;

    double inverse_nm = 1.0 / nm;
    double dy_over_dt = p.dy / p.dt;
    double dx_over_dt = p.dx / p.dt;

    bool unstable_left, unstable_right;

    std::vector<double> v_y_vec(nx * ny, 0.0);
    std::vector<bool> unstable;
    
    if (p.advection_model == "stress") {
        StressResult sigma = harr_substep_core(View3<const double>{s.data, nx, ny, nm, s.sx, s.sy, s.sz}, p);
        
        for (int idx = 0; idx < nx * ny; idx++) {
            double pressure = 0.5 * (sigma.sigma_xx[idx] + sigma.sigma_yy[idx]);
            v_y_vec[idx] = std::sqrt(2.0 * std::abs(pressure) / p.solid_density);
        }

        unstable = check_mohr_coulomb_core(
            sigma,
            p
        );
    }

    // storage arrays
    std::array<int, 3> dest = {0, 0, 0};
    std::vector<double> N_swap_arr(nx * ny, 0.0);

    // Thread-safe PRNG
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> rand_dist(0.0, 1.0);
    
    // Cache shuffled indices across calls
    static ShuffledIndicesCache cache;
    cache.initialize(nx, ny, nm, gen);

    for (size_t idx_pos = 0; idx_pos < cache.shuffle_indices.size(); idx_pos++) {
        size_t index = cache.shuffle_indices[idx_pos];
        int i = cache.i_cache[index];
        int j = cache.j_cache[index];
        int k = cache.k_cache[index];
        int idx = cache.idx_cache[index];
        int idx_up = idx + 1;  // i * ny + j + 1

        if (some_particles[idx]) {
            if (std::isnan(s(i, j, k))) {
                if (nu[idx] < p.nu_cs) {
                    if (mask(i, j)) {
                        continue;
                    }

                    // if (p.advection_model == "stress") {
                    //     v_y = v_y_vec[idx];
                    //     P_u_bar = v_y * p.dt / p.dy;
                    //     P_lr_ref = p.alpha * v_y * p.dt / p.dx / p.dx;
                    //     printf("i=%d, j=%d, v_y=%.4f, P_u_bar=%.4f, P_lr_ref=%.4f\n", i, j, v_y, P_u_bar, P_lr_ref);
                    // }

                    double P_u = std::isnan(s(i, j + 1, k)) ? 0 : P_u_bar * std::pow(s_inv_bar[idx_up] / s(i, j + 1, k), seg_exponent);

                    // Use precomputed neighbor indices
                    int l = neighbours.left[idx];
                    int r = neighbours.right[idx];
                    int j_l = neighbours.j_left[idx];
                    int j_r = neighbours.j_right[idx];
                    int idx_l = neighbours.idx_left[idx];
                    int idx_r = neighbours.idx_right[idx];
                    
                    double nu_here = nu[idx];
                    double nu_left = nu[idx_l];
                    double nu_right = nu[idx_r];
                    if (p.advection_model == "stress") {
                        unstable_left = unstable[idx_l];
                        unstable_right = unstable[idx_r];
                    }
                    else {
                        unstable_left = nu_left - nu_here > delta_nu_limit;
                        unstable_right = nu_right - nu_here > delta_nu_limit;
                    }
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
                        int dest_idx = -1;

                        if (rand_val < P_u && P_u > 0) {
                            dest = {i, j + 1, k};
                            dest_idx = idx_up;
                            found = true;
                            // Particle moves DOWN from (i,j+1) to (i,j), velocity stored at new location (i,j)
                            u(i, j, k) = u(dest[0], dest[1], k);
                            v(i, j, k) = v(dest[0], dest[1], k) - dy_over_dt;
                            u(dest[0], dest[1], k) = 0.0; // Reset velocity at void
                            v(dest[0], dest[1], k) = 0.0; // Reset velocity at void
                        }
                        else if (rand_val < (P_l + P_u)) {
                            dest = {l, j_l, k};
                            dest_idx = idx_l;
                            found = true;
                            // Particle moves RIGHT from (l,j_l) to (i,j), velocity stored at new location (i,j)
                            u(i, j, k) = u(dest[0], dest[1], k) + dx_over_dt;
                            v(i, j, k) = v(dest[0], dest[1], k);
                            u(dest[0], dest[1], k) = 0.0; // Reset velocity at void
                            v(dest[0], dest[1], k) = 0.0; // Reset velocity at void
                        }
                        else if (rand_val < P_tot) {
                            dest = {r, j_r, k};
                            dest_idx = idx_r;
                            found = true;
                            // Particle moves LEFT from (r,j_r) to (i,j), velocity stored at new location (i,j)
                            u(i, j, k) = u(dest[0], dest[1], k) - dx_over_dt;
                            v(i, j, k) = v(dest[0], dest[1], k);
                            u(dest[0], dest[1], k) = 0.0; // Reset velocity at void
                            v(dest[0], dest[1], k) = 0.0; // Reset velocity at void
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

void stream_core(View3<double> u, View3<double> v, 
                 View3<double> s,
                 const View2<const uint8_t>& mask,
                 std::vector<double>& nu,
                 const Params& p) {
    
    int nx = p.nx, ny = p.ny, nm = p.nm;
    double inverse_nm = 1.0 / nm;
    double dt_over_dy = p.dt / p.dy;
    double dt_over_dx = p.dt / p.dx;

    // printf("Streaming step started\n");
    
    // Precompute neighbor indices once
    NeighborIndices neighbours(nx, ny, p.cyclic_BC_y_offset, p.cyclic_BC);

    // Thread-safe PRNG
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> rand_dist(0.0, 1.0);

    static ShuffledIndicesCache cache;
    cache.initialize(nx, ny, nm, gen);

    for (size_t idx_pos = 0; idx_pos < cache.shuffle_indices.size(); idx_pos++) {
        size_t index = cache.shuffle_indices[idx_pos];
        int i = cache.i_cache[index];
        int j = cache.j_cache[index];
        if (j == 0) continue; // Skip bottom row
        if (j >= ny - 1) continue; // Skip top row
        int k = cache.k_cache[index];
        int idx = i * ny + j;
        int idx_up = idx + 1;
        int idx_down = idx - 1;
        
        // Use precomputed neighbor indices
        int l = neighbours.left[idx];
        int r = neighbours.right[idx];
        int j_l = neighbours.j_left[idx];
        int j_r = neighbours.j_right[idx];
        int idx_l = neighbours.idx_left[idx];
        int idx_r = neighbours.idx_right[idx];

        std::array<int, 2> dest = {i, j}; // Default to current position
        int dest_idx;

        for (int k = 0; k < nm; k++) {
            if (!std::isnan(s(i, j, k))) {
                // Use the velocity at the CURRENT particle location, not neighbor locations
                double v_current = v(i, j, k);
                double u_current = u(i, j, k);
                
                // Particle moves up if it has positive vertical velocity
                double P_u = (mask(i, j + 1) || v_current <= 0) ? 0 : v_current * dt_over_dy;
                // Particle moves down if it has negative vertical velocity
                double P_d = (mask(i, j - 1) || v_current >= 0) ? 0 : -v_current * dt_over_dy;
                // Particle moves left if it has negative horizontal velocity
                double P_l = (mask(l, j_l) || u_current >= 0) ? 0 : -u_current * dt_over_dx;
                // Particle moves right if it has positive horizontal velocity
                double P_r = (mask(r, j_r) || u_current <= 0) ? 0 : u_current * dt_over_dx;
                
                double P_tot = P_u + P_d + P_l + P_r;
                if (P_tot > 1.0) {
                    printf("WARNING: Cell (%d,%d): P_u=%.3f, P_d=%.3f, P_l=%.3f, P_r=%.3f, P_tot=%.3f\n", i, j, P_u, P_d, P_l, P_r, P_tot);
                }

                bool moved = false;
                double rand_val = static_cast<double>(rand()) / RAND_MAX;                  
                
                if (rand_val < P_u && P_u > 0) {
                    dest = {i, j + 1};
                    dest_idx = idx_up;
                    moved = true;
                    // printf("Particle moves UP from (%d,%d,%d) to (%d,%d,%d)\n", i, j + 1, k, i, j, k);
                }
                else if (rand_val < (P_u + P_d)) {
                    dest = {i, j - 1};
                    dest_idx = idx_down;
                    moved = true;
                    // printf("Particle moves DOWN from (%d,%d,%d) to (%d,%d,%d)\n", i, j - 1, k, i, j, k);
                }
                else if (rand_val < (P_u + P_d + P_l)) {
                    dest = {l, j_l};
                    dest_idx = idx_l;
                    moved = true;
                    // printf("Particle moves LEFT from (%d,%d,%d) to (%d,%d,%d)\n", l, j_l, k, i, j, k);
                }
                else if (rand_val < (P_u + P_d + P_l + P_r)) {
                    dest = {r, j_r};
                    dest_idx = idx_r;
                    moved = true;
                    // printf("Particle moves RIGHT from (%d,%d,%d) to (%d,%d,%d)\n", r, j_r, k, i, j, k);
                }

                if (moved) {
                    // printf("Streaming particle from (%d,%d,%d) to (%d,%d,%d)\n", i, j, k, dest[0], dest[1], k);
                    // check if the destination is not a solid
                    if (nu[dest_idx] < p.nu_cs) {

                        // printf("  Successful move\n");
                        double tmp = s(i, j, k);
                        s(i, j, k) = s(dest[0], dest[1], k);
                        s(dest[0], dest[1], k) = tmp;

                        double tmp_u = u(i, j, k);
                        double tmp_v = v(i, j, k);
                        u(i, j, k) = 0.;//u(dest[0], dest[1], k); // Reset velocity at void
                        v(i, j, k) = 0.;//v(dest[0], dest[1], k); // Reset velocity at void
                        u(dest[0], dest[1], k) = tmp_u; // Move velocity to new location
                        v(dest[0], dest[1], k) = tmp_v; // Move velocity to new location

                        nu[idx] += inverse_nm;
                        nu[dest_idx] -= inverse_nm;
                    }
                    else {
                        // printf("  Move blocked, destination solid\n");
                        // Destination is solid, break the while loop
                        break;
                    }
                }
            }
        }
    }
}

// void stream_core_dep(View3<double> u, View3<double> v, 
//                  View3<double> s,
//                  const View2<const uint8_t>& mask,
//                  std::vector<double>& nu,
//                  const Params& p) {
    
//     int nx = p.nx, ny = p.ny, nm = p.nm;
//     double inverse_nm = 1.0 / nm;
//     double dy_over_dt = p.dy / p.dt;
//     double dx_over_dt = p.dx / p.dt;

//     // printf("Streaming step started\n");
    
//     // Precompute neighbor indices once
//     NeighborIndices neighbours(nx, ny, p.cyclic_BC_y_offset, p.cyclic_BC);

//     for (int i = 0; i < nx; i++) {
//         for (int j = 0; j < ny - 1; j++) {  // Stop before ny-1 to avoid j+1 out of bounds
//             int idx = i * ny + j;
//             int idx_up = idx + 1;
            
//             // Use precomputed neighbor indices
//             int l = neighbours.left[idx];
//             int r = neighbours.right[idx];
//             int j_l = neighbours.j_left[idx];
//             int j_r = neighbours.j_right[idx];
//             int idx_l = neighbours.idx_left[idx];
//             int idx_r = neighbours.idx_right[idx];

//             double u_up = (v_mean[idx_up] > 0) ? v_mean[idx_up] : 0;
//             double P_u = mask(i, j + 1) ? 0 : u_up * dy_over_dt;
//             double u_down = (v_mean[idx] < 0) ? v_mean[idx] : 0;
//             double P_d = mask(i, j) ? 0 : -u_down * dy_over_dt;
//             double u_left = (u_mean[idx_l] < 0) ? u_mean[idx_l] : 0;
//             double P_l = mask(l, j_l) ? 0 : u_left * dx_over_dt;
//             double u_right = (u_mean[idx_r] > 0) ? u_mean[idx_r] : 0;
//             double P_r = mask(r, j_r) ? 0 : u_right * dx_over_dt;

//             int N_u = static_cast<int>(nu[idx] * P_u * nm);
//             int N_d = static_cast<int>(nu[idx] * P_d * nm);
//             int N_l = static_cast<int>(nu[idx] * P_l * nm);
//             int N_r = static_cast<int>(nu[idx] * P_r * nm);

//             // printf("Cell (%d,%d): N_u=%d, N_l=%d, N_r=%d\n", i, j, N_u, N_l, N_r);

//             std::vector<int> N_arr = {N_u, N_d, N_l, N_r};
//             std::vector<int> dests = {i, j + 1, i, j - 1, l, j_l, r, j_r};
//             int k = 0;
            
//             for (int d = 0; d < 4; d += 1) {
//                 std::array<int, 2> dest = {dests[2 * d], dests[2 * d + 1]};
//                 int dest_idx = dest[0] * ny + dest[1];
//                 int N_added = 0;
                
//                 while (N_added < N_arr[d] && k < nm) {
//                     // if this is a solid and the dest is a void
//                     if (!std::isnan(s(i, j, k)) && std::isnan(s(dest[0], dest[1], k))) {

//                         // printf("Streaming particle from (%d,%d,%d) to (%d,%d,%d)\n", i, j, k, dest[0], dest[1], k);
                        
//                         // check if the destination is not a solid
//                         if (nu[dest_idx] < p.nu_cs) {

//                             // printf("  Successful move\n");
//                             double tmp = s(i, j, k);
//                             s(i, j, k) = s(dest[0], dest[1], k);
//                             s(dest[0], dest[1], k) = tmp;

//                             u(dest[0], dest[1], k) = u(i, j, k); // Move velocity to new location
//                             v(dest[0], dest[1], k) = v(i, j, k); // Move velocity to new location
//                             u(i, j, k) = 0.0; // Reset velocity at void
//                             v(i, j, k) = 0.0; // Reset velocity at void
                            

//                             nu[idx] += inverse_nm;
//                             nu[dest_idx] -= inverse_nm;
                            
//                             N_added += 1;
//                         }
//                         else {
//                             // printf("  Move blocked, destination solid\n");
//                             // Destination is solid, break the while loop
//                             break;
//                         }
//                     }
//                     k += 1;
//                 }
//             }
//         }
//     }
// }
