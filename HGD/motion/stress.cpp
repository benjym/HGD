#include "stress.h"
#include <cmath>
#include <algorithm>
#include <iostream>

// Helper function to compute top surface (index of topmost solid in each column)
std::vector<int> get_top_core(const std::vector<double>& nu, int nx, int ny, double nu_lim) {
    std::vector<int> top(nx, 0);
    
    for (int i = 0; i < nx; i++) {
        for (int j = ny - 1; j >= 0; j--) {
            if (nu[i * ny + j] > nu_lim) {
                top[i] = j;
                break;
            }
        }
    }
    
    return top;
}

// Helper function to compute depth field
std::vector<double> get_depth_core(const std::vector<double>& nu, int nx, int ny, const Params& p) {
    std::vector<int> top = get_top_core(nu, nx, ny);
    std::vector<double> depth(nx * ny, 0.0);
    
    for (int i = 0; i < nx; i++) {
        // Use p.y array like Python: depth[i, :] = p.y[top[i]] - p.y
        double y_top = p.y[top[i]];
        for (int j = 0; j < ny; j++) {
            depth[i * ny + j] = y_top - p.y[j];
        }
    }
    
    return depth;
}

// Harr method stress calculation with substeps
StressResult harr_substep_core(const View3<const double>& s,
                               const Params& p) {
    
    int nx = p.nx;
    int ny = p.ny;
    
    StressResult sigma(nx, ny);
    
    double D_0 = 0.1; // Set default D_0 value
    std::vector<double> K(nx * ny, D_0);
    
    // Weight of one cell
    // Note: In Python this was p.solid_density * p.dx * p.g * p.dy
    // But we need to check if solid_density exists in Params struct
    double weight_of_one_cell = p.solid_density * p.dx * p.g * p.dy;
    
    // Compute solid fraction
    std::vector<double> nu = compute_solid_fraction_core(s);
    
    // Get depth field
    std::vector<double> depth = get_depth_core(nu, nx, ny, p);
    
    // Compute D = K * depth
    std::vector<double> D(nx * ny);
    for (int i = 0; i < nx * ny; i++) {
        D[i] = K[i] * depth[i];
    }
    
    // Get cached neighbor indices (no y-offset for stress calculation)
    const NeighborIndices& neighbours = get_cached_neighbours(nx, ny, 0, p.cyclic_BC);
    
    // Loop from top to bottom (j from ny-2 down to 0)
    for (int j = ny - 2; j >= 0; j--) {
        // Weight per unit length for this row
        std::vector<double> this_weight_per_unit_length(nx);
        for (int i = 0; i < nx; i++) {
            this_weight_per_unit_length[i] = nu[i * ny + j] * weight_of_one_cell / p.dx;
        }
        
        // Calculate number of substeps based on CFL condition
        double max_D = 0.0;
        for (int i = 0; i < nx; i++) {
            max_D = std::max(max_D, D[i * ny + j]);
        }
        int nsubsteps = static_cast<int>(std::ceil(max_D * p.dy / (0.5 * p.dx * p.dx)));
        nsubsteps = std::max(nsubsteps, 1);  // At least one substep
        
        // Temporary storage for incremental sigma
        std::vector<double> sigma_inc(nx, 0.0);
        
        // Substep iteration
        for (int m = 0; m < nsubsteps; m++) {
            std::vector<double> up(nx);
            
            if (m == 0) {
                // Use sigma from row above
                for (int i = 0; i < nx; i++) {
                    up[i] = sigma.sigma_yy[i * ny + (j + 1)];
                }
            } else {
                // Use incremental sigma from previous substep
                up = sigma_inc;
            }
            
            // Apply diffusion operator using cached neighbor indices
            for (int i = 0; i < nx; i++) {
                int idx = i * ny + j;
                int idx_left = neighbours.idx_left[idx];
                int idx_right = neighbours.idx_right[idx];
                int i_left = neighbours.left[idx];
                int i_right = neighbours.right[idx];
                
                double D_ij = D[idx];
                // right_up[i] = D[i-1, j] * up[i-1] (from roll +1)
                double right_up = D[i_left * ny + j] * up[i_left];
                // left_up[i] = D[i+1, j] * up[i+1] (from roll -1)
                double left_up = D[i_right * ny + j] * up[i_right];
                
                sigma_inc[i] = this_weight_per_unit_length[i] / nsubsteps
                             + up[i]
                             + (p.dy / nsubsteps) / (p.dx * p.dx) 
                               * (left_up - 2.0 * up[i] * D_ij + right_up);
                
                // Zero out stress in void regions
                if (nu[idx] == 0.0) {
                    sigma_inc[i] = 0.0;
                }
            }
        }
        
        // Store final sigma_yy for this row
        for (int i = 0; i < nx; i++) {
            sigma.sigma_yy[i * ny + j] = sigma_inc[i];
        }
    }
    
    // Compute sigma_xy and sigma_xx using gradients
    // sigma_xy = - d/dx (K * Depth * sigma_yy)
    // sigma_xx = - d/dy (K * Depth * sigma_yy)
    // Using np.gradient-style differences: forward at start, backward at end, central elsewhere
    
    std::vector<double> Depth = get_depth_core(nu, nx, ny, p);
    
    // First compute K * Depth * sigma_yy for all points
    std::vector<double> D_sigma(nx * ny);
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            int idx = i * ny + j;
            D_sigma[idx] = K[idx] * Depth[idx] * sigma.sigma_yy[idx];
        }
    }
    
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            int idx = i * ny + j;
            
            // Gradient in x-direction using cached neighbor indices (central difference)
            int idx_left = neighbours.idx_left[idx];
            int idx_right = neighbours.idx_right[idx];
            double d_Dsigma_dx = (D_sigma[idx_right] - D_sigma[idx_left]) / (2.0 * p.dx);
            
            // Gradient in y-direction using np.gradient style:
            // j=0: forward difference
            // j=ny-1: backward difference
            // else: central difference
            double d_Dsigma_dy = 0.0;
            if (j == 0) {
                // Forward difference at bottom boundary
                d_Dsigma_dy = (D_sigma[i * ny + 1] - D_sigma[i * ny + 0]) / p.dy;
            } else if (j == ny - 1) {
                // Backward difference at top boundary
                d_Dsigma_dy = (D_sigma[i * ny + (ny - 1)] - D_sigma[i * ny + (ny - 2)]) / p.dy;
            } else {
                // Central difference
                d_Dsigma_dy = (D_sigma[i * ny + (j + 1)] - D_sigma[i * ny + (j - 1)]) / (2.0 * p.dy);
            }
            
            sigma.sigma_xy[idx] = -d_Dsigma_dx;
            sigma.sigma_xx[idx] = -d_Dsigma_dy;
        }
    }
    
    return sigma;
}

// Check if stress state exceeds Mohr-Coulomb failure criterion
std::vector<bool> check_mohr_coulomb_core(const StressResult& sigma,
                                             const Params& p) {
    int nx = p.nx;
    int ny = p.ny;
    std::vector<bool> failure_flag(nx * ny, false);
    
    // Convert repose angle from degrees to radians
    double phi = p.repose_angle * M_PI / 180.0;
    double sin_phi = std::sin(phi);
    double cohesion = p.cohesion;
    double N_phi = (1.0 + sin_phi) / (1.0 - sin_phi);
    double two_c_sqrt_Nphi = 2.0 * cohesion * std::sqrt(N_phi);
    
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            int idx = i * ny + j;
            
            double sigma_xx = sigma.sigma_xx[idx];
            double sigma_yy = sigma.sigma_yy[idx];
            double sigma_xy = sigma.sigma_xy[idx];
            
            // Calculate principal stresses using Mohr's circle
            double sigma_mean = (sigma_xx + sigma_yy) / 2.0;
            double radius = std::sqrt(std::pow((sigma_yy - sigma_xx) / 2.0, 2) + std::pow(sigma_xy, 2));
            
            double sigma_1 = sigma_mean + radius;  // Maximum principal stress
            double sigma_3 = sigma_mean - radius;  // Minimum principal stress
            
            // Mohr-Coulomb failure criterion:
            // sigma_1 = sigma_3 * (1 + sin(phi)) / (1 - sin(phi)) + 2*c*cos(phi) / (1 - sin(phi))
            // Rearranging: sigma_1 - sigma_3 * Nφ - 2*c*sqrt(Nφ) >= 0 indicates failure
            // where Nφ = (1 + sin(phi)) / (1 - sin(phi))
            
            
            // Check if stress state exceeds failure criterion
            if (sigma_1 - sigma_3 * N_phi >= two_c_sqrt_Nphi) {
                failure_flag[idx] = true;
            }

            // debug output
            // printf("Cell (%d,%d): sigma_xx=%.3f, sigma_yy=%.3f, sigma_1=%.3f, sigma_3=%.3f, strength=%.3f, failure=%d\n",
            //        i, j, sigma_xx, sigma_yy, sigma_1, sigma_3, strength, failure_flag[idx] ? 1 : 0);
        }
    }
    
    return failure_flag;
}
