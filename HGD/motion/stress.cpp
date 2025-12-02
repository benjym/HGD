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
        double y_top = top[i] * p.dy;  // y coordinate is j * dy
        for (int j = 0; j < ny; j++) {
            depth[i * ny + j] = y_top - j * p.dy;
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
    
    // Set default D_0 value (equivalent to hasattr check in Python)
    double D_0 = 0.1;  // 1/10 in Python
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
            
            // Apply diffusion operator with periodic boundary conditions
            for (int i = 0; i < nx; i++) {
                int left_idx = (i == 0) ? (nx - 1) : (i - 1);
                int right_idx = (i == nx - 1) ? 0 : (i + 1);
                
                double D_ij = D[i * ny + j];
                double right_up = D_ij * up[right_idx];
                double left_up = D_ij * up[left_idx];
                
                sigma_inc[i] = this_weight_per_unit_length[i] / nsubsteps
                             + up[i]
                             + (p.dy / nsubsteps) / (p.dx * p.dx) 
                               * (left_up - 2.0 * up[i] * D_ij + right_up);
                
                // Zero out stress in void regions
                if (nu[i * ny + j] == 0.0) {
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
    
    std::vector<double> Depth = get_depth_core(nu, nx, ny, p);
    
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            int idx = i * ny + j;
            double K_val = K[idx];
            double D_sigma = K_val * Depth[idx] * sigma.sigma_yy[idx];
            
            // Gradient in x-direction (with periodic BC)
            int i_right = (i == nx - 1) ? 0 : (i + 1);
            int i_left = (i == 0) ? (nx - 1) : (i - 1);
            double D_sigma_right = K[i_right * ny + j] * Depth[i_right * ny + j] 
                                  * sigma.sigma_yy[i_right * ny + j];
            double D_sigma_left = K[i_left * ny + j] * Depth[i_left * ny + j] 
                                 * sigma.sigma_yy[i_left * ny + j];
            double d_Dsigma_dx = (D_sigma_right - D_sigma_left) / (2.0 * p.dx);
            
            // Gradient in y-direction
            double d_Dsigma_dy = 0.0;
            if (j > 0 && j < ny - 1) {
                double D_sigma_up = K[i * ny + (j + 1)] * Depth[i * ny + (j + 1)] 
                                   * sigma.sigma_yy[i * ny + (j + 1)];
                double D_sigma_down = K[i * ny + (j - 1)] * Depth[i * ny + (j - 1)] 
                                     * sigma.sigma_yy[i * ny + (j - 1)];
                d_Dsigma_dy = (D_sigma_up - D_sigma_down) / (2.0 * p.dy);
            }
            
            sigma.sigma_xy[idx] = -d_Dsigma_dx;
            sigma.sigma_xx[idx] = -d_Dsigma_dy;  // Negative because y is decreasing
        }
    }
    
    return sigma;
}
