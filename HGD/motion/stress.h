#pragma once
#include "core.h"
#include <vector>

// Stress calculation result structure
struct StressResult {
    std::vector<double> sigma_xy;  // nx * ny
    std::vector<double> sigma_yy;  // nx * ny
    std::vector<double> sigma_xx;  // nx * ny
    
    StressResult(int nx, int ny) 
        : sigma_xy(nx * ny, 0.0),
          sigma_yy(nx * ny, 0.0),
          sigma_xx(nx * ny, 0.0) {}
};

// Helper functions for stress calculations
std::vector<int> get_top_core(const std::vector<double>& nu, int nx, int ny, double nu_lim = 0.0);
std::vector<double> get_depth_core(const std::vector<double>& nu, int nx, int ny,const Params& p);

// Main stress calculation function using Harr method with substeps
StressResult harr_substep_core(const View3<const double>& s,
                               const Params& p);
