#pragma once
#include <cstddef>
#include <vector>
#include <random>
#include <cmath>
#include <tuple>

struct Params {
    double g, dt, dx, dy, alpha, nu_cs, P_stab, delta_limit;
    double solid_density, seg_exponent;
    double cohesion, repose_angle;
    bool cyclic_BC;
    bool inertia;
    int cyclic_BC_y_offset;
    int nx, ny, nm;
    std::vector<double> y;
    std::string advection_model;
};

template<typename T>
struct View3 {
    T* data;           // contiguous buffer of size nx*ny*nm
    int nx, ny, nm;
    int sx, sy, sz;    // strides in elements (typically sy=nm, sx=ny*nm, sz=1)
    
    inline T& operator()(int i, int j, int k) { 
        return data[i*sx + j*sy + k*sz]; 
    }
    
    inline const T& operator()(int i, int j, int k) const { 
        return data[i*sx + j*sy + k*sz]; 
    }
};

template<typename T>
struct View2 {
    T* data; 
    int nx, ny; 
    int sx, sy;
    
    inline T& operator()(int i, int j) { 
        return data[i*sx + j*sy]; 
    }
    
    inline const T& operator()(int i, int j) const { 
        return data[i*sx + j*sy]; 
    }
};

// Helper struct to store precomputed neighbor indices
struct NeighborIndices {
    std::vector<int> left;      // left neighbor i-index for each (i,j)
    std::vector<int> right;     // right neighbor i-index for each (i,j)
    std::vector<int> j_left;    // j-index when accessing left neighbor
    std::vector<int> j_right;   // j-index when accessing right neighbor
    std::vector<int> idx_left;  // flattened index for left neighbor
    std::vector<int> idx_right; // flattened index for right neighbor
    
    NeighborIndices(int nx, int ny, int cyclic_BC_y_offset, bool cyclic_BC);
};

// Cached singleton accessor for NeighborIndices
const NeighborIndices& get_cached_neighbours(int nx, int ny, int cyclic_BC_y_offset, bool cyclic_BC);

// Core function declarations
std::vector<double> compute_solid_fraction_core(const View3<const double>& s);
std::vector<double> compute_mean_core(const View3<const double>& a);
std::vector<double> compute_s_inv_bar_core(const View3<const double>& s);
std::vector<bool> compute_some_particles_core(const std::vector<double>& nu,
                                             const View2<const uint8_t>& mask, int nx, int ny);
std::vector<bool> compute_locally_fluid_core(const std::vector<double>& nu, int nx, int ny, double nu_cs);

std::tuple<int, int, int, int> get_lr_core(int i, int j, int nx, int ny, int cyclic_BC_y_offset, bool cyclic_BC);

void move_voids_core(View3<double> u, View3<double> v, View3<double> s,
                     const View2<const uint8_t>& mask,
                     Params p,
                     std::vector<double>& nu,
                     std::vector<double>& chi_out);

void stream_core(View3<double> u, View3<double> v, View3<double> s,
                 const View2<const uint8_t>& mask,
                 std::vector<double>& nu,
                 const Params& p);
