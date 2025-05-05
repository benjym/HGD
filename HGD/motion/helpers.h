#ifndef HELPERS_H
#define HELPERS_H

#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <cstdio>

namespace py = pybind11;
using Eigen::MatrixXd;
using Eigen::VectorXd;

// Function declarations
std::vector<double> compute_solid_fraction(py::array_t<double> s);
std::vector<double> compute_s_inv_bar(py::array_t<double> s);
std::vector<double> compute_mean(py::array_t<double> input);
std::vector<bool> compute_some_particles(const std::vector<double>& nu, py::array_t<bool> mask_buf, int nx, int ny);
std::vector<bool> compute_locally_fluid(const std::vector<double>& nu, int nx, int ny, double nu_cs);
std::tuple<int, int, int, int> get_lr(int i, int j, int nx, int ny, int cyclic_BC_y_offset, bool cyclic_BC);

#endif
