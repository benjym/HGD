#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <omp.h>
#include <vector>
#include <cmath>
#include <iostream>

namespace py = pybind11;
using Eigen::MatrixXd;
using Eigen::VectorXd;

// Compute solid fraction using Eigen
// Compute solid fraction using Eigen
std::vector<double> compute_solid_fraction(py::array_t<double> s)
{
    auto s_buf = s.unchecked<3>(); 
    int nx = s.shape(0), ny = s.shape(1), nm = s.shape(2);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> nu(nx, ny);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            VectorXd col(nm);
            for (int k = 0; k < nm; k++)
                col(k) = s_buf(i, j, k);
            
            nu(i, j) = (col.array().isNaN()).count() / static_cast<double>(nm);
            nu(i,j) = 1.0 - nu(i,j);
        }
    }

    return std::vector<double>(nu.data(), nu.data() + nu.size());
}

// Compute inverse solid fraction average using Eigen
std::vector<double> compute_s_inv_bar(py::array_t<double> s)
{
    auto s_buf = s.unchecked<3>(); 
    int nx = s.shape(0), ny = s.shape(1), nm = s.shape(2);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> s_inv_bar(nx, ny);


    #pragma omp parallel for collapse(2)
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            VectorXd col(nm);
            for (int k = 0; k < nm; k++)
                col(k) = s_buf(i, j, k);

            VectorXd valid = col.array().isNaN().select(0, col);
            valid = (valid.array() > 0.0).select(valid.array().inverse(), 0.0);


            double sum = valid.sum();
            int count = (valid.array() > 0).count();

            s_inv_bar(i, j) = (count > 0) ? count / sum : 0.0;
        }
    }

    return std::vector<double>(s_inv_bar.data(), s_inv_bar.data() + s_inv_bar.size());
}

// Compute solid fraction mean using Eigen
std::vector<double> compute_s_bar(py::array_t<double> s)
{
    auto s_buf = s.unchecked<3>(); 
    int nx = s.shape(0), ny = s.shape(1), nm = s.shape(2);

    // MatrixXd s_bar = MatrixXd::Zero(nx, ny);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> s_bar(nx, ny);


    #pragma omp parallel for collapse(2)
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            VectorXd col(nm);
            for (int k = 0; k < nm; k++)
                col(k) = s_buf(i, j, k);

            VectorXd valid = col.array().isNaN().select(0, col);
            int count = (valid.array() > 0.0).count();

            s_bar(i, j) = (count > 0) ? valid.sum() / count : 0.0;
        }
    }

    return std::vector<double>(s_bar.data(), s_bar.data() + s_bar.size());
}

// Compute "some particles" mask using Eigen
std::vector<bool> compute_some_particles(const std::vector<double>& nu, py::array_t<bool> mask, int nx, int ny)
{
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> some_particles(nx, ny);
    some_particles.setConstant(true);
    auto mask_buf = mask.unchecked<2>(); // Unchecked view of the mask array

    #pragma omp parallel for collapse(2)
    for (int i = 1; i < nx - 1; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            // if (mask_buf(i, j)) // Access mask using the buffer
            // {
            //     some_particles(i, j) = false;
            //     continue;
            // }
            // else
            // {
                // Check if the particle is solid
                some_particles(i, j) = (nu[i * ny + j] + 
                                        nu[(i - 1) * ny + j] + 
                                        nu[(i + 1) * ny + j] + 
                                        nu[i * ny + j + 1]) > 0.0;
            // }
        }
    }

    return std::vector<bool>(some_particles.data(), some_particles.data() + some_particles.size());
}

// Compute "locally fluid" mask using Eigen
std::vector<bool> compute_locally_fluid(const std::vector<double>& nu, int nx, int ny, double nu_cs)
{
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> locally_fluid(nx, ny);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            locally_fluid(i, j) = nu[i * ny + j] < nu_cs;
        }
    }

    return std::vector<bool>(locally_fluid.data(), locally_fluid.data() + locally_fluid.size());
}