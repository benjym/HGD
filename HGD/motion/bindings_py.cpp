#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "core.h"
#include "stress.h"

namespace py = pybind11;

static View3<double> as_view3(py::array_t<double>& a) {
    auto buf = a.mutable_unchecked<3>();
    return { 
        buf.mutable_data(0, 0, 0), 
        static_cast<int>(buf.shape(0)), 
        static_cast<int>(buf.shape(1)), 
        static_cast<int>(buf.shape(2)),
        static_cast<int>(a.strides(0) / sizeof(double)),
        static_cast<int>(a.strides(1) / sizeof(double)),
        static_cast<int>(a.strides(2) / sizeof(double))
    };
}

static View3<const double> as_view3_const(const py::array_t<double>& a) {
    auto buf = a.unchecked<3>();
    return { 
        const_cast<double*>(buf.data(0, 0, 0)), 
        static_cast<int>(buf.shape(0)), 
        static_cast<int>(buf.shape(1)), 
        static_cast<int>(buf.shape(2)),
        static_cast<int>(a.strides(0) / sizeof(double)),
        static_cast<int>(a.strides(1) / sizeof(double)),
        static_cast<int>(a.strides(2) / sizeof(double))
    };
}

static View2<const uint8_t> as_view2u8(const py::array_t<bool>& m) {
    auto b = m.unchecked<2>();
    return { 
        reinterpret_cast<uint8_t*>(const_cast<bool*>(b.data(0, 0))), 
        static_cast<int>(b.shape(0)), 
        static_cast<int>(b.shape(1)),
        static_cast<int>(m.strides(0) / sizeof(bool)),
        static_cast<int>(m.strides(1) / sizeof(bool))
    };
}

py::array_t<double> compute_solid_fraction_py(py::array_t<double> s) {
    auto vs = as_view3_const(s);
    auto nu = compute_solid_fraction_core(vs);
    return py::array_t<double>({vs.nx, vs.ny}, nu.data());
}

py::array_t<double> compute_mean_py(py::array_t<double> a) {
    auto va = as_view3_const(a);
    auto mean_vals = compute_mean_core(va);
    return py::array_t<double>({va.nx, va.ny}, mean_vals.data());
}

py::array_t<double> compute_s_inv_bar_py(py::array_t<double> s) {
    auto vs = as_view3_const(s);
    auto s_inv_bar = compute_s_inv_bar_core(vs);
    return py::array_t<double>({vs.nx, vs.ny}, s_inv_bar.data());
}

py::array_t<bool> compute_some_particles_py(py::array_t<double> nu_array, py::array_t<bool> mask) {
    auto nu_buf = nu_array.unchecked<2>();
    auto mask_view = as_view2u8(mask);
    
    std::vector<double> nu(nu_buf.shape(0) * nu_buf.shape(1));
    for (int i = 0; i < nu_buf.shape(0); i++) {
        for (int j = 0; j < nu_buf.shape(1); j++) {
            nu[i * nu_buf.shape(1) + j] = nu_buf(i, j);
        }
    }
    
    auto some_particles = compute_some_particles_core(nu, mask_view, mask_view.nx, mask_view.ny);
    
    // Convert std::vector<bool> to regular bool array for numpy
    std::vector<bool> result_bool(some_particles.begin(), some_particles.end());
    py::array_t<bool> result = py::array_t<bool>({mask_view.nx, mask_view.ny});
    auto result_buf = result.mutable_unchecked<2>();
    for (int i = 0; i < mask_view.nx; i++) {
        for (int j = 0; j < mask_view.ny; j++) {
            result_buf(i, j) = some_particles[i * mask_view.ny + j];
        }
    }
    return result;
}

py::array_t<bool> compute_locally_fluid_py(py::array_t<double> nu_array, double nu_cs) {
    auto nu_buf = nu_array.unchecked<2>();
    
    std::vector<double> nu(nu_buf.shape(0) * nu_buf.shape(1));
    for (int i = 0; i < nu_buf.shape(0); i++) {
        for (int j = 0; j < nu_buf.shape(1); j++) {
            nu[i * nu_buf.shape(1) + j] = nu_buf(i, j);
        }
    }
    
    auto locally_fluid = compute_locally_fluid_core(nu, nu_buf.shape(0), nu_buf.shape(1), nu_cs);
    
    // Convert std::vector<bool> to numpy array manually
    py::array_t<bool> result = py::array_t<bool>({static_cast<int>(nu_buf.shape(0)), static_cast<int>(nu_buf.shape(1))});
    auto result_buf = result.mutable_unchecked<2>();
    for (int i = 0; i < nu_buf.shape(0); i++) {
        for (int j = 0; j < nu_buf.shape(1); j++) {
            result_buf(i, j) = locally_fluid[i * nu_buf.shape(1) + j];
        }
    }
    return result;
}

py::tuple move_voids_py(py::array_t<double> u, py::array_t<double> v, py::array_t<double> s,
                        py::object p, int dummy,
                        py::object c, py::object T,
                        py::object chi, py::object last_swap) {
    
    // Extract parameters from Python object
    Params P{
        p.attr("g").cast<double>(),
        p.attr("dt").cast<double>(),
        p.attr("dx").cast<double>(),
        p.attr("dy").cast<double>(),
        p.attr("alpha").cast<double>(),
        p.attr("nu_cs").cast<double>(),
        p.attr("P_stab").cast<double>(),
        p.attr("delta_limit").cast<double>(),
        p.attr("solid_density").cast<double>(),
        p.attr("seg_exponent").cast<double>(),
        p.attr("cohesion").cast<double>(),
        p.attr("repose_angle").cast<double>(),
        p.attr("cyclic_BC").cast<bool>(),
        p.attr("inertia").cast<bool>(),
        p.attr("cyclic_BC_y_offset").cast<int>(),
        p.attr("nx").cast<int>(),
        p.attr("ny").cast<int>(),
        p.attr("nm").cast<int>(),
        p.attr("y").cast<std::vector<double>>(),
        p.attr("advection_model").cast<std::string>()
    };
    
    auto vU = as_view3(u);
    auto vV = as_view3(v);
    auto vS = as_view3(s);
    
    auto mask = p.attr("boundary_mask").cast<py::array_t<bool>>();
    auto vM = as_view2u8(mask);

    std::vector<double> nu = compute_solid_fraction_core(View3<const double>{vS.data, vS.nx, vS.ny, vS.nm, vS.sx, vS.sy, vS.sz});
    std::vector<double> chi_out;
    
    move_voids_core(vU, vV, vS, vM, P, nu, chi_out);

    // Return chi as a new NumPy array (nx, ny)
    py::array_t<double> chi_py({P.nx, P.ny}, chi_out.data());
    return py::make_tuple(u, v, s, c, T, chi_py, last_swap);
}

py::tuple stream_py(py::array_t<double> u, py::array_t<double> v, py::array_t<double> s,
                    py::object p) {
    
    // Extract parameters from Python object
    Params P{
        p.attr("g").cast<double>(),
        p.attr("dt").cast<double>(),
        p.attr("dx").cast<double>(),
        p.attr("dy").cast<double>(),
        p.attr("alpha").cast<double>(),
        p.attr("nu_cs").cast<double>(),
        p.attr("P_stab").cast<double>(),
        p.attr("delta_limit").cast<double>(),
        p.attr("solid_density").cast<double>(),
        p.attr("seg_exponent").cast<double>(),
        p.attr("cohesion").cast<double>(),
        p.attr("repose_angle").cast<double>(),
        p.attr("cyclic_BC").cast<bool>(),
        p.attr("inertia").cast<bool>(),
        p.attr("cyclic_BC_y_offset").cast<int>(),
        p.attr("nx").cast<int>(),
        p.attr("ny").cast<int>(),
        p.attr("nm").cast<int>(),
        p.attr("y").cast<std::vector<double>>(),
        p.attr("advection_model").cast<std::string>()
    };
    
    auto vU = as_view3(u);
    auto vV = as_view3(v);
    auto vS = as_view3(s);
    
    auto mask = p.attr("boundary_mask").cast<py::array_t<bool>>();
    auto vM = as_view2u8(mask);

    // Compute solid fraction and mean velocities
    std::vector<double> nu = compute_solid_fraction_core(View3<const double>{vS.data, vS.nx, vS.ny, vS.nm, vS.sx, vS.sy, vS.sz});
    // std::vector<double> u_mean = compute_mean_core(View3<const double>{vU.data, vU.nx, vU.ny, vU.nm, vU.sx, vU.sy, vU.sz});
    // std::vector<double> v_mean = compute_mean_core(View3<const double>{vV.data, vV.nx, vV.ny, vV.nm, vV.sx, vV.sy, vV.sz});
    
    stream_core(vU, vV, vS, vM, nu, P);
    
    return py::make_tuple(u, v, s);
}

py::array_t<double> harr_substep_py(py::array_t<double> s, py::object p) {
    // Extract parameters from Python object
    Params P{
        p.attr("g").cast<double>(),
        p.attr("dt").cast<double>(),
        p.attr("dx").cast<double>(),
        p.attr("dy").cast<double>(),
        p.attr("alpha").cast<double>(),
        p.attr("nu_cs").cast<double>(),
        p.attr("P_stab").cast<double>(),
        p.attr("delta_limit").cast<double>(),
        p.attr("solid_density").cast<double>(),
        p.attr("seg_exponent").cast<double>(),
        p.attr("cohesion").cast<double>(),
        p.attr("repose_angle").cast<double>(),
        p.attr("cyclic_BC").cast<bool>(),
        p.attr("inertia").cast<bool>(),
        p.attr("cyclic_BC_y_offset").cast<int>(),
        p.attr("nx").cast<int>(),
        p.attr("ny").cast<int>(),
        p.attr("nm").cast<int>(),
        p.attr("y").cast<std::vector<double>>(),
        p.attr("advection_model").cast<std::string>()
    };
    
    auto vS = as_view3_const(s);
    
    // Call the C++ stress calculation
    StressResult sigma = harr_substep_core(vS, P);
    
    // Create output arrays
    py::array_t<double> sigma_out({P.nx, P.ny, 3});
    auto sigma_buf = sigma_out.mutable_unchecked<3>();
    
    for (int i = 0; i < P.nx; i++) {
        for (int j = 0; j < P.ny; j++) {
            int idx = i * P.ny + j;
            sigma_buf(i, j, 0) = sigma.sigma_xy[idx];
            sigma_buf(i, j, 1) = sigma.sigma_yy[idx];
            sigma_buf(i, j, 2) = sigma.sigma_xx[idx];
        }
    }
    
    return sigma_out;
}

py::array_t<bool> check_mohr_coulomb_py(py::array_t<double> sigma_array, py::object p) {
    // Extract parameters from Python object
    Params P{
        p.attr("g").cast<double>(),
        p.attr("dt").cast<double>(),
        p.attr("dx").cast<double>(),
        p.attr("dy").cast<double>(),
        p.attr("alpha").cast<double>(),
        p.attr("nu_cs").cast<double>(),
        p.attr("P_stab").cast<double>(),
        p.attr("delta_limit").cast<double>(),
        p.attr("solid_density").cast<double>(),
        p.attr("seg_exponent").cast<double>(),
        p.attr("cohesion").cast<double>(),
        p.attr("repose_angle").cast<double>(),
        p.attr("cyclic_BC").cast<bool>(),
        p.attr("inertia").cast<bool>(),
        p.attr("cyclic_BC_y_offset").cast<int>(),
        p.attr("nx").cast<int>(),
        p.attr("ny").cast<int>(),
        p.attr("nm").cast<int>(),
        p.attr("y").cast<std::vector<double>>(),
        p.attr("advection_model").cast<std::string>()
    };
    
    // Extract stress tensor from input array (nx, ny, 3)
    auto sigma_buf = sigma_array.unchecked<3>();
    
    StressResult sigma(P.nx, P.ny);
    for (int i = 0; i < P.nx; i++) {
        for (int j = 0; j < P.ny; j++) {
            int idx = i * P.ny + j;
            sigma.sigma_xy[idx] = sigma_buf(i, j, 0);
            sigma.sigma_yy[idx] = sigma_buf(i, j, 1);
            sigma.sigma_xx[idx] = sigma_buf(i, j, 2);
        }
    }
    
    // Call the C++ Mohr-Coulomb check
    std::vector<bool> failure = check_mohr_coulomb_core(sigma, P);
    
    // Create output array (nx, ny)
    py::array_t<double> failure_out({P.nx, P.ny});
    auto failure_buf = failure_out.mutable_unchecked<2>();
    
    for (int i = 0; i < P.nx; i++) {
        for (int j = 0; j < P.ny; j++) {
            failure_buf(i, j) = failure[i * P.ny + j];
        }
    }
    
    return failure_out;
}

PYBIND11_MODULE(d2q4_cpp, m) {
    m.def("compute_solid_fraction", &compute_solid_fraction_py, "Compute solid fraction");
    m.def("compute_s_inv_bar", &compute_s_inv_bar_py, "Compute inverse solid fraction");
    m.def("compute_mean", &compute_mean_py, "Compute mean over last dimension of a 3D array");
    m.def("compute_some_particles", &compute_some_particles_py, "Compute some particles mask");
    m.def("compute_locally_fluid", &compute_locally_fluid_py, "Compute locally fluid mask");
    m.def("move_voids", &move_voids_py, "Moves voids in the system",
          py::arg("u"), py::arg("v"), py::arg("s"), py::arg("p"), py::arg("dummy"),
          py::arg("c") = py::none(), py::arg("T") = py::none(),
          py::arg("chi") = py::none(), py::arg("last_swap") = py::none());
    m.def("stream", &stream_py, "Stream mass based on inertia",
          py::arg("u"), py::arg("v"), py::arg("s"), py::arg("p"));
    m.def("harr_substep", &harr_substep_py, "Calculate stress using Harr method with substeps",
          py::arg("s"), py::arg("p"));
    m.def("check_mohr_coulomb", &check_mohr_coulomb_py, "Check if stress state exceeds Mohr-Coulomb failure criterion",
          py::arg("sigma"), py::arg("p"));
}
