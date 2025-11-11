# HGD Motion Models: Comprehensive Documentation Index

This directory contains complete documentation analyzing all motion model implementations in the Heterarchical Granular Dynamics package.

## Quick Navigation

### For Understanding Motion Models
→ Start with: **motion_models_summary.md**

### For Method Comparison
→ Start with: **method_comparison_summary.md**

### For Complete Technical Details
→ Read: **motion_models_analysis.pdf** (16 pages)
→ Read: **method_comparison_and_benchmarks.pdf** (14 pages)

## Document Summary

### 1. Motion Models Analysis

**Files:**
- `motion_models_analysis.pdf` (16 pages)
- `motion_models_analysis.tex` (LaTeX source)
- `motion_models_summary.md` (Quick reference)
- `MOTION_MODELS_README.md` (Guide)

**What's covered:**
- Analysis of 6 implementations (d2q4_slow.py, d2q4_array.py, d2q4_array_v2.py, d2q4_SA_array.py, core.cpp, d2q4.cpp.dep)
- Physics equations for each model
- Inertia implementation details
- Performance comparison (1× to 200×)
- Recommendations for improving stream_core

**Key finding:** Only d2q4_array_v2.py has complete inertia. core.cpp needs upgrade.

### 2. Method Comparison

**Files:**
- `method_comparison_and_benchmarks.pdf` (14 pages)
- `method_comparison_and_benchmarks.tex` (LaTeX source)
- `method_comparison_summary.md` (Quick reference)

**What's covered:**
- HGD vs Lattice Gas Automata (LGA)
- HGD vs Lattice Boltzmann Method (LBM)
- HGD vs Discrete Element Method (DEM)
- HGD vs Monte Carlo / Kinetic Monte Carlo
- HGD vs Continuum methods
- Robustness and accuracy issues
- Optimal implementation strategies
- Validation and benchmarking protocols

**Key finding:** HGD fills niche between DEM (too slow) and continuum (can't capture segregation).

### 3. Test Suite

**Files:**
- `../test/test_robustness.py` (New validation tests)

**What's tested:**
- Mass conservation
- Probability limits
- Statistical convergence
- Numerical accuracy
- Boundary conditions
- Edge cases

## Quick Comparison Tables

### Implementation Comparison

| Implementation | Speed | Inertia | Memory | Best For |
|---------------|-------|---------|--------|----------|
| d2q4_slow.py | 1× | No | Low | Learning/Reference |
| d2q4_array.py | 10-50× | No | Medium | Fast Python |
| d2q4_array_v2.py | 10-30× | **Yes** | High | Research/Advanced |
| d2q4_SA_array.py | 5-20× | No | High | Experimental |
| core.cpp | **50-200×** | Partial | Low | **Production** |
| d2q4.cpp.dep | 40-150× | Yes | Medium | Deprecated |

### Method Comparison

| Method | Speed vs DEM | Particles | Segregation | Accuracy |
|--------|--------------|-----------|-------------|----------|
| **DEM** | 1× | 10⁴ | ★★★★★ | ★★★★★ |
| **HGD** | **100×** | **10⁶-10⁸** | **★★★★★** | **★★★** |
| **LBM** | 100× | 10⁶ | ★★ | ★★★★ |
| **LGA** | 50× | 10⁶ | ★ | ★★ |
| **Continuum** | 1000× | 10⁶ | ★ | ★★ |

## Current Issues & Solutions

### Issue 1: Conflict Resolution (Not Robust)

**Problem:** Multiple voids swap to same cell, winner is arbitrary

**Current code:**
```cpp
// Random shuffle, first one wins
std::shuffle(indices.begin(), indices.end(), gen);
```

**Better solution:**
```cpp
// Rate-based selection
P(swap_i wins) = P_i / Σ_j P_j
```

**Status:** Code example provided, not yet implemented

### Issue 2: Probability > 1 (Not Accurate)

**Problem:** Large timesteps → P_tot > 1 (unphysical)

**Current code:**
```cpp
// Just errors or warns
if (P_tot > 1) throw std::runtime_error("P_tot > 1");
```

**Better solution:**
```cpp
// Automatic renormalization
if (P_tot > 1.0) {
    double scale = 1.0 / P_tot;
    P_u *= scale; P_l *= scale; P_r *= scale;
}
```

**Status:** Code example provided, not yet implemented

### Issue 3: Incomplete Inertia (Not Accurate)

**Problem:** core.cpp has inertia parameter but doesn't use it fully

**Solution:** Add per-particle velocities, swap with particles

See: `motion_models_analysis.pdf` Section 7.3.1 (Option 1)

**Status:** Detailed recommendations provided, not yet implemented

### Issue 4: No Validation (Not Robust)

**Problem:** No systematic convergence studies

**Solution:** Test suite created in `test/test_robustness.py`

**Tests include:**
- Mass conservation (should be exact)
- Probability limits (0 ≤ P ≤ 1)
- Statistical convergence (σ ∝ 1/√N)
- Grid/timestep convergence
- Boundary conditions

**Status:** ✅ Test suite implemented

## Recommendations

### Immediate Priorities

1. **Add probability renormalization** (1 day)
   - Prevents crashes with large timesteps
   - Makes code more robust

2. **Implement rate-based conflict resolution** (2-3 days)
   - More physically consistent
   - Better accuracy

3. **Add validation tests** (1 day)
   - Mass conservation checks
   - Run existing test_robustness.py

4. **Document convergence behavior** (2-3 days)
   - Grid refinement studies
   - Timestep refinement studies

### Short-term Goals (1-2 months)

1. **Complete inertia in core.cpp** (1-2 weeks)
   - Follow Option 1 from motion_models_analysis.pdf
   - Add per-particle velocities
   - Implement velocity swapping
   - Add momentum conservation tests

2. **Benchmark against DEM** (1 week)
   - Small system comparison
   - Validate segregation patterns
   - Document agreement/differences

3. **Experimental validation** (2-3 weeks)
   - Compare with published data
   - Document accuracy metrics
   - Identify parameter calibration needs

### Long-term Goals (6-12 months)

1. **Alternative solver modes**
   - Kinetic Monte Carlo variant
   - LBM-inspired variant
   - Template-based architecture

2. **Hybrid coupling**
   - HGD-DEM coupling for accuracy
   - HGD-Continuum for scale

3. **GPU acceleration**
   - CUDA/HIP implementation
   - 10-100× additional speedup

## Usage Guidelines

### When to Use Each Implementation

**For learning the method:**
→ Use `d2q4_slow.py`
- Clear, readable code
- Easy to modify
- Good for understanding physics

**For Python research:**
→ Use `d2q4_array_v2.py`
- Most complete physics
- Inertia support
- Stress models
- Good for validation

**For production simulations:**
→ Use `core.cpp`
- Fastest (50-200× baseline)
- C++ efficiency
- After adding fixes above

**For validation:**
→ Compare `d2q4_array_v2.py` and `core.cpp`
- Python as reference
- C++ for speed
- Cross-validate results

### When to Use HGD vs Other Methods

**Use HGD when:**
- Large systems (>10⁶ particles)
- Segregation is important
- Parameter studies needed
- Quasi-static/slow flows

**Use DEM when:**
- Detailed contact mechanics needed
- Small systems (<10⁵ particles)
- Validation/calibration
- Accurate force resolution required

**Use LBM when:**
- Fluid-dominated flows
- Viscous effects important
- Single-phase incompressible flow
- Good theoretical foundation needed

**Use Continuum when:**
- Engineering estimates
- Very large systems
- Uniform flows
- Fast results needed

## Compiling Documentation

### LaTeX Documents

Requirements:
```bash
sudo apt-get install texlive-latex-base texlive-latex-extra
```

Compile:
```bash
cd docs
pdflatex motion_models_analysis.tex
pdflatex method_comparison_and_benchmarks.tex
```

### Running Tests

```bash
cd test
python -m unittest test_robustness -v
```

## Contributing

If you implement any of the recommended fixes:

1. Update the relevant documentation
2. Add tests to `test/test_robustness.py`
3. Update this README with status
4. Document any changes in behavior

## Citation

If you use this analysis in research:

```bibtex
@misc{hgd_analysis_2024,
  title={Heterarchical Granular Dynamics: Motion Models Analysis and Method Comparison},
  author={Analysis of HGD Package},
  year={2024},
  howpublished={GitHub Repository Documentation},
  url={https://github.com/benjym/HGD}
}
```

## Contact

For questions about this analysis:
- Open an issue on GitHub
- See main README.md for author contact information

## Version History

- **2024-11-11**: Initial comprehensive analysis
  - Motion models characterization (16 pages)
  - Method comparison with LGA/LBM/DEM (14 pages)
  - Robustness test suite
  - Implementation recommendations

## Related Files

```
docs/
├── motion_models_analysis.pdf          # 16-page technical analysis
├── motion_models_analysis.tex          # LaTeX source
├── motion_models_summary.md            # Quick reference
├── MOTION_MODELS_README.md             # Guide to motion models docs
├── method_comparison_and_benchmarks.pdf # 14-page comparison
├── method_comparison_and_benchmarks.tex # LaTeX source
├── method_comparison_summary.md         # Quick comparison reference
└── README_COMPREHENSIVE.md             # This file

test/
└── test_robustness.py                  # Validation test suite
```

## Quick Links

- [Main Repository README](../README.md)
- [Motion Models Summary](motion_models_summary.md)
- [Method Comparison Summary](method_comparison_summary.md)
- [Robustness Tests](../test/test_robustness.py)

---

**Total Documentation**: ~30 pages (PDFs) + 4 markdown summaries + test suite

**Estimated Reading Time**: 
- Quick overview: 15 minutes (this file + summaries)
- Complete analysis: 2-3 hours (all PDFs)
- Implementation: 1-2 weeks (fixes) to 6-12 months (full upgrade)
