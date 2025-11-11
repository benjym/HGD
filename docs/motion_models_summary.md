# Motion Models Analysis - Executive Summary

This is a quick reference summary of the comprehensive analysis in `motion_models_analysis.pdf`.

## Quick Comparison Table

| Implementation | Speed | Inertia | Status | Use Case |
|---------------|-------|---------|--------|----------|
| d2q4_slow.py | 1× | ❌ | Active | Reference/Learning |
| d2q4_array.py | 10-50× | ❌ | Active | Fast Python |
| d2q4_array_v2.py | 10-30× | ✅ Full | Active | Research/Advanced |
| d2q4_SA_array.py | 5-20× | ❌ | Experimental | Alternative approach |
| core.cpp | 50-200× | ⚠️ Partial | Production | Fast simulations |
| d2q4.cpp.dep | 40-150× | ✅ | Deprecated | Historical |

## Key Findings

### Inertia Implementation Status

**Complete Inertia** (✅):
- `d2q4_array_v2.py` - Most advanced Python implementation with full momentum transfer
  - Velocities stored per particle: `u[i,j,k]`, `v[i,j,k]`
  - Velocities swap with particles during motion
  - Contribution to probabilities: `P += (dt/dy) * v_dest`
  - Velocity of new void set to zero

**Partial Inertia** (⚠️):
- `core.cpp` - Has inertia parameter but not fully used in `stream_core`
  - `move_voids_core` updates velocities
  - `stream_core` uses mean velocities only
  - Missing: velocity swap, probabilistic motion with inertia

**No Inertia** (❌):
- `d2q4_slow.py`, `d2q4_array.py`, `d2q4_SA_array.py`

### Physics Models Supported

| Model | Stress | Multi-length | Slope Stability |
|-------|--------|--------------|-----------------|
| d2q4_slow.py | Partial | No | Gradient |
| d2q4_array.py | Yes | No | Gradient |
| d2q4_array_v2.py | Yes | Yes | Gradient/Stress |
| d2q4_SA_array.py | No | No | Angle-based |
| core.cpp | No | No | Gradient |

### Performance Factors

**Fast (50-200×)**:
- C++ compilation with -O3 -march=native
- Efficient memory layout (flattened arrays)
- No Python overhead
- **But**: Single-threaded, no SIMD

**Medium (10-50×)**:
- NumPy vectorization (BLAS/LAPACK)
- Array operations instead of loops
- Python overhead remains

**Slow (1×)**:
- Nested Python loops
- Individual element access
- Good for debugging only

## Recommendations for stream_core

### 🏆 Recommended: Option 1 - Minimal Upgrade

**Goal**: Add essential inertia while maintaining performance

**Changes Needed**:

1. **Add velocity fields to function signature**:
```cpp
void stream_core(
    const std::vector<double>& u_mean,
    const std::vector<double>& v_mean,
    View3<double> u,  // NEW: full velocity field
    View3<double> v,  // NEW: full velocity field
    View3<double> s,
    const View2<const uint8_t>& mask,
    std::vector<double>& nu,
    const Params& p);
```

2. **Include inertia in probability calculation**:
```cpp
double P_u = mask(i, j + 1) ? 
    (v_mean[idx_up] + (p.inertia ? v(i, j+1, k) : 0.0)) * dy_over_dt : 0;
```

3. **Swap velocities with particles**:
```cpp
if (p.inertia && swap_occurred) {
    // Swap velocities
    std::swap(u(i, j, k), u(dest[0], dest[1], k));
    std::swap(v(i, j, k), v(dest[0], dest[1], k));
    // Zero void velocity
    u(i, j, k) = 0.0;
    v(i, j, k) = 0.0;
}
```

4. **Make swaps probabilistic when inertia enabled**:
- Non-inertial: Keep `N = floor(nu * P)` deterministic swaps
- Inertial: Use probabilistic like `move_voids_core`

**Benefits**:
- ✅ Maintains C++ performance
- ✅ Adds critical missing feature
- ✅ Simple to implement and test
- ✅ Compatible with existing code

**Validation**:
- Compare with `d2q4_array_v2.py` for simple test cases
- Verify mass conservation
- Verify momentum conservation (when inertia enabled)
- Run standard benchmarks (collapse, segregation)

### Alternative Options

**Option 2**: Full feature parity with d2q4_array_v2.py
- ➕ Most accurate physics
- ➕ Enables advanced research
- ➖ Complex, high memory, longer development

**Option 3**: Hybrid with compile-time switching
- ➕ Optimal performance for each mode
- ➕ Future-proof flexibility
- ➖ More complex code structure

## When to Use Each Implementation

### For Understanding Physics
→ Use `d2q4_slow.py`
- Clear, readable code
- Easy to experiment
- Good for learning

### For Python Research
→ Use `d2q4_array_v2.py`
- Full feature set
- Inertia support
- Stress models
- Validated for research

### For Production Simulations
→ Use `core.cpp`
- Fastest performance
- C++ efficiency
- **After adding inertia per recommendations**

### For Validation
→ Compare `d2q4_array_v2.py` and `core.cpp`
- Python as reference
- C++ for speed
- Cross-validate results

## Key Physics Equations

### Upward Motion
```
P_u = (v_y * dt/dy) * (s_inv_bar_dest / s_dest)
```
With inertia: `+ (dt/dy) * v_dest`

### Lateral Motion
```
P_l/r = alpha * v_y * (dt/dx²) * s_neighbor
```
With inertia: `+ (dt/dy) * u_dest`

### Characteristic Velocity
- Average size: `v_y = sqrt(g * s_bar)`
- Freefall: `v_y = sqrt(2 * g * dy)`
- Stress: `v_y = sqrt(2 * pressure / rho)`

### Slope Stability
```
nu_neighbor - nu_current > delta_limit
```

## Testing Checklist

After implementing changes to `stream_core`:

- [ ] Unit test: probability calculation
- [ ] Unit test: velocity swap
- [ ] Integration test: mass conservation
- [ ] Integration test: momentum conservation (inertia mode)
- [ ] Benchmark: collapse test
- [ ] Benchmark: segregation test
- [ ] Comparison: match d2q4_array_v2.py results
- [ ] Performance: no significant regression
- [ ] Edge cases: boundaries, masked cells
- [ ] Edge cases: fully packed regions

## References

1. Full analysis: `motion_models_analysis.pdf` (16 pages)
2. LaTeX source: `motion_models_analysis.tex`
3. Code files in: `HGD/motion/`
4. Main documentation: See `MOTION_MODELS_README.md`

## Questions?

For detailed physics derivations, implementation notes, and mathematical notation, see the full PDF document `motion_models_analysis.pdf`.
