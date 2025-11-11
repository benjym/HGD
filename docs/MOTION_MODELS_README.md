# Motion Models Analysis

This directory contains a comprehensive analysis of all motion model implementations in HGD.

## Files

- **motion_models_analysis.tex** - LaTeX source document analyzing all motion models
- **motion_models_analysis.pdf** - Compiled PDF document (16 pages)

## Overview

The document provides:

1. **Detailed analysis** of each motion model implementation:
   - `d2q4_slow.py` - Reference Python implementation
   - `d2q4_array.py` - Vectorized Python implementation
   - `d2q4_array_v2.py` - Advanced Python with inertia support
   - `d2q4_SA_array.py` - Alternative vectorized formulation
   - `core.cpp` - Production C++ implementation
   - `d2q4.cpp.dep` - Deprecated C++ implementation

2. **Physics characterization** including:
   - Mathematical formulation of void motion
   - Probability calculations for different directions
   - Slope stability criteria
   - Advection models (average_size, freefall, stress)

3. **Implementation features** including:
   - Inertia support (present/absent)
   - Parallelization strategies
   - Memory usage
   - Performance comparison

4. **Pros and cons** of each implementation

5. **Recommendations for stream_core** function:
   - Three options ranging from minimal upgrade to full feature parity
   - Specific implementation details for adding inertia support
   - Validation and testing strategies

## Key Findings

### Inertia Implementations

Only the following models have proper inertia support:
- ✅ `d2q4_array_v2.py` - Full inertia with momentum transfer
- ✅ `d2q4.cpp.dep` - Deprecated but had inertia support
- ⚠️ `core.cpp` - Parameter exists but not fully implemented in `stream_core`

### Performance

Relative speed (approximate):
- `d2q4_slow.py`: 1× (baseline)
- `d2q4_array.py`: 10-50×
- `d2q4_array_v2.py`: 10-30×
- `d2q4_SA_array.py`: 5-20×
- `core.cpp`: 50-200×

### Recommendations

**For the `stream_core` function in `core.cpp`:**

The recommended approach is **Option 1: Minimal Upgrade** which includes:
1. Add full velocity field support (`u[i,j,k]`, `v[i,j,k]`)
2. Include inertia contribution to swap probabilities
3. Swap velocities with particles when inertia is enabled
4. Consider probabilistic swaps in inertia mode
5. Validate against `d2q4_array_v2.py`

This maintains C++ performance while adding the critical inertia feature that's currently missing.

## Compiling the Document

To recompile the LaTeX document:

```bash
cd docs
pdflatex motion_models_analysis.tex
```

Requirements:
- `texlive-latex-base`
- `texlive-latex-extra`

On Ubuntu/Debian:
```bash
sudo apt-get install texlive-latex-base texlive-latex-extra
```

## Citation

If you use this analysis in your research, please cite the HGD package:

```
Marks, B., Athani, S., & Li, J. (2024). Heterarchical Granular Dynamics (HGD).
https://github.com/benjym/HGD
```

## Contributing

If you find errors or have suggestions for improving this analysis, please:
1. Open an issue on GitHub
2. Submit a pull request with corrections
3. Contact the authors directly

## License

This documentation is provided under the same license as the HGD package (see LICENSE.txt in the root directory).
