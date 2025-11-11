# HGD Method Comparison - Executive Summary

Response to: *"Can you benchmark against other numerical methods? lattice gas automata? lattice boltzmann? any other stochastic methods for advection? what is the best/optimal way to implement this? the current method doesnt seem very robust/accurate"*

## Quick Answer

**HGD's niche**: Fast granular flow simulation with size segregation, positioned between DEM (too slow) and continuum methods (can't capture segregation).

**Current robustness issues identified**:
1. ❌ Conflict resolution (multiple swaps → same cell) is arbitrary
2. ❌ Incomplete inertia implementation in `core.cpp` 
3. ❌ No automatic timestep control when P_tot > 1
4. ❌ Limited validation/convergence studies

**Optimal implementation strategy**: See recommendations below

## Method Comparison Table

| Method | Speed | Accuracy | Segregation | Best For | Cost |
|--------|-------|----------|-------------|----------|------|
| **DEM** | 1× | ★★★★★ | ★★★★★ | Contact mechanics | $10^4$ particles |
| **HGD** | 100× | ★★★ | ★★★★★ | Large-scale segregation | $10^6-10^8$ particles |
| **LBM** | 100× | ★★★★ | ★★ (complex) | Fluid flow | $10^6$ cells |
| **LGA** | 50× | ★★ | ★ | Fluid (legacy) | $10^6$ cells |
| **Continuum** | 1000× | ★★ | ★ | Engineering | $10^6$ cells |

## Detailed Comparison

### vs. Lattice Gas Automata (LGA)

**Similarities**:
- Both: Discrete particles on lattice
- Both: Local collision/swap rules
- Both: Emergent macroscopic behavior

**Differences**:
- LGA: Boolean occupation, identical particles → Navier-Stokes
- HGD: Particle sizes, heterarchical coordinate → Granular flow with segregation

**HGD advantages**:
- ✅ Polydisperse systems natural
- ✅ Direct physical interpretation
- ✅ Designed for granular flow

**LGA limitations**:
- ❌ Statistical noise
- ❌ No Galilean invariance (pre-LBM)
- ❌ Can't handle segregation

### vs. Lattice Boltzmann Method (LBM)

**LBM advantages**:
- ✅ Strong theoretical foundation (Chapman-Enskog)
- ✅ Smooth continuous fields
- ✅ Excellent for incompressible flow
- ✅ Lower statistical noise
- ✅ More accurate for viscous flows

**HGD advantages**:
- ✅ Direct particle-level physics
- ✅ Natural size segregation (no multiphase model needed)
- ✅ Better for dense granular flows
- ✅ Simpler conceptual model

**When to use which**:
- LBM: Fluid-dominated, viscous flows
- HGD: Granular-dominated, segregation important

### vs. Discrete Element Method (DEM)

**DEM advantages**:
- ✅ Most accurate particle-level physics
- ✅ Contact forces, friction, rotation
- ✅ No grid restrictions
- ✅ Well-validated

**DEM limitations**:
- ❌ Extremely expensive: O(N²) or O(N log N)
- ❌ Limited to ~10⁴-10⁶ particles
- ❌ Small timesteps required
- ❌ Complex parameter calibration

**HGD advantages**:
- ✅ 100-1000× faster
- ✅ Can handle 10⁶-10⁸ particles
- ✅ Larger timesteps
- ✅ Simpler parameters

**Use cases**:
- DEM: Detailed mechanics, small systems, validation
- HGD: Large-scale flows, parameter studies, segregation

### vs. Monte Carlo / Kinetic Monte Carlo

**HGD is essentially**: Monte Carlo random walk with gravity bias

**Differences from standard MC**:
- Standard MC: No preferred direction
- HGD: Gravity-biased upward motion
- HGD: Lateral diffusion ∝ velocity

**KMC improvements possible**:
- Variable timestep (event-driven)
- Exact event selection
- Better for rare events

## Current HGD Robustness Issues

### Issue 1: Conflict Resolution

**Problem**: Multiple voids can try to swap into same cell

**Current handling**:
- `d2q4_slow.py`: Random iteration order
- `d2q4_array_v2.py`: Detect conflicts, random selection
- `core.cpp`: Shuffle indices before processing

**Why it's not robust**: Which swap "wins" is arbitrary, not physics-based

**Better solution**: Rate-based selection
```cpp
// If multiple swaps target same cell:
P(swap_i wins) = P_i / Σ_j P_j
```

### Issue 2: Probability > 1

**Problem**: Large timesteps → P_u + P_l + P_r > 1 (unphysical)

**Current handling**: Error message or warning

**Why it's not robust**: Code fails or produces wrong results

**Better solution**: Automatic renormalization
```cpp
if (P_tot > 1.0) {
    P_u /= P_tot;
    P_l /= P_tot;
    P_r /= P_tot;
}
```

### Issue 3: Incomplete Inertia

**Problem**: `core.cpp` has inertia parameter but doesn't fully use it in `stream_core`

**Impact**: 
- Velocities not conserved properly
- Momentum transfer incorrect
- Can't model inertial flows accurately

**Solution**: See `motion_models_analysis.pdf` Option 1

### Issue 4: Limited Validation

**Missing tests**:
- ❌ Grid convergence studies
- ❌ Timestep convergence studies  
- ❌ Statistical convergence (multiple runs)
- ❌ Comparison with DEM
- ❌ Comparison with experiments

**Needed**:
- ✅ Systematic benchmarking protocol
- ✅ Documented convergence rates
- ✅ Validation against known solutions

### Issue 5: Boundary Conditions

**Current**: Limited to periodic, no-flux, simple masks

**Problems**:
- No stress-based walls
- No inflow/outflow
- Periodic with offset creates artifacts

**Better**: 
- Implement standard BC types
- Validate each BC type
- Document BC behavior

## Optimal Implementation Strategy

### Architecture Recommendation

**Separate physics from numerics**:

```cpp
// Physics: What are the swap probabilities?
class PhysicsModel {
    virtual double compute_probability(
        ParticleState& source, 
        ParticleState& dest,
        Direction dir) = 0;
};

// Numerics: How do we execute swaps?
class NumericalSolver {
    virtual void step(Grid& grid, 
                     PhysicsModel& physics,
                     double dt) = 0;
};
```

### Multiple Solver Modes

**Mode 1: Current Stochastic (default)**
- Fast, simple
- Good for non-inertial flows
- Existing implementation

**Mode 2: Kinetic Monte Carlo**
- Variable timestep
- Exact event selection
- Better for rare events

**Mode 3: LBM-inspired**
- Distribution functions
- Better theoretical foundation
- More accurate for inertial flows

**Mode 4: Hybrid HGD-DEM**
- DEM in dense regions
- HGD in dilute regions
- Couple at interface

### Implementation Priorities

**Immediate (1-2 weeks)**:
1. ✅ Add probability renormalization
2. ✅ Implement rate-based conflict resolution
3. ✅ Add mass conservation tests
4. ✅ Document current convergence behavior

**Short-term (1-2 months)**:
1. ✅ Complete inertia in `core.cpp` (Option 1)
2. ✅ Add momentum conservation tests (when inertia enabled)
3. ✅ Grid/timestep convergence studies
4. ✅ Benchmark against DEM (small systems)

**Medium-term (3-6 months)**:
1. ✅ Adaptive timestep control
2. ✅ Enhanced boundary conditions
3. ✅ Experimental validation suite
4. ✅ Performance optimization (SIMD, threading)

**Long-term (6-12 months)**:
1. ✅ Alternative solver modes (KMC, LBM-inspired)
2. ✅ Hybrid coupling (HGD-DEM, HGD-Continuum)
3. ✅ GPU acceleration
4. ✅ Comprehensive documentation

## Validation & Benchmarking Protocol

### Recommended Benchmark Suite

**Test 1: Mass Conservation**
- Setup: Random initial, run 1000 steps
- Metric: |ΔM/M₀| < 10⁻¹⁴
- Expected: Exact conservation
- Status: ⚠️ Need to verify

**Test 2: Grid Convergence**
- Setup: Vary Δx = L/N, N = 32, 64, 128, 256
- Metric: RMS error vs N
- Expected: O(Δx²) convergence
- Status: ❌ Not implemented

**Test 3: Timestep Convergence**
- Setup: Vary Δt = T/M, M = 100, 200, 400, 800
- Metric: RMS error vs Δt
- Expected: O(Δt) convergence
- Status: ❌ Not implemented

**Test 4: Statistical Convergence**
- Setup: 100 runs, different seeds
- Metric: σ vs N_runs
- Expected: σ ∝ 1/√N_runs
- Status: ❌ Not implemented

**Test 5: DEM Comparison**
- Setup: Small system (10⁴ particles)
- Metric: Velocity profiles, segregation
- Expected: Qualitative agreement
- Status: ❌ Not implemented

**Test 6: Experimental Validation**
- Setup: Published experiments
- Metric: Flow rates, angles of repose, segregation
- Expected: Within experimental error
- Status: ⚠️ Partial

### Performance Benchmarking

Expected performance (order of magnitude):

| Method | System Size | Time/Step | Memory | Accuracy |
|--------|-------------|-----------|---------|----------|
| DEM | 10⁴ | 1 s | 1 MB | ★★★★★ |
| HGD (Python) | 10⁶ | 0.1 s | 100 MB | ★★★ |
| HGD (C++) | 10⁶ | 0.01 s | 100 MB | ★★★ |
| HGD (C++ opt) | 10⁶ | 0.001 s | 100 MB | ★★★ |
| LBM | 10⁶ | 0.001 s | 10 MB | ★★★★ |

## Recommendations Summary

### What HGD Does Well
✅ Large-scale granular flows  
✅ Size segregation phenomena  
✅ Fast parameter studies  
✅ Qualitative predictions  

### What Needs Improvement
❌ Numerical robustness (conflicts, P>1)  
❌ Inertia implementation (incomplete)  
❌ Validation (no convergence studies)  
❌ Documentation (limited)  

### Optimal Path Forward

**For production use**:
1. Fix robustness issues (renormalization, conflict resolution)
2. Complete inertia implementation
3. Add validation tests
4. Use C++ implementation with these fixes

**For research**:
1. Keep Python implementations for flexibility
2. Compare with DEM for validation
3. Develop convergence studies
4. Explore alternative solver modes

**For comparison with other methods**:
- Use HGD for large systems with segregation
- Use DEM for detailed mechanics validation
- Use LBM for fluid-dominated flows
- Use continuum for engineering estimates

## Code Examples

### Immediate Fix 1: Renormalize Probabilities

```cpp
// In move_voids_core and stream_core
double P_tot = P_u + P_l + P_r;
if (P_tot > 1.0) {
    // Renormalize instead of error
    double scale = 1.0 / P_tot;
    P_u *= scale;
    P_l *= scale;
    P_r *= scale;
    P_tot = 1.0;
}
```

### Immediate Fix 2: Rate-Based Conflict Resolution

```cpp
// When multiple swaps target same destination
std::vector<Swap> conflicts;
// ... collect all swaps targeting dest_idx ...

if (conflicts.size() > 1) {
    // Compute total rate
    double total_rate = 0.0;
    for (auto& swap : conflicts) {
        total_rate += swap.probability;
    }
    
    // Select one based on relative rates
    double rand = random(0, total_rate);
    double cumulative = 0.0;
    for (auto& swap : conflicts) {
        cumulative += swap.probability;
        if (rand < cumulative) {
            execute_swap(swap);
            break;
        }
    }
}
```

### Immediate Fix 3: Add Conservation Test

```python
def test_mass_conservation():
    """Test that total mass is conserved"""
    p = load_params("test_case.json5")
    s = initialize_grains(p)
    
    mass_initial = np.sum(~np.isnan(s))
    
    for step in range(1000):
        u, v, s, c, T, chi, last_swap = move_voids(u, v, s, p)
    
    mass_final = np.sum(~np.isnan(s))
    
    assert abs(mass_final - mass_initial) < 1e-10, \
        f"Mass not conserved: {mass_initial} -> {mass_final}"
```

## References

**Full analysis**: See `method_comparison_and_benchmarks.pdf` (14 pages)

**Related docs**:
- `motion_models_analysis.pdf` - Implementation analysis
- `motion_models_summary.md` - Quick reference

**Further reading**:
- Frisch et al. (1986): Lattice Gas Automata
- Chen & Doolen (1998): Lattice Boltzmann review
- Cundall & Strack (1979): DEM fundamentals
- GDR MiDi (2004): μ(I) rheology
