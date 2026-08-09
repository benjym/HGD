# HGD Example Simulations

This document describes the example simulation configurations included with HGD. Each example demonstrates different features and physical scenarios.

## Table of Contents

- [Quick Reference](#quick-reference)
- [Beginner Examples](#beginner-examples)
- [Intermediate Examples](#intermediate-examples)
- [Advanced Examples](#advanced-examples)
- [How to Run Examples](#how-to-run-examples)

## Quick Reference

| Example | Difficulty | Runtime | Description |
|---------|-----------|---------|-------------|
| `minimal_test.json5` | ⚡ Quickest | ~10 sec | Installation verification |
| `hopper.json5` | ⭐ Easy | ~2 min | Material flowing through a hopper outlet |
| `collapse.json5` | ⭐ Easy | ~1 min | Column collapse under gravity |
| `test_slope.json5` | ⭐ Easy | ~1 min | Slope stability test |
| `footing.json5` | ⭐⭐ Medium | ~3 min | Load bearing capacity of granular material |
| `collapse_bi.json5` | ⭐⭐ Medium | ~5 min | Bidisperse collapse with segregation |
| `fluidized_bed.json5` | ⭐⭐ Medium | ~1 min | Two-way coupled laminar gas fluidization |
| `hopper_emptying.json5` | ⭐⭐⭐ Advanced | ~10 min | Detailed hopper flow analysis |
| `temperature.json5` | ⭐⭐⭐ Advanced | ~5 min | Thermal effects in granular flow |

## Beginner Examples

### 0. Minimal Test (`minimal_test.json5`) ⚡ FASTEST

**What it simulates:** Ultra-simple test to verify HGD installation.

**Key features:**
- Smallest possible grid for fast execution
- Very short simulation time (0.5 seconds)
- No video generation (doesn't require ffmpeg)
- Perfect for testing installation

**Run command:**
```bash
python HGD/main.py json/minimal_test.json5
```

**Expected runtime:** ~5-10 seconds

**Output:** Just PNG images in `output/minimal_test/`

**Purpose:** Quick validation that HGD is installed and working correctly.

---

### 1. Hopper Flow (`hopper.json5`)

**What it simulates:** Granular material flowing out of a hopper (funnel-shaped container) through a central outlet.

**Key features:**
- Central outlet boundary condition
- Particle refill from top (continuous flow)
- Tests multiple friction angles (0°, 30°, 60°)
- Bidisperse particle size distribution

**Run command:**
```bash
python HGD/main.py json/hopper.json5
```

**Output:** Videos showing density, particle size, stress field, and strain rate

**What to observe:**
- Flow patterns and velocity profiles
- Effect of friction angle on flow rate
- Particle segregation (large vs small particles)
- Stress distribution in the granular material

**Parameters you can modify:**
- `repose_angle`: Change friction angle (currently [0, 30, 60] degrees)
- `half_width`: Outlet width
- `outlet_rate`: Flow rate (0 to 1)
- `alpha`: Diffusion parameter

---

### 2. Granular Collapse (`collapse.json5`)

**What it simulates:** A column of granular material collapsing under gravity.

**Key features:**
- Simple rectangular column initial condition
- Monodisperse (single size) particles
- Tests both with and without inertia effects
- Fast simulation for quick testing

**Run command:**
```bash
python HGD/main.py json/collapse.json5
```

**Output:** Videos of particle density evolution

**What to observe:**
- Collapse dynamics and runout distance
- Formation of static and flowing zones
- Difference between inertial and non-inertial behavior

**Typical applications:**
- Avalanche dynamics
- Landslide runout
- Silo discharge

---

### 3. Slope Stability (`test_slope.json5`)

**What it simulates:** Granular material on an inclined slope, testing stability conditions.

**Key features:**
- Slope boundary condition
- Tests critical angle of repose
- Simple geometry for validation

**Run command:**
```bash
python HGD/main.py json/test_slope.json5
```

**What to observe:**
- Critical angle where material begins to flow
- Development of surface avalanches
- Equilibrium slope angle

---

## Intermediate Examples

### 4. Bidisperse Collapse (`collapse_bi.json5`)

**What it simulates:** Column collapse with two different particle sizes.

**Key features:**
- Bidisperse particle size distribution (small and large)
- Higher resolution than basic collapse
- Tests both inertial and non-inertial modes
- Demonstrates particle segregation

**Run command:**
```bash
python HGD/main.py json/collapse_bi.json5
```

**Output:** Videos of both density (nu) and particle size (s)

**What to observe:**
- Size segregation during collapse (Brazil nut effect)
- Different velocities for different particle sizes
- Formation of layered structures

**Good for:**
- Studying mixing and segregation
- Understanding size-dependent flow
- Industrial mixing/separation processes

---

### 5. Footing Load Test (`footing.json5`)

**What it simulates:** A loaded footing (foundation) on granular soil.

**Key features:**
- Point load application
- Stress field calculation
- Tests bearing capacity

**Run command:**
```bash
python HGD/main.py json/footing.json5
```

**What to observe:**
- Stress distribution under the load
- Failure mechanisms
- Load-displacement relationship

**Applications:**
- Foundation design
- Bearing capacity analysis
- Geotechnical engineering

---

### 6. Collapse with Inertia (`collapse_inertia.json5`)

**What it simulates:** High-energy granular collapse with full inertial effects.

**Key features:**
- Explicit inertia calculation
- Fixed time step size
- Higher resolution grid

**Run command:**
```bash
python HGD/main.py json/collapse_inertia.json5
```

**What to observe:**
- Dynamic impact forces
- Wave propagation in granular material
- Energy dissipation

---

### 7. Laminar Gas Fluidized Bed (`fluidized_bed.json5`)

**What it simulates:** A monodisperse bed driven by a prescribed superficial gas velocity through the bottom boundary.

```bash
python HGD/main.py json/fluidized_bed.json5
```

The `hgfd` motion model adds layer-wise particle inertia, Gidaspow drag (Wen–Yu in dilute cells and Ergun in dense cells), and equal-and-opposite momentum feedback to a two-dimensional fluid-fraction-weighted incompressible solver. The pressure projection enforces `div(n * u_f) = 0` on the co-located HGD grid.

![Fluidized-bed solid fraction, gas velocity, and gas pressure](images/fluidized_bed.png)

The example writes three-panel `fluidized_bed_*.png` frames and `fluid_diagnostics.csv`. The latter reports pressure drop, pressure-drop/effective-bed-weight ratio, locally fluidized fraction, maximum continuity residual, and maximum stochastic transition probability.

Important parameters:

- `fluid_inlet_velocity`: bottom superficial velocity in m/s
- `gas_density`, `gas_viscosity`: fluid properties
- `defined_time_step_size`: shared HGD/fluid time step
- `fluid_probability_policy`: reject or safely scale an excessive stochastic transition probability
- `fluid_cfl_limit`: fail-fast limit for the fluid Courant number

This solver is intended for two-dimensional, isothermal, laminar research cases. It has not been validated for bubbling, slugging, turbulent, reacting, or industrial-scale fluidized beds.

The coupling equations follow [Li et al., *Coupling Heterarchical Granular Dynamics and Computational Fluid Dynamics*](https://arxiv.org/abs/2606.23202).

## Advanced Examples

### 8. Hopper Emptying Studies (`hopper_emptying*.json5`)

Several detailed hopper studies with different parameters:

- **`hopper_emptying.json5`**: Basic emptying study
- **`hopper_emptying_f10_bi.json5`**: Bidisperse with specific friction
- **`hopper_emptying_mu0p1_poly.json5`**: Polydisperse, low friction
- **`hopper_emptying_mu1.json5`**: High friction case
- **`hopper_emptying_mu1_poly.json5`**: Polydisperse, high friction

**Features:**
- High resolution (nx=51, ny=101, nm=500)
- Long simulation times
- Detailed flow analysis
- Multiple particle size distributions

**Warning:** These simulations require significant computational resources and may take 10-30 minutes to complete.

**Run command (example):**
```bash
python HGD/main.py json/hopper_emptying.json5
```

**What to observe:**
- Detailed discharge rates
- Particle segregation patterns
- Effect of friction on flow
- Jamming and intermittent flow

---

### 9. Temperature Effects (`temperature.json5`)

**What it simulates:** Thermal transport in flowing granular material.

**Key features:**
- Coupled thermal calculation
- Heat diffusion and advection
- Temperature field evolution

**Run command:**
```bash
python HGD/main.py json/temperature.json5
```

**What to observe:**
- Temperature distribution
- Heat transfer between particles
- Effect of flow on thermal mixing

**Applications:**
- Industrial processes (e.g., kilns, reactors)
- Energy dissipation
- Thermal management

---

### 10. Stress Field Test (`test_stress.json5`)

**What it simulates:** Stress distribution calculation in static granular material.

**Key features:**
- Stress tensor calculation
- Different stress models ('active', 'passive', 'K_0')
- Wall friction effects

**Run command:**
```bash
python HGD/main.py json/test_stress.json5
```

**What to observe:**
- Stress chains and force networks
- Pressure distribution
- Effect of boundary conditions

---

### 11. Mesh Dependency Study (`mesh_dependency.json5`)

**What it simulates:** Same problem with different grid resolutions.

**Key features:**
- Multiple grid sizes: [40, 80, 160]
- Tests numerical convergence
- Validates results

**Run command:**
```bash
python HGD/main.py json/mesh_dependency.json5
```

**Purpose:**
- Verify mesh-independent results
- Determine optimal resolution
- Validate numerical methods

---

## Special Examples

### Cyclic Hopper (`hopper_cycle.json5`)

**What it simulates:** Hopper with charge-discharge cycles.

**Key features:**
- Cyclic loading/unloading
- Dynamic boundary conditions
- Time-varying operations

**Applications:**
- Batch processing
- Industrial feeders
- Storage and retrieval systems

---

### Dry Filling (`dry_filling.json5`)

**What it simulates:** Container being filled with granular material.

**Key features:**
- Top filling boundary condition
- Realistic aspect ratio from experimental data
- Deposition and compaction

**Applications:**
- Silo filling
- Powder processing
- Bulk material handling

---

## How to Run Examples

### Basic Usage

```bash
python HGD/main.py json/<example_name>.json5
```

### Monitor Progress

The simulation will display progress bars:
```
Sim:  50%|█████     | 1/2 [00:30<00:30, 30.0s/it]
```

### Stop a Running Simulation

Press `Ctrl+C` to stop gracefully. The simulation will save current progress.

### Adjust Simulation Speed

To run faster (lower quality):
- Reduce `nx`, `ny`, `nm` values
- Increase `save_inc` (save less frequently)
- Reduce `t_f` (shorter simulation time)

Example modification in JSON file:
```json
{
    nx: 25,    // Reduced from 50
    ny: 25,    // Reduced from 50
    t_f: 2,    // Reduced from 5
}
```

## Understanding Output

After running an example, find results in:
```
output/<example_name>/
├── <parameter_variation_1>/
│   ├── nu_0000.png  # Initial state
│   ├── nu_0100.png  # Intermediate frames
│   ├── nu.mp4       # Complete video
│   └── ...
└── <parameter_variation_2>/
    └── ...
```

## Modifying Examples

### Create Your Own Variant

1. Copy an example:
   ```bash
   cp json/hopper.json5 json/my_hopper.json5
   ```

2. Edit parameters in `my_hopper.json5`

3. Run your version:
   ```bash
   python HGD/main.py json/my_hopper.json5
   ```

### Common Modifications

**Change grid resolution:**
```json
nx: 100,  // More horizontal cells
ny: 50,   // More vertical cells
```

**Change material properties:**
```json
repose_angle: 35,       // Friction angle
s_m: 0.001,             // Min particle size (m)
s_M: 0.010,             // Max particle size (m)
gsd_mode: 'poly',       // 'mono', 'bi', or 'poly'
```

**Change simulation time:**
```json
t_f: 10,        // Simulate 10 seconds
save_inc: 50,   // Save every 50 timesteps
```

## Tips for Choosing Examples

**New to HGD?** Start with:
1. `minimal_test.json5` - Fastest verification (10 seconds)
2. `collapse.json5` - Simple physics example
3. `hopper.json5` - Shows boundary conditions
4. `test_slope.json5` - Another simple validation

**Interested in segregation?** Try:
- `collapse_bi.json5`
- `hopper_emptying_*_bi.json5`

**Studying industrial processes?** Look at:
- `hopper_cycle.json5`
- `dry_filling.json5`
- `temperature.json5`

**Validating/developing?** Use:
- `test_slope.json5`
- `test_stress.json5`
- `mesh_dependency.json5`

## Need Help?

- See the [Getting Started Guide](getting-started.md) for installation help
- See the [Troubleshooting Guide](troubleshooting.md) for common issues
- Check [defaults.json5](json/defaults.json5) for all available parameters
- Contact the development team via [Matrix chat](https://matrix.to/#/!UZnyhaLhQymfFbLJYI:matrix.org?via=matrix.org)

Happy simulating! 🎉
