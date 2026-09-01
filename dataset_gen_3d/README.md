# 3D FEM Dataset Generation Pipeline

Generates synthetic 3D linear-elastic FEM datasets for training a
physics-informed Graph Neural Network (MeshGraphNet architecture).

## Design Constraint: No Monte Carlo Aggregation

This pipeline generates one independent FEM solve per base sample.
Displacement, stress, strain, and reaction force for a given sample all
come from that single solve — never averaged, aggregated, or combined
across multiple randomized realizations. This is a deliberate departure
from the stochastic-aggregate approach used by some published FEM-surrogate
datasets.

**Why not Monte Carlo?** The CMU SFEM dataset (Ezemba, McComb & Tucker,
2025, ASME J. Mech. Design) performs genuine Monte Carlo aggregation —
50 stochastic point-load realizations per geometry/load combination —
and is deliberately designed to predict the converged statistical /
distributional stress field for uncertainty quantification. That is a
different learning objective from this project's deterministic single-solve
paired-field regression, not a flawed methodology in general. Separately,
SFEM's material properties (E~2.3 GPa, ν~0.40) are far outside
structural-steel range, an independent reason it cannot be reused even
absent the objective mismatch. See ADR-006 in `gnn_project_version_2`
for the full analysis that motivated this decision.

## Material Values

### Numeric / Datasheet Accuracy — **CONFIRMED** (2026-08-18)

| Property | Value | Source |
|----------|-------|--------|
| Young's modulus E | 210 GPa | Eurocode 3 (EN 1993-1-1), SSAB Domex 420MC datasheet |
| Poisson's ratio ν | 0.3 | Standard for structural steel |
| Yield strength σ_yield | 420 MPa | SSAB Domex 420MC datasheet |
| UTS | 480–620 MPa | SSAB Domex 420MC datasheet (conservative min used) |

### Representativeness for Volvo Component Scope — **OPEN**

Whether Domex 420MC is the right material to target for the actual Volvo
components has not been confirmed. The question to resolve before full
production runs: *"What material(s) and rough size range should the
synthetic validation dataset target to be representative of real
components?"* Resolving the datasheet accuracy above does NOT resolve
this representativeness question — they are independent validations.

## Toolchain & Prerequisites

- **Geometry + meshing:** Gmsh Python API, OCC kernel
- **Solving:** FEniCSx (`dolfinx`) or legacy FEniCS (`dolfin`) via `fem-on-colab`
- **Mesh interchange:** meshio (Gmsh .msh → XDMF)
- **No ML dependencies:** PyTorch and PyG are NOT required for dataset generation.

### Installing FEniCSx / DOLFINx

DOLFINx includes compiled C++/MPI bindings and cannot be installed via a simple `pip install`. Choose the installation method for your environment:

#### 1. Google Colab (Recommended)
```bash
# Option A: FEniCSx (dolfinx)
!wget -O - https://fem-on-colab.github.io/releases.sh | bash -s -- --install-fenicsx

# Option B: Legacy FEniCS (dolfin)
!wget "https://fem-on-colab.github.io/releases/fenics-install-real.sh" -O "/tmp/fenics-install.sh"
!bash "/tmp/fenics-install.sh"
```

#### 2. Ubuntu / Debian Linux
```bash
sudo add-apt-repository ppa:fenics-packages/fenics
sudo apt-get update
sudo apt-get install -y python3-dolfinx
```

#### 3. Conda / Mamba Environment
```bash
conda create -n fenicsx-env -c conda-forge fenics-dolfinx mpich python=3.11
conda activate fenicsx-env
```

## Running on Colab

```python
# Cell 1: Install FEniCSx
!wget -O - https://fem-on-colab.github.io/releases.sh | bash -s -- --install-fenicsx

# Cell 2: Install Python dependencies
!pip install -r requirements.txt

# Cell 3: Run calibration
!python -m pipeline.generate_dataset --config config.yaml --calibration-only

# Cell 4: Run full generation
!python -m pipeline.generate_dataset --config config.yaml
```

## Output Schema

Per-sample HDF5 files following ADR-007. See `output/hdf5_writer.py` for
the full schema specification.

## Geometry Families

1. **Block with holes** — box + 1–4 cylindrical through-holes
2. **L-bracket** — box with rectangular notch + parameterized fillet radius
3. **Elongated bar** — high aspect ratio (≥8:1), optional holes
4. **Thin plate** — low height:width ratio, optional holes
5. **Block with fillet** — box with filleted/rounded edges

## License

Internal project — not for redistribution.
