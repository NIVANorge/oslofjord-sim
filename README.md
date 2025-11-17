# Oslofjord simulation

## Data preparation

In the preprocess folder there are Python Jupyter notebook preprocessing scripts to make:
- A grid file;
- Forcing file containing for boundary conditions, rivers, etc.;
- Atmospheric forcing files (todo).

## Installation

Python:
1. Install conda <https://conda-forge.org/download/>.
2. Create a conda environment `conda create --name oslofjord python=3.12`
3. Activate the envoronment `conda activate oslofjord`
4. Navigate to the oslofjord directory
5. Install the dependencies `pip install -e .`
6. Xesmf is required, see <https://xesmf.readthedocs.io/en/stable/installation.html>;
   install with `conda install -c conda-forge xesmf`.

Julia:
1. Install julia <https://julialang.org/downloads/>.
2. Run Julia REPL from the directory with `Project.toml` and activate the FjordsSim environment `julia --project`.
3. Enter the Pkg REPL by pressing `]` from Julia REPL.
4. Type `instantiate` to 'resolve' a `Manifest.toml` from a `Project.toml` to install and precompile dependency packages.

## Usage

1. Download the [initial grid (OF160_grid_v2.nc) and atmospheric forcing files](https://www.dropbox.com/scl/fo/gc3yc155b5eohi7998wgh/AGN2Yt3HyQ0LlZGImpcca6o?rlkey=x6okc3uxe2avud6sbxgd00l14&st=093llyqp&dl=0).
2. Use scripts in the preprocess folder to download and prepare the bathymetry and forcing files.
3. Run simulation `julia --project simulation.jl`