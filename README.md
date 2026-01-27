# Oslofjord simulation with [FjordSim](https://github.com/NIVANorge/FjordSim.jl)

## Data preparation

In the preprocess folder there are Python Jupyter notebooks to make:
- A grid file;
- A forcing file for boundary conditions, rivers, etc.;
- An atmospheric forcing file can be prepared using <https://github.com/limash/atm-forcing.git>.

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
2. Clone the repository: `git clone https://github.com/NIVANorge/oslofjord-sim.git` 
3. Run Julia REPL from the directory with `Project.toml` and activate the environment: `julia --project`.
4. Enter the Pkg REPL by pressing `]` from Julia REPL.
5. Type `instantiate` to 'resolve' a `Manifest.toml` from a `Project.toml` to install and precompile dependency packages.
(you may need to `add https://github.com/NIVANorge/FjordSim.jl.git` from Pkg REPL to install the latest FjordSim).

## Usage

there are 2 options:
1. Download the prepared in advance files (`bathymetry_105to232.nc, forcing_105to232.nc, JRA55 files or NORA3.nc`) from [here](https://www.dropbox.com/scl/fo/gc3yc155b5eohi7998wgh/AGN2Yt3HyQ0LlZGImpcca6o?rlkey=x6okc3uxe2avud6sbxgd00l14&st=093llyqp&dl=0) to run a simulation.
2. Use scripts in the preprocess folder to download and prepare the bathymetry and forcing files.
In this case you can add rivers, other sinks and sources, change other forcing for any variable.

Run simulation `julia --project simulation.jl`