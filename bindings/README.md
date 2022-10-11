# About

Python bindings for the Rust implementation of the Particle Gibbs sampler. 

# Code organization

- `src/data.rs` is a very simple implementation of the `ExternalData` trait expected by the core rust implementation
- `src/lib.rs` exposes the relevant rust functions to python, and doesn't implement any logic
- `py_src/bart.py` was copied from the original codebase, and implements a PyMC BART Random Variable 
- `py_src/pgbart.py` implements the PyMC sampler class (which will call the Rust sampler internally)
- `py_src/logp.py` is a very magical place, and implements the `log-p` function in such a way that it can be called from Rust without GIL.

# Building

Unfortunately, installing PyMC is a pain without `conda`, so I would recommend using it here:

```bash
export ENV_NAME="rust_pgbart_dev"
conda create -c conda-forge -n ${ENV_NAME} "pymc>=4" maturin ipykernel arviz numba numpy
conda activate ${ENV_NAME}
```

Building the code is done with `maturin`

For development:

```bash
maturin develop
```

For release:

```bash
maturin build -r
```
