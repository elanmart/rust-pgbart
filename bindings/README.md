# Building

Unfortunately, installing PyMC is a pain without `conda`, so I would recommend using it here:

```bash
export ENV_NAME="rust_pgbart_dev"
conda create -c conda-forge -n ${ENV_NAME} "pymc>=4" "maturin"
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
