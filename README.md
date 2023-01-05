# About

[![ci](https://github.com/elanmart/rust-pgbart/actions/workflows/ci.yaml/badge.svg)](https://github.com/elanmart/rust-pgbart/actions/workflows/ci.yaml)

This repo implements Particle Gibbs sampler for Bayesian Additive Regression Trees (BART) written in Rust.

This implementation is ~10x faster than a vanilla python version, which makes it feasible to run the algorithm on much larger datasets.

Python bindings and integration with PyMC and Numpyro (coming soon) are also provided.

To use the model in PyMC, you can simply write

```python
with pm.Model() as model_coal:
    μ_ = rust_pgbart.BART("μ_", X=x_data, Y=y_data, m=20)
    μ = pm.Deterministic("μ", np.abs(μ_))
    y_pred = pm.Poisson("y_pred", mu=μ, observed=y_data)
    idata_coal = pm.sample()
```

See `examples` for the full code. 

# Code organization

- `pgbart/src` contains the generic implementation of the Particle Gibbs sampler. 
It does not have any notion of Python or PyMC. It only assumes you can provide the input data (`X`, `y`) and
a function `logp(predictions) -> float`.
- `bindings/` contains the Rust + Python + PyMC bindings code. See its `README.md` for details. 

# Building

See `./bindings/README.md`

# ToDo

This is still work in progress, and not all features are supported. Most notably out-of-sample prediction and variable importance plots are not yet available in Python (but the Rust code is mostly there).

# Credits

I've developed this code as part of my effort to re-write [Bayesian Modeling and Computation in Python](https://bayesiancomputationbook.com/welcome.html) into `numpyro`.

The implementation was based on:
- BART implementation [paper](https://arxiv.org/abs/2206.03619)
- [Original PyMC implementation](https://github.com/pymc-devs/pymc-bart/tree/0f0e3617ac03877448f5eded315e8cb810d1d0cb)
- [nutpie](https://github.com/pymc-devs/nutpie/tree/9029f9167496ad72fcd56975e56836798da75e0d)
