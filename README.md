# About

Implementation of Bayesian Additive Regression Trees (BART) written in Rust.

Python bindings and integration with PyMC and Numpyro (wip) are also provided.

# Code organization

- `pgbart/src` contains the generic implementation of the Particle Gibbs sampler. 
It does not have any notion of Python or PyMC. It only assumes you can provide the input data (`X`, `y`) and
a function `logp(predictions) -> float`.
- `bindings/` contains the Rust + Python + PyMC bindings code. See its `README.md` for details. 

# Todo

This is a work in progress

- [ ] Out-of-sample predictions (Rust code is there, need to expose it in python)
- [ ] Variable importance plots (Rust code is there, nede to expose it in python)
- [ ] Partial dependence plots 
- [ ] Tests and CI
- [ ] Docs
- [ ] Numpyro integration
- [ ] Support `ND` targets
- [ ] Support `NaNs`
- [ ] PyPI package?

# Building

See `./bindings/README.md`


# Credits

- BART implementation [paper](https://arxiv.org/abs/2206.03619)
- [Original PyMC implementation](https://github.com/pymc-devs/pymc-bart/tree/0f0e3617ac03877448f5eded315e8cb810d1d0cb)
- [nutpie](https://github.com/pymc-devs/nutpie/tree/9029f9167496ad72fcd56975e56836798da75e0d)
