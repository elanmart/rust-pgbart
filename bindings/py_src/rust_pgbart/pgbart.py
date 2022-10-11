# The code in this file is based on 
# https://github.com/pymc-devs/pymc-bart/blob/0f0e3617ac03877448f5eded315e8cb810d1d0cb/pymc_bart/pgbart.py

from time import perf_counter

import numpy as np
from pymc.aesaraf import inputvars
from pymc.model import modelcontext
from pymc.step_methods.arraystep import ArrayStepShared, Competence

from rust_pgbart.bart import BARTRV
from rust_pgbart.logp import ModelLogPWrapper

# Defined in Rust
from rust_pgbart.rust_pgbart import initialize, step


class PGBART(ArrayStepShared):
    """
    Particle Gibss BART sampling step.

    Parameters
    ----------
    vars: list
        List of value variables for sampler
    num_particles : int
        Number of particles for the conditional SMC sampler. Defaults to 40
    batch : int or tuple
        Number of trees fitted per step. Defaults to  "auto", which is the 10% of the `m` trees
        during tuning and after tuning. If a tuple is passed the first element is the batch size
        during tuning and the second the batch size after tuning.
    model: PyMC Model
        Optional model for sampling step. Defaults to None (taken from context).
    """

    name = "pgbart"
    default_blocked = False
    generates_stats = True
    stats_dtypes = [{"time_py": float, "time_rs": float}]

    def __init__(
        self,
        vars=None,  # pylint: disable=redefined-builtin
        num_particles=20,
        batch=(0.1, 0.1),
        model=None,
    ):
        # PyMC setup
        model = modelcontext(model)

        # Get the data and params from the BART instance
        self.bart = self.get_bart_rv(model, vars)
        X = self.bart.X
        y = self.bart.Y
        m = self.bart.m
        alpha = self.bart.alpha
        alpha_vec = self.bart.split_prior

        # TODO(elantkom): generalize the code to handle NaNs?
        if np.any(np.isnan(X)):
            raise NotImplementedError()

        # Generate the log-p C function that can be passed to Rust
        self.logp_wrapper = ModelLogPWrapper(model, vars)

        # Intialize the Rust sampler
        self.state = initialize(
            X=X,
            y=y,
            logp=self.logp_wrapper.logp_function_pointer(),
            alpha=alpha,
            n_trees=m,
            n_particles=num_particles,
            kfactor=0.75,
            batch=batch,
            split_covar_prior=alpha_vec,
        )

        self.tune = True
        super().__init__(vars, self.logp_wrapper.shared)

    def get_bart_rv(self, model, vars):
        """ Get the instance of BART RV from the PyMC model
        """

        if vars is None:
            vars = model.value_vars
        else:
            vars = [model.rvs_to_values.get(var, var) for var in vars]
            vars = inputvars(vars)
        value_bart = vars[0]

        bart = model.values_to_rvs[value_bart].owner.op
        return bart


    def astep(self, _):
        
        t0 = perf_counter()
        
        self.logp_wrapper.update_persistent_arrays()
        
        t1 = perf_counter()
        
        sum_trees = step(self.state, self.tune)

        t2 = perf_counter()

        return sum_trees, [{"time_py": t1 - t0, "time_rs": t2 - t1}]

    @staticmethod
    def competence(var, has_grad):
        """PGBART is only suitable for BART distributions."""
        dist = getattr(var.owner, "op", None)
        if isinstance(dist, BARTRV):
            return Competence.IDEAL
        return Competence.INCOMPATIBLE
