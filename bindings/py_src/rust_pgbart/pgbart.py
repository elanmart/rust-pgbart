#   Copyright 2022 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from time import perf_counter, time

import aesara
import aesara.link.numba.dispatch
import numpy as np
from aeppl.logprob import CheckParameterValue
from aesara import function as aesara_function  # type: ignore
from numba import carray, cfunc, literal_unroll, njit, types
from pymc.aesaraf import inputvars, join_nonshared_inputs, make_shared_replacements
from pymc.model import modelcontext
from pymc.step_methods.arraystep import ArrayStepShared, Competence

from rust_pgbart.bart import BARTRV
from rust_pgbart.rust_pgbart import initialize, step


# Provide a numba implementation for CheckParameterValue,
# which doesn't exist in aesara
@aesara.link.numba.dispatch.basic.numba_funcify.register(CheckParameterValue)
def numba_functify_CheckParameterValue(op, **kwargs):
    msg = f"Invalid parameter value {str(op)}"

    @aesara.link.numba.dispatch.basic.numba_njit
    def check(value, *conditions):
        for cond in literal_unroll(conditions):
            if not cond:
                raise ValueError(msg)
        return value

    return check


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
    stats_dtypes = [{"Time": object, "N calls": object}]

    def __init__(
        self,
        vars=None,  # pylint: disable=redefined-builtin
        num_particles=20,
        batch=(0.1, 0.1),
        model=None,
    ):
        model = modelcontext(model)
        initial_values = model.initial_point()
        if vars is None:
            vars = model.value_vars
        else:
            vars = [model.rvs_to_values.get(var, var) for var in vars]
            vars = inputvars(vars)
        value_bart = vars[0]

        self.bart = model.values_to_rvs[value_bart].owner.op
        X = self.bart.X
        y = self.bart.Y
        m = self.bart.m
        alpha = self.bart.alpha
        alpha_vec = self.bart.split_prior

        # TODO(elantkom): generalize the code to handle NaNs?
        if np.any(np.isnan(X)):
            raise NotImplementedError()

        # TODO(elantkom): generalize the code to mutlivariate inputs?
        shape = initial_values[value_bart.name].shape  # type: ignore
        assert len(shape) == 1
        shape = 1

        shared = make_shared_replacements(initial_values, vars, model)
        super().__init__(vars, shared)

        numba_fn = make_numba_fn(initial_values, [model.datalogp], vars, shared)
        self.state = initialize(
            X=X,
            y=y,
            logp=numba_fn.address,
            alpha=alpha,
            n_trees=m,
            n_particles=num_particles,
            kfactor=0.75,
            batch=batch,
            split_covar_prior=alpha_vec,
        )

        self.tune = True

    def astep(self, _):

        self.logp_call_counter = 0
        t0 = perf_counter()

        sum_trees = step(self.state, self.tune)

        stats = {"Time": perf_counter() - t0, "N calls": 0}
        return sum_trees, [stats]

    @staticmethod
    def competence(var, has_grad):
        """PGBART is only suitable for BART distributions."""
        dist = getattr(var.owner, "op", None)
        if isinstance(dist, BARTRV):
            return Competence.IDEAL
        return Competence.INCOMPATIBLE


def make_numba_fn(point, out_vars, vars, shared):  # pylint: disable=redefined-builtin
    """Compile Aesara function of the model and the input and output variables.

    Parameters
    ----------
    out_vars: List
        containing :class:`pymc.Distribution` for the output variables
    vars: List
        containing :class:`pymc.Distribution` for the input variables
    shared: List
        containing :class:`aesara.tensor.Tensor` for depended shared data
    """

    out_list, inarray0 = join_nonshared_inputs(point, out_vars, vars, shared)
    function = aesara_function([inarray0], out_list[0], mode=aesara.compile.NUMBA)
    function.trust_input = True
    full_logp = function.vm.jit_fn

    @cfunc(types.float32(types.CPointer(types.float32), types.intc))
    @njit
    def logp(predictions_ptr, predictions_size):
        arr = carray(predictions_ptr, (predictions_size,), np.float32)
        ret = full_logp(arr)[0]
        return ret

    return logp
