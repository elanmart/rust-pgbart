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

import numpy as np

from aesara import function as aesara_function  # type: ignore
from pymc.model import modelcontext
from pymc.step_methods.arraystep import ArrayStepShared, Competence
from pymc.aesaraf import inputvars, join_nonshared_inputs, make_shared_replacements

from rust_pgbart.bart import BARTRV
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
    stats_dtypes = [{"variable_inclusion": object, "bart_trees": object}]

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
        shared = make_shared_replacements(initial_values, vars, model)
        pymc_logp = logp(initial_values, [model.datalogp], vars, shared)

        def model_logp(predictions):
            r = pymc_logp(predictions)
            return r

        # TODO(elantkom): generalize the code to handle NaNs?
        if np.any(np.isnan(X)):
            raise NotImplementedError()

        # TODO(elantkom): generalize the code to mutlivariate inputs?
        shape = initial_values[value_bart.name].shape  # type: ignore
        assert len(shape) == 1
        shape = 1

        self.state = initialize(
            X=X,
            y=y,
            logp=model_logp,
            alpha=alpha,
            n_trees=m,
            n_particles=num_particles,
            kfactor=0.75,
            batch=batch,
            split_covar_prior=alpha_vec,
        )

        self.tune = True
        super().__init__(vars, shared)

    def astep(self, _):
        sum_trees = step(self.state, self.tune)
        stats = {"variable_inclusion": 0., "bart_trees": 0.}
        return sum_trees, [stats]

    @staticmethod
    def competence(var, has_grad):
        """PGBART is only suitable for BART distributions."""
        dist = getattr(var.owner, "op", None)
        if isinstance(dist, BARTRV):
            return Competence.IDEAL
        return Competence.INCOMPATIBLE


def logp(point, out_vars, vars, shared):  # pylint: disable=redefined-builtin
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
    function = aesara_function([inarray0], out_list[0])
    function.trust_input = True
    return function
