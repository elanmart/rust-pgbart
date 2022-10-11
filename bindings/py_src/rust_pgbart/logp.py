import aesara
import aesara.link.numba.dispatch
import aesara.link.numba.dispatch.basic
import numba
import numba.extending
import numpy as np
from aeppl.logprob import CheckParameterValue
from aesara import function as aesara_function  # type: ignore
from numba import carray, cfunc, float64, literal_unroll, njit, types
from numba.core import cgutils, types
from pymc.aesaraf import join_nonshared_inputs, make_shared_replacements


@aesara.link.numba.dispatch.basic.numba_funcify.register(CheckParameterValue)
def numba_functify_CheckParameterValue(op, **kwargs):
    """Provides a numba implementation for CheckParameterValue, which doesn't exist in aesara

    Full credit goes to nutpie:
    https://github.com/pymc-devs/nutpie/blob/e25525be56cafa8732852acf1e64a8056932f40a/nutpie/compile_pymc.py#L24-L35
    """

    msg = f"Invalid parameter value {str(op)}"

    @aesara.link.numba.dispatch.basic.numba_njit
    def check(value, *conditions):
        for cond in literal_unroll(conditions):
            if not cond:
                raise ValueError(msg)
        return value

    return check


@numba.extending.intrinsic
def address_as_void_pointer(typingctx, src):
    """Returns a void pointer from a given memory address

    Full credit goes to: https://stackoverflow.com/a/61550054
    """

    sig = types.voidptr(src)

    def codegen(cgctx, builder, sig, args):
        return builder.inttoptr(args[0], cgutils.voidptr_t)

    return sig, codegen


class ModelLogPWrapper:
    def __init__(self, model, vars):

        aesara_fn, shared = self.make_aesara_fn(model, vars)
        self.aesara_fn = aesara_fn
        self.shared = shared
        self.logp_args = self.make_persistent_arrays()
        self.logp = self.generate_logp_function()

    def logp_function_pointer(self):
        return self.logp.address

    def make_aesara_fn(self, model, vars):
        """ Generate an aesara function given the model.
        
        This function will take a single input: BART predictions,
        and return a log-p of the whole model at a given time step.
        """

        initial_values = model.initial_point()
        shared = make_shared_replacements(initial_values, vars, model)
        out_vars = [model.datalogp]
        out_list, inarray0 = join_nonshared_inputs(initial_values, out_vars, vars, shared)
        function = aesara_function([inarray0], out_list[0], mode=aesara.compile.NUMBA)
        function.trust_input = True

        return function, shared

    def make_persistent_arrays(self):
        """ Generate the shared arrays that will be captured by the log-p C function
        """

        arrays = [item.storage[0].copy() for item in self.aesara_fn.input_storage[1:]]
        assert all(arr.dtype == np.float64 for arr in arrays)

        return arrays
    
    def update_persistent_arrays(self):
        """ Update the shared arrays with new data 
        """
        
        for array, storage in zip(self.logp_args, self.aesara_fn.input_storage[1:]):
            new_arr = storage.storage[0]
            assert array.shape == new_arr.shape

            # we can't use arr[:] = new_arr with 0-dim inputs
            array *= 0.0
            array += new_arr

    def generate_logp_function(self):
        """ Generate a C function for evaluating log-p of the model.
        
        This is tricky, because Rust expects a function with signature
        fn(predictions) -> float

        But the numba-compiled aesara function has a signature
        fn(predictions, *args) -> float

        Where args change at each iteration of the algorithm (they are sampled by NUTS)

        So we need to do some really weird stuff in here to make that work.
        Hold tight.
        """

        # Arrays containing the remaining arguments for log-p evaluation
        extra_args = self.logp_args

        # Numba-compiled log-p function
        aesara_logp = self.aesara_fn.vm.jit_fn  # type: ignore

        # And now we need to generate some code
        # This is due to limitations in numba support for creating tuples
        code = [
            "def _logp(ptr, size):",
            "    data = carray(ptr, (size, ), dtype=float64)"
        ]

        # Now we will fetch the remaining arguments
        # You might be asking: what is happening here?
        # Well...
        # We have these shared numpy arrays, `extra_args` above.
        # And each numpy array has a pointer to data that we can extract.
        # We will essentially bake those pointers (int64 values) 
        # into this function source code. 
        # When we update the array contents in python
        # this function will still have only the pointer.
        # So next time it's called, it will see the updated data
        # Don't @ me
        for i, array in enumerate(extra_args):
            line = "    arg{} = carray(address_as_void_pointer({}), {}, dtype={})".format(
                i, array.ctypes.data, array.shape, array.dtype
            )
            code.append(line)

        # the last line simply calls the model logp function
        # [0] because this returns a tuple
        # Note: maybe there is a reason it returns a tuple and [0] will break smth?
        ret = "    return aesara_logp(data, {})[0]".format(
            ", ".join(f"arg{i}" for i in range(len(extra_args)))
        )
        code.append(ret)

        # FYI an example source code this would generate looks like this
        # def _logp(ptr, size):
        #     data = carray(ptr, (size, ), dtype=float64)
        #     arg0 = carray(12345, (2, ), dtype=float64)
        #     arg1 = carray(56789, (), dtype=float64)
        #     return aesara_logp(data, arg0, arg1)[0]
        source = "\n".join(code)

        # Compile -- we need to pass everyhing in the current scope as "globals"
        ldict = locals()
        gdict = {**globals(), **locals()}
        exec(source, gdict, ldict)

        # And we fetch the function
        logp = njit(ldict["_logp"])

        # Which we then convert to a pointer to a C function...
        sig = types.float64(
            types.CPointer(types.float64),
            types.intc,
        )
        logp = cfunc(sig)(logp)

        # Aaaand we're done
        return logp
