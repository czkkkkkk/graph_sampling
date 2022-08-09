import imp
import torch
from .trace import gs_symbolic_trace
from ..matrix_api import Matrix

CONVERT_2_MATRIX = "Convert2Matrix"


def parser_args(args):
    inner_wrapper_func_args = []
    global_args = []
    actions = []
    for arg in args:
        if isinstance(arg, Matrix):
            inner_wrapper_func_args.append(None)
            global_args.append(arg._graph)
            actions.append(CONVERT_2_MATRIX)
        else:
            inner_wrapper_func_args.append(arg)
            global_args.append(None)
            actions.append(None)

    return inner_wrapper_func_args, global_args, actions


def generate_args(inner_wrapper_func_args, global_args, actions):
    args = []
    for index, action in enumerate(actions):
        if action == CONVERT_2_MATRIX:
            args.append(Matrix(global_args[index]))
        else:
            args.append(inner_wrapper_func_args[index])
    return tuple(args)

# [todo(ping)]
# 1. How to cache gm and its global_args after compilation?
# 2. After compilation with caching, some user args will not works (e.g. Matrix, fan_out[5,10,15]).
#   because we have cached them in global_args. How to solve it?
def compile(func):

    def wrapper(*args, **kwargs):
        assert (len(kwargs) == 0), "gs.jit.compile not support kwargs"
        inner_wrapper_func_args, global_args, actions = parser_args(args)

        def inner_wrapper(inner_wrapper_func_args):
            args = generate_args(inner_wrapper_func_args, global_args, actions)
            return func(*args)

        gm = gs_symbolic_trace(inner_wrapper)
        # do pass, optimization and transformation here
        ## gm = dce(gm)
        return gm(inner_wrapper_func_args)

    return wrapper