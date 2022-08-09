import imp
import torch
from .trace import gs_symbolic_trace
from ..matrix_api import Matrix

CONVERT_2_MATRIX = "Convert2Matrix"


def parser_args(args):
    global_args = []
    actions = []
    for arg in args:
        if isinstance(arg, Matrix):
            global_args.append(arg._graph)
            actions.append(CONVERT_2_MATRIX)
        else:
            global_args.append(None)
            actions.append(None)

    return global_args, actions


def generate_args(args, global_args, actions):
    new_args = []
    for index, action in enumerate(actions):
        if action == CONVERT_2_MATRIX:
            new_args.append(Matrix(global_args[index]))
        else:
            new_args.append(args[index])
    return tuple(new_args)

# [todo(ping)]
# 1. How to cache gm and its global_args after compilation?
# 2. After compilation with caching, some user args will not works (e.g. Matrix, fan_out[5,10,15]).
#   because we have cached them in global_args. How to solve it?
def compile(func):

    def wrapper(*args, **kwargs):
        assert (len(kwargs) == 0), "gs.jit.compile not support kwargs"
        global_args, actions = parser_args(args)

        def inner_wrapper(args):
            new_args = generate_args(args, global_args, actions)
            return func(*new_args)

        gm = gs_symbolic_trace(inner_wrapper)
        # do pass, optimization and transformation here
        ## gm = dce(gm)
        return gm(args)

    return wrapper