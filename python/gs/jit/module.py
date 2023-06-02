from typing import List
import torch
import time
from .trace import gs_symbolic_trace
from ..matrix_api import Matrix
import numpy as np
from .optimize import (
    merge_relabel_and_all_indices,
    dce,
    cse,
    merge_fused_u_mul_v,
    fuse_e_div_u_SumReduce,
    fuse_ESqure_and_SumReduce,
    fuse_slicing_and_sampling,
    move_constant_to_top,
)

CONVERT_2_MATRIX = "Convert2Matrix"
STATIS_LIST = "StatisList"
GRAPH_ARG = "GRAPH_ARG"
STATIC_ARG = "STATIC_ARG"


def get_actions(args):
    actions = []
    graph_args_count = 0
    static_args_count = 0
    for arg_offset, arg in enumerate(args):
        if isinstance(arg, Matrix):
            actions.append((GRAPH_ARG, graph_args_count, arg_offset, CONVERT_2_MATRIX))
            graph_args_count += 1
        elif isinstance(arg, List):
            actions.append((STATIC_ARG, static_args_count, arg_offset, STATIS_LIST))
            static_args_count += 1
        else:
            actions.append(None)
    return actions


def split_actions(actions):
    graph_actions = []
    static_actions = []
    for action in actions:
        if action is None:
            continue

        if action[0] == GRAPH_ARG:
            graph_actions.append(action)
        elif action[0] == STATIC_ARG:
            static_actions.append(action)

    return graph_actions, static_actions


def generate_graph_args(args, graph_actions):
    graph_args = []
    graph_data_args = []
    for action in graph_actions:
        _, _, arg_offset, a = action
        if a == CONVERT_2_MATRIX:
            graph_args.append(args[arg_offset]._graph)
            graph_data_args.append(
                [
                    args[arg_offset].row_ndata,
                    args[arg_offset].col_ndata,
                    args[arg_offset].edata,
                ]
            )
        else:
            raise ValueError
    return graph_args, graph_data_args


def generate_static_args(args, static_actions):
    static_args = []
    for action in static_actions:
        _, _, arg_offset, a = action
        if a == STATIS_LIST:
            static_args.append(args[arg_offset])
        else:
            raise ValueError
    return static_args


def generate_new_args(
    args, graph_args, inner_graph_data_args, static_args, actions, compact
):
    new_args = []
    for index, action in enumerate(actions):
        if action is None:
            new_args.append(args[index])
        else:
            _, offset, _, a = action
            if a == CONVERT_2_MATRIX:
                new_args.append(
                    Matrix(
                        graph_args[offset],
                        inner_graph_data_args[offset][0],
                        inner_graph_data_args[offset][1],
                        inner_graph_data_args[offset][2],
                        compact,
                    )
                )
            elif a == STATIS_LIST:
                new_args.append(static_args[offset])
            else:
                raise ValueError
    return tuple(new_args)


class compile:
    def __init__(self, func, args, try_compact=True):
        """
        This is auto wrapper for user's func.
        We will create an func inner_wrapper according user's func and its args.
        Each arg in args will have an action, which is used to tell what we should do for the arg.

        There are three types action, 'None', 'GRAPH_ARG', 'STATIC_ARG'.
        We use actions to generate graph_args, static_args.
        And in inner_wrapper, we will leverage graph_args, static_args and user's args with actions to generate new_args.
        By this way, we can convert some args into static args (e.g. fanout=[5,10]) and expand Matrix's methods.

        For 'None' action, we will do nothing and pass arg through into inner_wrapper.

        For 'GRAPH_ARG' action, where arg is 'Matrix', we will store Matrix._graph in graph_args
        and in inner_wrapper, we will use Matrix._graph in graph_args to re-generate origin Matrix.

        For 'STATIC_ARG' action, where arg will not change during training (e.g. fanout, metapath or others). We will store them
        in static_args and they will appear as constants in torch.fx.
        """

        # generate actions
        self.actions = get_actions(args)
        # extract graph_actions and static_actions from acitons
        graph_actions, static_actions = split_actions(self.actions)
        self.graph_actions = graph_actions
        self.static_actions = static_actions
        self._try_compact = try_compact
        self.func = func
        self.iter = 5

        # generate static_args via static_actions
        self.static_args = generate_static_args(args, self.static_actions)

        if self._try_compact:
            compact_gm = self.generate_gm(args, True)
            non_compcat_gm = self.generate_gm(args, False)
            self.gm = self.compact_bench(compact_gm, non_compcat_gm, args)
        else:
            self.gm = self.generate_gm(args, False)

    def __call__(self, *args):
        # generate graph_actions via graph_actions
        # In fact, it stores *._graph in graph_args
        graph_args, graph_data_args = generate_graph_args(args, self.graph_actions)
        return self.gm(args, graph_args, graph_data_args)

    def _bench_gm(self, gm, args):
        time_list = []
        for _ in range(self.iter):
            begin = time.time()
            graph_args, graph_data_args = generate_graph_args(args, self.graph_actions)
            gm(args, graph_args, graph_data_args)
            torch.cuda.synchronize()
            end = time.time()
            time_list.append(end - begin)
        return np.mean(time_list[1:])

    def compact_bench(self, compact_gm, non_compcat_gm, args):
        if self._bench_gm(compact_gm, args) > self._bench_gm(non_compcat_gm, args):
            return non_compcat_gm
        else:
            return compact_gm

    def optimiza_gm(self, gm):
        # pass
        gm = cse(gm)
        gm = dce(gm)
        gm = move_constant_to_top(gm)

        # optimization
        gm = merge_relabel_and_all_indices(gm)
        gm = fuse_slicing_and_sampling(gm)
        gm = fuse_e_div_u_SumReduce(gm)
        gm = fuse_ESqure_and_SumReduce(gm)
        gm = merge_fused_u_mul_v(gm)

        # pass
        gm = dce(gm)
        return gm

    def format_selection_gm(self, gm):
        pass

    def generate_gm(self, args, try_compat):
        def inner_wrapper(inner_args, inner_graph_args, inner_graph_data_args):
            # generate new_args for user's arg.
            # arg in static_args will be compiled as contants.
            # arg in graph_args will be leveraged to generate Matrix.
            new_args = generate_new_args(
                inner_args,
                inner_graph_args,
                inner_graph_data_args,
                self.static_args,
                self.actions,
                try_compat,
            )
            return self.func(*new_args)

        # compiled to torch.fx IR
        graph_args, graph_data_args = generate_graph_args(args, self.graph_actions)
        gm = gs_symbolic_trace(
            inner_wrapper, concrete_args={"inner_graph_data_args": graph_data_args}
        )

        return self.optimiza_gm(gm)
