from ast import arg
from re import A
from typing import List
from .trace import gs_symbolic_trace
from ..matrix_api import Matrix

CONVERT_2_MATRIX = "Convert2Matrix"
STATIS_LIST = "StatisList"
GRAPH_ARG = 1
STATIC_ARG = 2


def get_actions(args):
    actions = []
    graph_args_count = 0
    static_args_count = 0
    for arg in args:
        if isinstance(arg, Matrix):
            actions.append((GRAPH_ARG, graph_args_count, CONVERT_2_MATRIX))
            graph_args_count += 1
        elif isinstance(arg, List):
            actions.append((STATIC_ARG, static_args_count, STATIS_LIST))
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
    for index, action in enumerate(graph_actions):
        a = action[2]
        if a == CONVERT_2_MATRIX:
            graph_args.append(args[index]._graph)
        else:
            raise ValueError
    return graph_args


def generate_static_args(args, static_actions):
    static_args = []
    for index, action in enumerate(static_actions):
        a = action[2]
        if a == STATIS_LIST:
            static_args.append(args[index])
        else:
            raise ValueError
    return static_args


def generate_new_args(args, graph_args, static_args, actions):
    new_args = []
    for index, action in enumerate(actions):
        if action is None:
            new_args.append(args[index])
        else:
            _, offset, a = action
            if a == CONVERT_2_MATRIX:
                new_args.append(Matrix(graph_args[index]))
            elif a == STATIS_LIST:
                new_args.append(static_args[offset])
            else:
                raise ValueError
    return tuple(new_args)


class compile_class:

    def __init__(self, func, args):
        actions = get_actions(args)
        graph_actions, static_actions = split_actions(actions)
        self.graph_actions = graph_actions
        self.static_actions = static_actions
        static_args = generate_static_args(args, self.static_actions)

        def inner_wrapper(inner_args, inner_graph_args):
            new_args = generate_new_args(inner_args, inner_graph_args,
                                         static_args, actions)
            return func(*new_args)

        gm = gs_symbolic_trace(inner_wrapper)
        self.gm = gm

    def __call__(self, *args):
        graph_args = generate_graph_args(args, self.graph_actions)
        return self.gm(args, graph_args)