import operator
import torch.fx as fx
from typing import List, Tuple


def flatten(iter):
    ret = []
    for i in iter:
        if isinstance(i, List) or isinstance(i, Tuple):
            ret = ret + flatten(i)
        else:
            ret.append(i)
    return ret


def dce(gm: fx.GraphModule) -> fx.GraphModule:
    used_nodes_set = set()
    nodes_list = gm.graph.nodes
    for node in reversed(nodes_list):
        if node.op == "output" or node.op == "placeholder":
            used_nodes_set.add(node)

        if node in used_nodes_set:
            for pre_node in flatten(node.args):
                if isinstance(pre_node, fx.Node):
                    used_nodes_set.add(pre_node)

            for _, value in node.kwargs.items():
                if isinstance(value, fx.Node):
                    used_nodes_set.add(value)

    for node in reversed(nodes_list):
        if node not in used_nodes_set:
            gm.graph.erase_node(node)

    gm.graph.lint()
    gm.recompile()
    return gm


def cse(gm: fx.GraphModule) -> fx.GraphModule:
    nodes_list = gm.graph.nodes
    first_appear_ce_node = {}
    replace_nodes_set = set()
    for index, node in enumerate(nodes_list):
        key = str(node.target) + str(node.args) + str(node.kwargs)
        if key not in first_appear_ce_node:
            first_appear_ce_node[key] = node
        else:
            replace_nodes_set.add(node)

    for node in replace_nodes_set:
        key = str(node.target) + str(node.args) + str(node.kwargs)
        new_node = first_appear_ce_node[key]
        node.replace_all_uses_with(new_node)
        gm.graph.erase_node(node)

    gm.graph.lint()
    gm.recompile()
    return gm


def merge_relabel_and_all_indices(gm: fx.GraphModule) -> fx.GraphModule:
    gm = cse(gm)
    merge_dir = {}

    # scan
    for node in gm.graph.nodes:
        if node.target == "_CAPI_GetValidNodes" or node.target == "_CAPI_GraphRelabel":
            # node.args[0] is the parent of node
            if node.args[0] not in merge_dir:
                merge_dir[node.args[0]] = [node]
            else:
                merge_dir[node.args[0]].append(node)

    # begin merge
    for key, value in merge_dir.items():
        if len(value) < 2:
            continue

        if len(value) > 2:
            print(
                "{} has more than two children: {}, something wrong?".format(key, value)
            )

        with gm.graph.inserting_before(value[0]):
            new_relabel_node = None
            if value[0].target == "_CAPI_GraphRelabel":
                new_relabel_node = gm.graph.node_copy(value[0])
            else:
                new_relabel_node = gm.graph.node_copy(value[1])

            getitem_1 = gm.graph.call_function(
                operator.getitem, args=(new_relabel_node, 0)
            )
            getitem_2 = gm.graph.call_function(
                operator.getitem, args=(new_relabel_node, 1)
            )
            getitem_3 = gm.graph.call_function(
                operator.getitem, args=(new_relabel_node, 2)
            )
            getitem_4 = gm.graph.call_function(
                operator.getitem, args=(new_relabel_node, 3)
            )

            new_getitem_list = [getitem_1, getitem_2, getitem_3, getitem_4]

            for v in value:
                if v.target == "_CAPI_GetValidNodes":
                    # replace original all_idices
                    v.replace_all_uses_with(getitem_1)

                if v.target == "_CAPI_GraphRelabel":
                    # replace original gettitem of relabel
                    for i in v.users:
                        i.replace_all_uses_with(new_getitem_list[i.args[1]])

                    # replace original relabel
                    v.replace_all_uses_with(new_relabel_node)

    # remove dead code
    gm = dce(gm)
    return gm


def fuse_slicing_and_sampling(gm):
    """
    Fuses columnwise_slicing and columnwise_sampling
    """
    for node in gm.graph.nodes:
        if (
            node.target == "_CAPI_columnwise_sampling"
            and node.args[0].target == "_CAPI_columnwise_slicing"
        ):
            if len(node.args[0].users) > 1:
                continue
            with gm.graph.inserting_after(node):
                new_node = gm.graph.call_method(
                    "_CAPI_fused_columnwise_slicing_sampling",
                    args=(
                        *node.args[0].args,
                        *node.args[1:],
                    ),
                )
                node.replace_all_uses_with(new_node)

    # remove dead code
    gm = dce(gm)
    return gm
