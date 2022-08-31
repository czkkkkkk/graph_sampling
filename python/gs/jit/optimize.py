import operator
from .passes import dce
import torch.fx as fx


def merge_relabel_and_all_indices(gm: fx.GraphModule) -> fx.GraphModule:
    merge_dir = {}

    # scan
    for node in gm.graph.nodes:
        if node.target == 'all_indices':
            # node.args[0] is the parent of node
            if node.args[0] not in merge_dir:
                merge_dir[node.args[0]] = [node]
            else:
                merge_dir[node.args[0]].append(node)

        if node.target == 'relabel':
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
            print("{} has more than two children: {}, something wrong?".format(
                key, value))

        all_indices_node = None
        relabel_node = None
        all_indices_is_before = None

        if value[0].target == 'all_indices' and value[1].target == 'relabel':
            all_indices_node = value[0]
            relabel_node = value[1]
            all_indices_is_before = True
        elif value[0].target == 'relabel' and value[1].target == 'all_indices':
            all_indices_node = value[1]
            relabel_node = value[0]
            all_indices_is_before = False
        else:
            raise ValueError

        if all_indices_is_before:
            with gm.graph.inserting_before(all_indices_node):

                # create new relabel node and fetch the results
                new_relabel_node = gm.graph.node_copy(relabel_node)
                getitem_1 = gm.graph.call_function(operator.getitem,
                                                   args=(new_relabel_node, 0))
                getitem_2 = gm.graph.call_function(operator.getitem,
                                                   args=(new_relabel_node, 1))
                getitem_3 = gm.graph.call_function(operator.getitem,
                                                   args=(new_relabel_node, 2))

                new_getitem_list = [getitem_1, getitem_2, getitem_3]

                # replace original all_idices
                all_indices_node.replace_all_uses_with(getitem_1)

                # replace original gettitem of relabel
                for i in relabel_node.users:
                    i.replace_all_uses_with(new_getitem_list[i.args[1]])

                # replace original relabel
                relabel_node.replace_all_uses_with(new_relabel_node)

        else:
            with gm.graph.inserting_after(relabel_node):
                old_getitem = [i for i in relabel_node.users]

                getitem_3 = gm.graph.call_function(operator.getitem,
                                                   args=(relabel_node, 2))
                getitem_2 = gm.graph.call_function(operator.getitem,
                                                   args=(relabel_node, 1))
                getitem_1 = gm.graph.call_function(operator.getitem,
                                                   args=(relabel_node, 0))

                new_getitem_list = [getitem_1, getitem_2, getitem_3]

                # replace original all_idices
                all_indices_node.replace_all_uses_with(getitem_1)

                # replace original gettitem of relabel
                for i in old_getitem:
                    i.replace_all_uses_with(new_getitem_list[i.args[1]])

    # remove dead code
    gm = dce(gm)
    return gm


def fuse_slicing_and_sampling(gm):
    """
    Fuses columnwise_slicing and columnwise_sampling
    """
    for node in gm.graph.nodes:
        if node.target == 'columnwise_sampling' and node.args[
                0].target == 'columnwise_slicing':
            if len(node.args[0].users) > 1:
                continue
            with gm.graph.inserting_after(node):
                new_node = gm.graph.call_method(
                    'fused_columnwise_slicing_sampling',
                    args=(
                        *node.args[0].args,
                        *node.args[1:],
                    ))
                node.replace_all_uses_with(new_node)

    # remove dead code
    gm = dce(gm)
    return gm