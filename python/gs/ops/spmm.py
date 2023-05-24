"""Internal module for general spmm operators."""
import sys
import torch

from ..format import _COO, _CSC, _CSR


__all__ = ["gspmm"]


def reshape_lhs_rhs(lhs_data, rhs_data):
    r"""Expand dims so that there will be no broadcasting issues with different
    number of dimensions. For example, given two shapes (N, 3, 1), (E, 5, 3, 4)
    that are valid broadcastable shapes, change them to (N, 1, 3, 1) and
    (E, 5, 3, 4)

    Parameters
    ----------
    lhs_data : tensor or None
        The left operand, could be None if it's not required by op.
    rhs_data : tensor or None
        The right operand, could be None if it's not required by op.
    """
    lhs_shape = lhs_data.shape
    rhs_shape = rhs_data.shape
    if len(lhs_shape) != len(rhs_shape):
        max_ndims = max(len(lhs_shape), len(rhs_shape))
        lhs_pad_ndims = max_ndims - len(lhs_shape)
        rhs_pad_ndims = max_ndims - len(rhs_shape)
        new_lhs_shape = (lhs_shape[0],) + (1,) * lhs_pad_ndims + lhs_shape[1:]
        new_rhs_shape = (rhs_shape[0],) + (1,) * rhs_pad_ndims + rhs_shape[1:]
        lhs_data = lhs_data.view(new_lhs_shape)
        rhs_data = rhs_data.view(new_rhs_shape)
    return lhs_data, rhs_data


def infer_broadcast_shape(op, shp1, shp2):
    r"""Check the shape validity, and infer the output shape given input shape and operator.
    Note the both :attr:`shp1`, :attr:`shp2` and the returned shape are feature
    shapes (i.e. we remove the first dimension, which correspond to graph statistics
    such as number of nodes, number of edges, etc.).

    We allow applying op on operands with different shapes, according to the
    broadcasting semantics of Numpy/Scipy:
    https://numpy.org/doc/stable/user/basics.broadcasting.html

    Parameters
    ----------
    op : str
        The binary op's name, could be `add`, `sub`, `mul`, `div`, `dot`, `copy_lhs`, `copy_rhs`.
    shp1 : tuple[int]
        The shape of lhs operand.
    shp2 : tuple[int]
        The shape of rhs operand.

    Returns
    -------
    tuple[int]
        shape after broadcasting
    """
    pad_shp1, pad_shp2 = shp1, shp2
    if op == "dot":
        if shp1[-1] != shp2[-1]:
            raise "Dot operator is only available for arrays with the same size on last dimension, but got {} and {}.".format(
                shp1, shp2
            )
    # operands are padded to have the same dimensionality with leading 1's.
    if len(shp1) > len(shp2):
        pad_shp2 = (1,) * (len(shp1) - len(shp2)) + shp2
    elif len(shp1) < len(shp2):
        pad_shp1 = (1,) * (len(shp2) - len(shp1)) + shp1
    for d1, d2 in zip(pad_shp1, pad_shp2):
        if d1 != d2 and d1 != 1 and d2 != 1:
            raise "Feature shapes {} and {} are not valid for broadcasting.".format(
                shp1, shp2
            )
    rst = tuple(max(d1, d2) for d1, d2 in zip(pad_shp1, pad_shp2))
    return rst[:-1] + (1,) if op == "dot" else rst


def gspmm(g, op, reduce_op, lhs_data, rhs_data, lhs_target, on_format):
    r"""Generalized Sparse Matrix Multiplication interface.
    It fuses two steps into one kernel.

    1. Computes messages by :attr:`op` source node and edge features.
    2. Aggregate the messages by :attr:`reduce_op` as the features on destination nodes.

    .. math::
        x_v = \psi_{(u, v, e)\in \mathcal{G}}(\rho(x_u, x_e))

    where :math:`x_v` is the returned feature on destination nodes, and :math:`x_u`,
    :math:`x_e` refers to :attr:`u`, :attr:`e` respectively. :math:`\rho` means binary
    operator :attr:`op` and :math:`\psi` means reduce operator :attr:`reduce_op`,
    :math:`\mathcal{G}` is the graph we apply gspmm on: :attr:`g`.

    Note that this function does not handle gradients.

    Parameters
    ----------
    g : DGLGraph
        The input graph.
    op : str
        The binary op's name, could be ``add``, ``sub``, ``mul``, ``div``,
        ``copy_lhs``, ``copy_rhs``.
    reduce_op : str
        Reduce operator, could be ``sum``, ``max``, ``min``, ``mean``.
    lhs_data : tensor or None
        The left operand, could be None if it's not required by the op.
    rhs_data : tensor or None
        The right operand, could be None if it's not required by the op.

    Returns
    -------
    tensor
        The result tensor.
    """
    if op not in ["copy_lhs", "copy_rhs"]:
        lhs_data, rhs_data = reshape_lhs_rhs(lhs_data, rhs_data)
    # With max and min reducers infinity will be returned for zero degree nodes
    if op == "sub":
        op = "add"
        rhs_data = -rhs_data
    if op == "div":
        op = "mul"
        rhs_data = 1.0 / rhs_data

    u = lhs_data
    e = rhs_data
    u_target = lhs_target

    use_u = op != "copy_rhs"
    use_e = op != "copy_lhs"
    if use_u and use_e:
        if u.device != e.device:
            raise "The operands data device don't match: {} and {}, please move them to the same device.".format(
                u.device, e.device
            )
        if u.dtype != e.dtype:
            raise "The node features' data type {} doesn't match edge features' data type {}, please convert them to the same type.".format(
                u.dtype, e.dtype
            )
    # deal with scalar features.
    expand_u, expand_e = False, False
    if use_u and u.dim() == 1:
        u = torch.unsqueeze(u, -1)
        expand_u = True
    if use_e and e.dim() == 1:
        e = torch.unsqueeze(e, -1)
        expand_e = True

    device = u.device if use_u else e.device
    dtype = u.dtype if use_u else e.dtype
    u_shp = u.shape if use_u else (0,)
    e_shp = e.shape if use_e else (0,)
    v_out_dim = (
        g._graph._CAPI_GetNumCols() if u_target == 0 else g._graph._CAPI_GetNumRows()
    )
    v_shp = (v_out_dim,) + infer_broadcast_shape(op, u_shp[1:], e_shp[1:])
    v = torch.zeros(v_shp, dtype=dtype, device=device)
    use_cmp = reduce_op in ["max", "min"]
    arg_u, arg_e = None, None
    if use_cmp:
        if use_u:
            arg_u = torch.zeros(v_shp, dtype=torch.int64, device=device)
        if use_e:
            arg_e = torch.zeros(v_shp, dtype=torch.int64, device=device)
    if g._graph._CAPI_GetNumEdges() > 0:
        g._graph._CAPI_SpMM(op, reduce_op, u, e, v, arg_u, arg_e, u_target, on_format)
    # To deal with scalar node/edge features.
    if (expand_u or not use_u) and (expand_e or not use_e):
        v = torch.squeeze(v, -1)
    if expand_u and use_cmp:
        arg_u = torch.squeeze(arg_u, -1)
    if expand_e and use_cmp:
        arg_e = torch.squeeze(arg_e, -1)
    # return v, (arg_u, arg_e)
    return v


def _attach_zerodeg_note(docstring, reducer):
    note1 = """
    The {} function will return zero for nodes with no incoming messages.""".format(
        reducer
    )
    note2 = """
    This is implemented by replacing all {} values to zero.
    """.format(
        "infinity" if reducer == "min" else "negative infinity"
    )

    docstring = docstring + note1
    if reducer in ("min", "max"):
        docstring = docstring + note2
    return docstring


def _gen_spmm_func(binary_op, reduce_op):
    name = "u_{}_e_{}".format(binary_op, reduce_op)
    docstring = """Generalized SpMM function.
    It fuses two steps into one kernel.

    1. Computes messages by {} source node and edge features.
    2. Aggregate the messages by {} as the features on destination nodes.

    Parameters
    ----------
    g : DGLGraph
        The input graph
    x : tensor
        The source node features.
    y : tensor
        The edge features.

    Returns
    -------
    tensor
        The result tensor.

    Notes
    -----
    This function supports autograd (computing input gradients given the output gradient). If the
    feature shape of two input operands do not match, we first broadcasts the features to a unified
    shape (note that the memory usage will not increase accordingly) and then performs the operation.

    Broadcasting follows NumPy semantics. Please see
    https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
    for more details about the NumPy broadcasting semantics.
    """.format(
        binary_op, reduce_op
    )
    docstring = _attach_zerodeg_note(docstring, reduce_op)

    def func(g, x, y, on_format=_CSC):
        return gspmm(g, binary_op, reduce_op, x, y, on_format)

    func.__name__ = name
    func.__doc__ = docstring
    return func


def _gen_copy_u_func(binary_op, reduce_op):
    name = "{}_{}".format(binary_op, reduce_op)
    binary_str = {
        "copy_u": "It copies src node feature to edge as the message.",
        "copy_e": "It regards edge feature as message.",
    }
    x_str = {"copy_u": "source node", "copy_e": "edge"}

    def docstring(binary_op):
        return _attach_zerodeg_note(
            """Generalized SpMM function. {}
    Then aggregates the message by {} on destination nodes.

    Parameters
    ----------
    g : DGLGraph
        The input graph
    x : tensor
        The {} features.

    Returns
    -------
    tensor
        The result tensor.

    Notes
    -----
    This function supports autograd (computing input gradients given the output gradient).
    """.format(
                binary_str[binary_op], reduce_op, x_str[binary_op]
            ),
            reduce_op,
        )

    def func(g, x, x_target, on_format=_CSC):
        return gspmm(g, "copy_lhs", reduce_op, x, None, x_target, on_format)

    func.__name__ = name
    func.__doc__ = docstring(binary_op)
    return func


def _gen_copy_e_func(binary_op, reduce_op):
    name = "{}_{}".format(binary_op, reduce_op)
    binary_str = {
        "copy_u": "It copies src node feature to edge as the message.",
        "copy_e": "It regards edge feature as message.",
    }
    x_str = {"copy_u": "source node", "copy_e": "edge"}

    def docstring(binary_op):
        return _attach_zerodeg_note(
            """Generalized SpMM function. {}
    Then aggregates the message by {} on destination nodes.

    Parameters
    ----------
    g : DGLGraph
        The input graph
    x : tensor
        The {} features.

    Returns
    -------
    tensor
        The result tensor.

    Notes
    -----
    This function supports autograd (computing input gradients given the output gradient).
    """.format(
                binary_str[binary_op], reduce_op, x_str[binary_op]
            ),
            reduce_op,
        )

    def func(g, x, reduce_target, on_format=_CSC):
        lhs_target = 2 - reduce_target
        return gspmm(g, "copy_rhs", reduce_op, None, x, lhs_target, on_format)

    func.__name__ = name
    func.__doc__ = docstring(binary_op)
    return func


def _register_spmm_func():
    """Register spmm functions

    - Binary operation plus reduction between node and edge: u/v_[]_e_[]
    - Copy u plus reduction: copy_u/v_[]
    - Copy e plus reduction: copy_e_[]
    """
    for binary_op in ["add", "sub", "mul", "div", "copy_u", "copy_e"]:
        for reduce_op in ["sum", "max", "min"]:
            if binary_op.startswith("copy_u"):
                func = _gen_copy_u_func(binary_op, reduce_op)
            elif binary_op.startswith("copy_e"):
                func = _gen_copy_e_func(binary_op, reduce_op)
            else:
                func = _gen_spmm_func(binary_op, reduce_op)
            setattr(sys.modules[__name__], func.__name__, func)
            __all__.append(func.__name__)


_register_spmm_func()
