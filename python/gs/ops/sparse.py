import torch

from .sparse_ops import _gsddmm, _gspmm
from ..format import _COO, _CSC, _CSR, _DCSC, _DCSR


def _reduce_grad(grad, shape):
    """Reduce gradient on the broadcast dimension
    If there is broadcast in forward pass, gradients need to be reduced on
    broadcast dimension. This function checks the input tensor shape and
    gradient shape and perform the reduction.

    Parameters
    ----------
    grad: Tensor
        Gradient tensor
    shape: tuple
        Shape of input tensor

    Returns
    -------
    Tensor
    """
    grad_shape = grad.shape[1:]
    in_shape = shape[1:]
    if in_shape == grad_shape:
        # no need to reduce
        return grad
    num_to_squeeze = len(grad_shape) - len(in_shape)
    # pad inshape
    in_shape = (1,) * num_to_squeeze + in_shape
    reduce_idx = torch.nonzero(
        torch.tensor(grad_shape) - torch.tensor(in_shape), as_tuple=False
    )
    reduce_idx += 1  # skip batch dim
    if len(reduce_idx) > 0:
        grad = grad.sum(dim=tuple(reduce_idx), keepdim=True)
    return grad.view(-1, *shape[1:])


def _need_reduce_last_dim(ufeat, efeat):
    """Indicates whether to reduce the last dimension on edges
    in the backward pass of spmm,
    if so, use dot instead of mul."""
    if ufeat is None or efeat is None:
        return False
    ushp = ufeat.shape
    eshp = efeat.shape
    return ushp[1:-1] == eshp[1:-1] and eshp[-1] == 1 and ushp[-1] > 1


def _expand(x, shape):
    return x.expand(-1, *shape)


def spmm_cache_X(binary_op, reduce_op, req_grad_X, req_grad_Y):
    """Rules to identify whether to cache X in SpMM forward stage."""
    if binary_op != "copy_lhs" and req_grad_Y:
        if reduce_op == "sum":
            return True
        else:
            if binary_op == "mul":
                return True
    return False


def spmm_cache_Y(binary_op, reduce_op, req_grad_X, req_grad_Y):
    """Rules to identify whether to cache Y in SpMM forward stage."""
    if binary_op != "copy_rhs" and req_grad_X:
        if reduce_op == "sum":
            if binary_op in ["mul", "add"]:
                return True
        else:
            if binary_op == "mul":
                return True
    return False


def spmm_cache_argX(binary_op, reduce_op, req_grad_X, req_grad_Y):
    """Rules to identify whether to cache argX in SpMM forward stage."""
    if req_grad_X or req_grad_Y:
        if reduce_op in ["min", "max"]:
            return True
    return False


def spmm_cache_argY(binary_op, reduce_op, req_grad_X, req_grad_Y):
    """Rules to identify whether to cache argY in SpMM forward stage."""
    if req_grad_X or req_grad_Y:
        if reduce_op in ["min", "max"]:
            return True
    return False


class GSpMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gidx, op, reduce_op, X, Y, on_format):
        out, (argX, argY) = _gspmm(gidx, op, reduce_op, X, Y, on_format)
        reduce_last = _need_reduce_last_dim(X, Y)
        X_shape = X.shape if X is not None else None
        Y_shape = Y.shape if Y is not None else None
        dtype = X.dtype if X is not None else Y.dtype
        device = X.device if X is not None else Y.device
        ctx.backward_cache = (
            gidx,
            op,
            reduce_op,
            X_shape,
            Y_shape,
            dtype,
            device,
            reduce_last,
            on_format,
        )
        req_grad_X = X.requires_grad if X is not None else False
        req_grad_Y = Y.requires_grad if Y is not None else False
        if not spmm_cache_X(op, reduce_op, req_grad_X, req_grad_Y):
            X = None
        if not spmm_cache_Y(op, reduce_op, req_grad_X, req_grad_Y):
            Y = None
        if not spmm_cache_argX(op, reduce_op, req_grad_X, req_grad_Y):
            argX = None
        if not spmm_cache_argY(op, reduce_op, req_grad_X, req_grad_Y):
            argY = None
        ctx.save_for_backward(X, Y, argX, argY)
        return out

    @staticmethod
    def backward(ctx, dZ):
        (
            gidx,
            op,
            reduce_op,
            X_shape,
            Y_shape,
            dtype,
            device,
            reduce_last,
            on_format,
        ) = ctx.backward_cache
        X, Y, argX, argY = ctx.saved_tensors
        if op != "copy_rhs" and ctx.needs_input_grad[3]:
            g_rev = gidx.reverse()
            if reduce_op == "sum":
                if op == "mul":
                    dX = gspmm(g_rev, "mul", "sum", dZ, Y, on_format)
                elif op == "add":
                    dX = gspmm(g_rev, "copy_lhs", "sum", dZ, Y, on_format)
                elif op == "copy_lhs":
                    dX = gspmm(g_rev, "copy_lhs", "sum", dZ, None, on_format)
            else:  # max/min
                dX = torch.zeros(
                    (X_shape[0],) + dZ.shape[1:], dtype=dtype, device=device
                )
                if op == "mul":
                    grad = _expand(Y, dZ.shape[1:]).gather(0, argY.long()) * dZ
                    dX.scatter_add_(0, argX.long(), grad)
                elif op in ["add", "copy_lhs"]:
                    dX.scatter_add_(0, argX.long(), dZ)
            dX = _reduce_grad(dX, X_shape)
        else:  # X has not gradient
            dX = None
        if op != "copy_lhs" and ctx.needs_input_grad[4]:
            if reduce_op == "sum":
                if op == "mul" and reduce_last:
                    dY = gsddmm(gidx, "dot", X, dZ, on_format=on_format)
                elif op == "mul":
                    dY = gsddmm(gidx, "mul", X, dZ, on_format=on_format)
                elif op in ["add", "copy_rhs"]:
                    dY = gsddmm(gidx, "copy_rhs", X, dZ, on_format=on_format)
            else:  # max/min
                dY = torch.zeros(
                    (Y_shape[0],) + dZ.shape[1:], dtype=dtype, device=device
                )
                if op == "mul":
                    grad = _expand(X, dZ.shape[1:]).gather(0, argX.long()) * dZ
                    dY.scatter_add_(0, argY.long(), grad)
                elif op in ["add", "copy_rhs"]:
                    dY.scatter_add_(0, argY.long(), dZ)
            dY = _reduce_grad(dY, Y_shape)
        else:  # Y has no gradient
            dY = None
        return None, None, None, dX, dY


def sddmm_cache_X(op, req_grad_X, req_grad_Y):
    """Rules to identify whether to cache X in SDDMM forward stage."""
    if op in ["mul", "dot"] and req_grad_Y:
        return True
    return False


def sddmm_cache_Y(op, req_grad_X, req_grad_Y):
    """Rules to identify whether to cache Y in SDDMM forward stage."""
    if op in ["mul", "dot"] and req_grad_X:
        return True
    return False


class GSDDMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gidx, op, X, Y, lhs_target, rhs_target, on_format):
        out = _gsddmm(gidx, op, X, Y, lhs_target, rhs_target, on_format)
        X_shape = X.shape if X is not None else None
        Y_shape = Y.shape if Y is not None else None
        ctx.backward_cache = gidx, op, lhs_target, rhs_target, X_shape, Y_shape, on_format
        req_grad_X = X.requires_grad if X is not None else False
        req_grad_Y = Y.requires_grad if Y is not None else False
        if not sddmm_cache_X(op, req_grad_X, req_grad_Y):
            X = None
        if not sddmm_cache_Y(op, req_grad_X, req_grad_Y):
            Y = None
        ctx.save_for_backward(X, Y)
        return out

    @staticmethod
    def backward(ctx, dZ):
        gidx, op, lhs_target, rhs_target, X_shape, Y_shape, on_format = ctx.backward_cache
        X, Y = ctx.saved_tensors
        if op != "copy_rhs" and ctx.needs_input_grad[2]:
            if lhs_target in ["u", "v"]:
                _gidx = gidx if lhs_target == "v" else gidx.reverse()
                rev_format = _COO if on_format == _COO else _CSC
                if op in ["add", "copy_lhs"]:
                    dX = gspmm(_gidx, "copy_rhs", "sum", None, dZ, rev_format)
                else:  # mul, dot
                    if rhs_target == lhs_target:
                        dX = gspmm(_gidx, "copy_rhs", "sum", None, dZ, rev_format) * Y
                    elif rhs_target == "e":
                        dX = gspmm(_gidx, "copy_rhs", "sum", None, dZ * Y, rev_format)
                    else:  # rhs_target = !lhs_target
                        dX = gspmm(_gidx, "mul", "sum", Y, dZ, rev_format)
            else:  # lhs_target == 'e'
                if op in ["add", "copy_lhs"]:
                    dX = dZ
                else:  # mul, dot
                    dX = gsddmm(gidx, "mul", dZ, Y, "e", rhs_target, on_format)
            dX = _reduce_grad(dX, X_shape)
        else:
            dX = None
        if op != "copy_lhs" and ctx.needs_input_grad[3]:
            if rhs_target in ["u", "v"]:
                _gidx = gidx if rhs_target == "v" else gidx.reverse()
                rev_format = _COO if on_format == _COO else _CSC
                if op in ["add", "copy_rhs"]:
                    dY = gspmm(_gidx, "copy_rhs", "sum", None, dZ, rev_format)
                else:  # mul, dot
                    if lhs_target == rhs_target:
                        dY = gspmm(_gidx, "copy_rhs", "sum", None, dZ, rev_format) * X
                    elif lhs_target == "e":
                        dY = gspmm(_gidx, "copy_rhs", "sum", None, dZ * X, rev_format)
                    else:  # rhs_target = !lhs_target
                        dY = gspmm(_gidx, "mul", "sum", X, dZ, rev_format)
            else:
                if op in ["add", "copy_rhs"]:
                    dY = dZ
                else:  # mul, dot
                    dY = gsddmm(gidx, "mul", dZ, X, "e", lhs_target, on_format)
            dY = _reduce_grad(dY, Y_shape)
        else:
            dY = None
        return None, None, dX, dY, None, None


def gspmm(gidx, op, reduce_op, lhs_data, rhs_data, on_format=_CSC):
    return GSpMM.apply(gidx, op, reduce_op, lhs_data, rhs_data, on_format)


def gsddmm(gidx, op, lhs_data, rhs_data, lhs_target='u', rhs_target='v', on_format=_COO):
    return GSDDMM.apply(gidx, op, lhs_data, rhs_data, lhs_target, rhs_target, on_format)
