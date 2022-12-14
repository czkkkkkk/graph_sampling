import gs
import dgl
import torch

_CSR = 4
_CSC = 2
_COO = 1

A = gs.Graph(False)
indptr = torch.tensor([
    0, 5, 10, 15, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    20, 20
]).long()
indices = torch.Tensor(
    [1, 7, 18, 5, 10, 0, 14, 11, 4, 2, 18, 9, 14, 4, 12, 0, 18, 8, 6,
     12]).long()

print(indptr.numel(), indptr)
print(indices.numel(), indices)

print("\nTest Loading")
A._CAPI_full_load_csc(indptr.cuda(), indices.cuda())
coo_row, coo_col = A._CAPI_get_coo()
print(coo_row)
print(coo_col)

print("\nTest CSC Slicing on row")
seeds = torch.Tensor([4, 8, 12, 16]).long().cuda()
subA = A._CAPI_full_slicing(seeds, 1, _CSC)
coo_row, coo_col = subA._CAPI_get_coo()
print(coo_row)
print(coo_col)

print("\nTest COO Slicing on row")
seeds = torch.Tensor([4, 8, 12, 16]).long().cuda()
subA = A._CAPI_full_slicing(seeds, 1, _COO)
coo_row, coo_col = subA._CAPI_get_coo()
print(coo_row)
print(coo_col)

print("\nTest CSR Slicing on row")
seeds = torch.Tensor([4, 8, 12, 16]).long().cuda()
subA = A._CAPI_full_slicing(seeds, 1, _CSR)
coo_row, coo_col = subA._CAPI_get_coo()
print(coo_row)
print(coo_col)

print("\nTest _CSC Samping num_picks=3")
subA = A._CAPI_full_sampling(0, 3, False, _CSC)
coo_row, coo_col = subA._CAPI_get_coo()
print(coo_row)
print(coo_col)

print("\nTest _CSC Samping  num_picks=6")
subA = A._CAPI_full_sampling(0, 6, False, _CSC)
coo_row, coo_col = subA._CAPI_get_coo()
print(coo_row)
print(coo_col)

print("\nTest _CSC Samping Probs num_picks=3")
edata = torch.ones(20).float().cuda()
subA = A._CAPI_full_sampling_with_probs(0, edata, 3, False, _CSC)
coo_row, coo_col = subA._CAPI_get_coo()
print(coo_row)
print(coo_col)

print("\nTest _CSC Samping Probs num_picks=6")
edata = torch.ones(20).float().cuda()
subA = A._CAPI_full_sampling_with_probs(0, edata, 6, False, _CSC)
coo_row, coo_col = subA._CAPI_get_coo()
print(coo_row)
print(coo_col)

print("\nTest col Sum on COO")
edata = torch.ones(20).float().cuda()
A._CAPI_set_data(edata, 'col')
sum = A._CAPI_full_sum(0, 1, _COO)
print(sum)

print("\nTest col Sum on CSC")
edata = torch.ones(20).float().cuda()
A._CAPI_set_data(edata, 'col')
sum = A._CAPI_full_sum(0, 1, _CSC)
print(sum)

print("\nTest col divide on COO")
edata = torch.ones(20).float().cuda()
divisor = torch.ones(20).float().cuda() * 2
A._CAPI_set_data(edata, 'col')
subA = A._CAPI_full_divide(divisor, 0, _COO)
print(subA._CAPI_get_data('col'))

print("\nTest col divide on CSC")
edata = torch.ones(20).float().cuda()
divisor = torch.ones(20).float().cuda() * 2
A._CAPI_set_data(edata, 'col')
subA = A._CAPI_full_divide(divisor, 0, _CSC)
print(subA._CAPI_get_data('col'))

print("\nTest col normalize on COO")
edata = torch.ones(20).float().cuda() * 2
A._CAPI_set_data(edata, 'col')
subA = A._CAPI_full_normalize(0, _COO)  # BUG Here
print(subA._CAPI_get_data('col'))

print("\nTest col normalize on CSC")
edata = torch.ones(20).float().cuda() * 2
A._CAPI_set_data(edata, 'col')
subA = A._CAPI_full_normalize(0, _CSC)
print(subA._CAPI_get_data('col'))

print("\nTest col SDDMM on COO")
lhs = torch.ones(20).float().cuda() * 3
rhs = torch.ones(20).float().cuda() * 2
out = torch.zeros(20).float().cuda()
A._CAPI_full_sddmm("add", lhs, rhs, out, 0, 2, _COO)
print(out)

print("\nTest col SDDMM on CSC")
lhs = torch.ones(20).float().cuda() * 3
rhs = torch.ones(20).float().cuda() * 2
out = torch.zeros(20).float().cuda()
A._CAPI_full_sddmm("add", lhs, rhs, out, 0, 2, _CSC)
print(out)

print("\nTest col SDDMM on CSR")
lhs = torch.ones(20).float().cuda() * 3
rhs = torch.ones(20).float().cuda() * 2
out = torch.zeros(20).float().cuda()
A._CAPI_full_sddmm("add", lhs, rhs, out, 0, 2, _CSR)
print(out)
