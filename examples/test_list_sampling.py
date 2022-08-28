import gs
import torch

data = torch.tensor([0, 1, 2, 3, 4, 5]).long().cuda()

select, index = torch.ops.gs_ops.list_sampling(data, 20, True)

print(select)
print(index)