import gs
import torch
import time
import numpy as np

data = torch.tensor([20, 21, 22, 21, 22] + [25, 26, 27, 22, 22]).long().cuda()
offset_ptr = torch.tensor([0, 5, 10]).long().cuda()
key_tensor = torch.tensor([0 for _ in range(5)] +
                          [1 for _ in range(5)]).long().cuda()

for i in torch.ops.gs_ops.BatchRelabelByKey(data, offset_ptr, key_tensor):
    print(i)