import gs
import torch
import time
import numpy as np

tensor1 = torch.tensor([1, 2, 3, 4, 5] + [5, 6, 7, 8, 9]).long().cuda()
tensor2 = torch.tensor([0, 0, 2, 1, 2] + [1, 2, 3, 4, 5]).long().cuda()
#tensor = torch.arange(300).long().cuda()
offset_ptr1 = torch.tensor([0, 5, 10]).long().cuda()
offset_ptr2 = torch.tensor([0, 5, 10]).long().cuda()

time_list = []
'''
for i in range(100):
    begin = time.time()
    torch.ops.gs_ops.BatchUnique(
        [tensor1, tensor2], [offset_ptr1, offset_ptr2], 2)
    end = time.time()
    time_list.append(end - begin)

print(np.mean(time_list[10:]) * 1000)
'''

for i in torch.ops.gs_ops.BatchUnique([tensor1, tensor2], [offset_ptr1, offset_ptr2], 2):
    print(i.numel(), i)
    print()

for i in torch.ops.gs_ops.BatchRelabel([tensor1, tensor2], [offset_ptr1, offset_ptr2], 2):
    print(i)
