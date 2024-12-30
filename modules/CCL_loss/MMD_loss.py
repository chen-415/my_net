import torch
import torch.nn as nn

import torch
import torch.nn as nn


class GMMDLoss(nn.Module):
    def __init__(self, a, b, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(GMMDLoss, self).__init__()
        self.a = a
        self.b = b
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma

    def gaussian_kernel(self, source, target):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if self.fix_sigma:
            bandwidth = self.fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul ** i)
                          for i in range(self.kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = torch.dot(delta, delta.T)
        return loss

    def forward(self, fog_pred, fog_gt, dehazed_pred, m=0.1, a=0.1, b=0.01):
        # 计算 MMD(f(x), y)
        mmd_fog_pred = self.gaussian_kernel(fog_pred, fog_gt)

        # 计算 MMD(g(x), y)
        mmd_dehazed_pred = self.gaussian_kernel(dehazed_pred, fog_gt)
        # 构造损失函数 L
        loss = a * mmd_fog_pred + b * torch.max(torch.tensor(0.0), m - mmd_dehazed_pred)

        return loss


#在这里 第2维一定要相同，否则报错
source1 = torch.rand(16,16)
source2 = torch.rand(16,16)# 可以理解为源域有64个14维数据
target = torch.rand(16,16)  # 可以理解为源域有32个14维数据
# print(target)

MMD = GMMDLoss(a=0.1, b=0.01)
a = MMD(fog_pred=source1, fog_gt=target, dehazed_pred=source2)
print(a)

