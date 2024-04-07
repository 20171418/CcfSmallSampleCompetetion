import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        def forward(self, input, target):
            # 当处理序列输入，如在NLP任务中时，输入的形状通常为[N, C, L]，其中N是批次大小，C是类别数，L是序列长度
            # 如果输入维度大于2，需要将其变形
            if input.dim() > 2:
                # 将输入维度从[N, C, L]变为[N, C, L]
                input = input.view(input.size(0), input.size(1), -1)
                # 交换维度，从[N, C, L]变为[N, L, C]
                input = input.transpose(1, 2)
                # 再将输入维度从[N, L, C]变为[N*L, C]
                input = input.contiguous().view(-1, input.size(2))

            # 调整target的形状为[N*L, 1]
            target = target.view(-1, 1)

            # 计算pt，即预测概率
            # 对input进行softmax操作，然后取对数得到logpt
            logpt = F.log_softmax(input)
            # gather函数用于选择目标类别的对数概率，target的每个元素表示选择哪个类别的对数概率
            logpt = logpt.gather(1, target)
            logpt = logpt.view(-1)
            # 用Variable将logpt包裹起来，使得它具有自动求导功能。exp函数对logpt进行指数运算，得到pt
            pt = Variable(logpt.data.exp())

            # 如果用户设定了alpha参数，即类别权重
            if self.alpha is not None:
                # 确保alpha与input的数据类型一致
                if self.alpha.type() != input.data.type():
                    self.alpha = self.alpha.type_as(input.data)
                # 根据target，选择对应的类别权重
                at = self.alpha.gather(0, target.data.view(-1))
                # 计算alpha * logpt
                logpt = logpt * Variable(at)

            # 根据Focal Loss的公式计算损失
            loss = -1 * (1 - pt) ** self.gamma * logpt

            # 如果size_average为True，则返回所有元素的损失的平均值，否则返回所有元素损失的总和
            if self.size_average:
                return loss.mean()
            else:
                return loss.sum()

