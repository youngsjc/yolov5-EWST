class ErfReLU(nn.Module):
    def __init__(self):
        super(ErfReLU, self).__init__()

    def forward(self, x):
        # 计算 ReLU 和 erf 的组合
        return torch.erf(torch.relu(x))

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.relu = ErfReLU()
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sig(y).view(b, c, 1, 1)
        return x * y

class TWIST_SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16, trivial_threshold=0.001):
        super(TWIST_SEBlock, self).__init__()
        self.se = SEBlock(in_channels, reduction)
        self.trivial_threshold = trivial_threshold

    def forward(self, x):
        b, c, _, _ = x.size()
        weight = self.se.avgpool(x).view(b, c)
        y = self.se.fc1(weight)
        y = self.se.relu(y)
        y = self.se.fc2(y)
        weight = self.se.sig(y).view(b, c, 1, 1)
        #print('w',weight)
        # 排序，得到排序后的索引
        _, indices = torch.sort(weight, dim=1, descending=False)
        # 通过索引重新排序权重，同时维护通道顺序
        sorted_weight = torch.gather(weight, dim=1, index=indices)
        # 计算小于阈值的权重之和
        trivial_weight_sum = torch.sum(torch.masked_select(sorted_weight, sorted_weight < self.trivial_threshold))
        if trivial_weight_sum == 0:
            # 如果小于阈值的权重之和为0，则将所有权重除以权重均值
            weight = weight / weight.mean()
        else:
            # 计算小于阈值的权重的平方和
            trivial_weight_square_sum = torch.sum(
                torch.masked_select(sorted_weight ** 2, sorted_weight < self.trivial_threshold))
            # 计算权重
            weight = torch.where(weight < self.trivial_threshold, trivial_weight_square_sum / trivial_weight_sum,
                                 weight)
            # 将所有权重除以权重均值
            weight = weight / weight.mean()
        return x * weight
