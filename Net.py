import torch
import torch.nn as nn
import torch.nn.functional as F
from AdaIN import *


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.extra = nn.Sequential()
        if in_ch != out_ch:
            self.extra = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # shortcut
        out = self.extra(x) + out
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=11):
        super(ResNet, self).__init__()

        self.in_planes = 128

        self.conv1 = nn.Conv2d(4, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.layer1_1 = self._make_layer(block, 128, num_blocks[0], stride=1)
        self.layer2_1 = self._make_layer(block, 256, num_blocks[1], stride=1)
        self.layer3_1 = self._make_layer(block, 512, num_blocks[2], stride=1)

        self.in_planes = 128
        self.layer1_2 = self._make_layer(block, 128, num_blocks[5], stride=1)
        self.layer2_2 = self._make_layer(block, 256, num_blocks[6], stride=1)
        self.layer3_2 = self._make_layer(block, 512, num_blocks[7], stride=1)

        self.q_linear = nn.Conv2d(512, 512, kernel_size=1, stride=1)
        self.k_linear = nn.Conv2d(512, 512, kernel_size=1, stride=1)
        self.v1_linear = nn.Conv2d(512, 512, kernel_size=1, stride=1)

        self.softmax = nn.Softmax(dim=-1)
        self.L1_mean = nn.L1Loss(reduction='mean')
        self.MSE = nn.MSELoss(size_average=None, reduce=None, reduction='mean')

        self.linear = nn.Linear(1024, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, y, labels, phase):
        x_1 = F.relu(self.bn1(self.conv1(x)))
        y_1 = F.relu(self.bn2(self.conv2(y)))

        # 单模态特征提取
        f_x_1 = self.layer1_1(x_1)
        f_y_1 = self.layer1_2(y_1)
        f_y_1 = Downsample(f_x_1, f_y_1)

        f_x_2 = self.layer2_1(f_x_1)
        f_y_2 = self.layer2_2(f_y_1)
        f_y_2 = Downsample(f_x_2, f_y_2)

        f_x_3 = self.layer3_1(f_x_2)
        f_y_3 = self.layer3_2(f_y_2)
        f_y_3 = Downsample(f_x_3, f_y_3)

        ms_prototype = prototype_generation(f_x_3, labels, num_classes=11)
        pan_prototype = prototype_generation(f_y_3, labels, num_classes=11)

        # MS的中间特征 AdaIN:先内容特征后风格特征
        f_m_a = adaptive_instance_normalization(ms_prototype, pan_prototype)

        # PAN的中间特征:与PAN的结构相同，但风格与MS相似
        f_p_a = adaptive_instance_normalization(pan_prototype, ms_prototype)

        # PAN特征解耦
        f_p_a_Q = self.q_linear(f_p_a)
        f_p_K = self.k_linear(pan_prototype)
        f_p_V1 = self.v1_linear(pan_prototype)

        b1, c1, h1, w1 = f_p_a_Q.size()
        f_p_a_Q = f_p_a_Q.view(b1, c1, h1 * w1)
        f_p_K = f_p_K.view(b1, c1, h1 * w1).transpose(1, 2)
        f_p_V1 = f_p_V1.view(b1, c1, h1 * w1)

        pan_attn = self.softmax(torch.matmul(f_p_a_Q, f_p_K))

        ms_related = torch.matmul(pan_attn, f_p_V1).view(b1, c1, h1, w1)
        ones_matrix = torch.ones_like(pan_attn)
        ms_irrelated = torch.matmul(torch.sub(ones_matrix, pan_attn), f_p_V1).view(b1, c1, h1, w1)

        # MS特征解耦
        f_m_a_Q = self.q_linear(f_m_a)
        f_m_K = self.k_linear(ms_prototype)
        f_m_V1 = self.v1_linear(ms_prototype)

        b2, c2, h2, w2 = f_m_a_Q.size()
        f_m_a_Q = f_m_a_Q.view(b2, c2, h2 * w2)
        f_m_K = f_m_K.view(b2, c2, h2 * w2).transpose(1, 2)
        f_m_V1 = f_m_V1.view(b2, c2, h2 * w2)

        ms_attn = self.softmax(torch.matmul(f_m_a_Q, f_m_K))

        pan_ralated = torch.matmul(ms_attn, f_m_V1).view(b2, c2, h2, w2)
        ones_matrix1 = torch.ones_like(ms_attn)
        pan_irralated = torch.matmul(torch.sub(ones_matrix1, ms_attn), f_m_V1).view(b2, c2, h2, w2)

        f_p_ = adaptive_instance_normalization(pan_prototype, ms_irrelated)
        f_m_ = adaptive_instance_normalization(ms_prototype, pan_irralated)

        # 特征相互转换 + PCL
        out = []
        if phase == 'train':
            # MS特征转换为PAN特征：MS原型特征与PAN原型特征引入ms_irralated的风格的结果做loss
            ms_to_pan_loss = 0.5*self.L1_mean(ms_related, ms_prototype) + 0.5*self.L1_mean(ms_prototype, f_p_)
            # PAN特征转换为MS特征:PAN原型特征与MS原型特征引入pan_irralated的风格的结果做loss
            pan_to_ms_loss = 0.5*self.L1_mean(pan_ralated, pan_prototype) + 0.5*self.L1_mean(pan_prototype, f_m_)
            PCL = contrastive_learning(ms_prototype, pan_prototype)
            total_loss = ms_to_pan_loss + pan_to_ms_loss + PCL
            out.append(total_loss)

        f_x_4 = F.adaptive_avg_pool2d(f_x_3, [1, 1])
        f_y_4 = F.adaptive_avg_pool2d(f_y_3, [1, 1])
        rel = torch.cat([f_x_4, f_y_4], dim=1)
        rel = rel.view(rel.size(0), -1)
        rel = self.linear(rel)
        out.append(rel)
        return out


# Prototype Contrastive Learning Loss
# def prototype_generation(features, labels, num_classes):
#     # 假设你有一个样本特征张量 features，维度为（b, c, h, w）
#     # 假设你有一个样本类别张量 labels，维度为（b）
#     # 假设类别数量为 num_classes
#     b, c, h, w = features.size()
#     # 初始化类别特征总和和样本数量
#     class_sums = torch.zeros(num_classes, c, h, w).cuda()
#     class_counts = torch.zeros(num_classes).cuda()
#
#     # 遍历样本特征张量和类别张量
#     for i in range(b):
#         feat = features[i]
#         label = labels[i]
#
#         # 将样本特征累加到相应类别的特征总和上
#         class_sums[label] += feat
#         # 增加相应类别的样本数量
#         class_counts[label] += 1
#
#     # 计算每个类别的特征平均值
#     print(class_counts)
#     class_prototype = class_sums / class_counts.view(-1, 1, 1, 1)
#     return class_prototype

def prototype_generation(features, labels, num_classes):
    b, c, h, w = features.size()
    class_sums = torch.zeros(num_classes, c, h, w).cuda()
    class_counts = torch.zeros(num_classes).cuda()

    for i in range(b):
        feat = features[i]
        label = labels[i]

        class_sums[label] += feat
        class_counts[label] += 1

    class_prototype = torch.zeros(num_classes, c, h, w).cuda()

    for label in range(num_classes):
        if class_counts[label] > 0:
            class_prototype[label] = class_sums[label] / class_counts[label]

    return class_prototype

# def prototype_generation(features, labels, num_classes):
#     # 假设你有一个样本特征张量 features，维度为（b, c, h, w）
#     # 假设你有一个样本类别张量 labels，维度为（b）
#     # 假设类别数量为 num_classes
#     b, c, h, w = features.size()
#     # 初始化类别特征总和和样本距离之和
#     class_sums = torch.zeros(num_classes, c, h, w).cuda()
#     sample_distances = torch.zeros(b).cuda()
#
#     # 遍历样本特征张量和类别张量
#     for i in range(b):
#         feat = features[i]
#         label = labels[i]
#         distance_sum = 0.0
#
#         # 计算相同类别中每个样本与其他样本之间的距离
#         for j in range(b):
#             if labels[j] == label:
#                 distance_sum += euclidean_dist(feat, features[j])
#
#         # 保存每个样本的距离之和
#         sample_distances[i] = compute_weights(distance_sum, 5)
#
#     # 归一化权重
#     weights = torch.div(sample_distances, torch.sum(sample_distances) + 1e-12)
#
#     # 遍历样本特征张量和类别张量
#     for i in range(b):
#         feat = features[i]
#         label = labels[i]
#         # 将样本特征乘以权重并累加到相应类别的特征总和上
#         class_sums[label] += feat * weights[i]
#
#     return class_sums


def contrastive_learning(ms_prototype, pan_prototype):
    ms_prototype = normalize(ms_prototype)
    pan_prototype = normalize(pan_prototype)
    sim = euclidean_dist(ms_prototype, pan_prototype)
    loss = -torch.log_softmax(sim + 1e-6, dim=-1)
    return torch.mean(loss)


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-6)
    return x


def euclidean_dist(x, y):
    dist = torch.norm(x - y, dim=-1)
    return dist

def compute_weights(distances, c):
    # 根据IMQ函数计算权重
    return 1 / (torch.sqrt(distances ** 2 + c ** 2) + 1e-6)


def ResNet18():
    return ResNet(ResBlock, [2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

