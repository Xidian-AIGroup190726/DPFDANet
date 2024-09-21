import torch.nn as nn


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps  # 对输入的图片的每个通道计算标准差
    feat_std = feat_var.sqrt().view(N, C, 1, 1)  # 标准差的平方得到方差
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    #   [batch,channel,H,W]
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def Downsample(x, y):
    _, _, h1, w1 = x.size()
    downsample = nn.AdaptiveAvgPool2d(output_size=(h1, w1))
    result = downsample(y)
    return result