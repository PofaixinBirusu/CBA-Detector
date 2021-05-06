import torch
from torch import nn
from torch.nn import functional as F
import math
from utils import square_distance, PointNetSetAbstraction, UpSample, PointNetSetAbstractionMsg

cpu, gpu = "cpu", "cuda:0"
device = torch.device(gpu)


class MutiHeadAttention(nn.Module):
    def __init__(self, n_head, in_channel, qk_channel, v_channel, out_channel, mid_channel, feedforward=True):
        super(MutiHeadAttention, self).__init__()
        # mutihead attention
        self.n_head = n_head
        self.qk_channel, self.v_channel = qk_channel, v_channel
        self.WQ = nn.Linear(in_channel, qk_channel*n_head, bias=False)
        self.WK = nn.Linear(in_channel, qk_channel*n_head, bias=False)
        self.WV = nn.Linear(in_channel, v_channel*n_head, bias=False)
        self.linear = nn.Linear(v_channel*n_head, out_channel, bias=False)
        # 不确定要不要仿射变换，先不加试试
        self.norm1 = nn.LayerNorm(out_channel, elementwise_affine=False)
        # feedforward
        self.feedforward = feedforward
        if self.feedforward:
            self.feedforward_layer = nn.Sequential(
                nn.Linear(out_channel, mid_channel, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(mid_channel, out_channel, bias=False)
            )
            self.norm2 = nn.LayerNorm(out_channel, elementwise_affine=False)

    def forward(self, query, key, value, mask=None):
        # q, k, v: batch_size x n x in_channel
        # mask： batch_size x n x n
        batch_size = query.shape[0]
        # batch_size x n x in_channel  -->  batch_size x n x v_channel
        Q = self.WQ(query).view(batch_size, -1, self.n_head, self.qk_channel).transpose(1, 2)  # batch_size x n_head x n x q_channel
        K = self.WK(key).view(batch_size, -1, self.n_head, self.qk_channel).transpose(1, 2)    # batch_size x n_head x n x k_channel
        V = self.WV(value).view(batch_size, -1, self.n_head, self.v_channel).transpose(1, 2)   # batch_size x n_head x n x v_channel
        weight = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.qk_channel)               # batch_size x n_head x n x n
        if mask is None:
            weight = torch.softmax(weight, dim=3)
        else:
            weight = torch.softmax(weight + mask.unsqueeze(1), dim=3)                              # batch_size x n_head x n x v_channel
        # print(weight.dtype, V.dtype, mask.dtype)
        out = torch.matmul(weight, V).transpose(1, 2).contiguous().view(batch_size, -1, self.n_head*self.v_channel)
        out = self.linear(out)                                                                 # batch_size x n x out_channel
        out = self.norm1(query + out)
        if self.feedforward:
            return self.norm2(out + self.feedforward_layer(out))
        else:
            return out


class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
        # downsample 1
        DownSample = PointNetSetAbstraction
        self.ds1 = DownSample(npoint=358, radius=0.2, nsample=64, in_channel=6+0, mlp=[64, 64, 128], group_all=False)
        # downsample 2
        self.ds2 = DownSample(npoint=128, radius=0.4, nsample=64, in_channel=128+3, mlp=[128, 128, 256], group_all=False)
        # downsample 3
        self.ds3 = DownSample(npoint=None, radius=None, nsample=None, in_channel=256+3, mlp=[256, 512, 1024], group_all=True)

        self.overlap_atte = MutiHeadAttention(n_head=4, in_channel=256, qk_channel=64, v_channel=64, out_channel=256, mid_channel=1024)

        self.up3 = UpSample(in_channel=1280, mlp=[256, 256])
        self.up2 = UpSample(in_channel=384, mlp=[256, 128])
        self.up1 = UpSample(in_channel=128+6+0, mlp=[512, 256, 256])

    def forward(self, x):
        # x: batch_size x n x d
        # downsample
        x = x.permute([0, 2, 1])
        l0_points, l0_xyz = x, x[:, :3, :]
        l1_xyz, l1_points = self.ds1(l0_xyz, l0_points)  # batch_size x n x 3, batch_size x n x d
        l2_xyz, l2_points = self.ds2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.ds3(l2_xyz, l2_points)
        # upsample
        l2_points = self.up3(l2_xyz, l3_xyz, l2_points, l3_points)
        # print(l2_points.shape)
        l2_points = l2_points.permute([0, 2, 1])
        l2_points = self.overlap_atte(l2_points, l2_points, l2_points, None)
        l2_points = l2_points.permute([0, 2, 1])
        # print(l2_points.shape)
        l1_points = self.up2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.up1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], dim=1), l1_points)
        return l0_points.permute([0, 2, 1])


class PointNetMsg(nn.Module):
    def __init__(self):
        super(PointNetMsg, self).__init__()
        DownSample = PointNetSetAbstractionMsg
        self.ds1 = DownSample(358, [0.1, 0.2, 0.4], [32, 64, 128], 3+3+162, [[32, 32, 64], [64, 64, 128], [64, 96, 128]], bn=False)
        # downsample 2
        self.ds2 = DownSample(128, [0.4, 0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]], bn=False)
        # downsample 3
        self.ds3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True, bn=False)

        # self.overlap_atte = MutiHeadAttention(n_head=4, in_channel=256, qk_channel=64, v_channel=64, out_channel=256,
        #                                       mid_channel=1024)

        self.up3 = UpSample(in_channel=1024+256+256, mlp=[256, 256], bn=False)
        self.up2 = UpSample(in_channel=256+128+128+64, mlp=[256, 128], bn=False)
        self.up1 = UpSample(in_channel=128 + 6 + 3 +162, mlp=[128, 256], bn=False)

    def forward(self, x):
        # x: batch_size x n x d
        # downsample
        x = x.permute([0, 2, 1])
        l0_points, l0_xyz = x, x[:, :3, :]
        l1_xyz, l1_points = self.ds1(l0_xyz, l0_points)  # batch_size x n x 3, batch_size x n x d
        l2_xyz, l2_points = self.ds2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.ds3(l2_xyz, l2_points)
        # print(l3_points.shape)
        # upsample
        l2_points = self.up3(l2_xyz, l3_xyz, l2_points, l3_points)
        # print(l2_points.shape)
        l2_points = l2_points.permute([0, 2, 1])
        # l2_points = self.overlap_atte(l2_points, l2_points, l2_points, None)
        l2_points = l2_points.permute([0, 2, 1])
        # print(l2_points.shape)
        l1_points = self.up2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.up1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], dim=1), l1_points)
        return l0_points.permute([0, 2, 1])


class Skr(nn.Module):
    def __init__(self):
        super(Skr, self).__init__()
        self.pointnet = PointNetMsg()

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, source_inp, target_inp, s2t_idx, t2s_idx):
        # batch_size x n x 6
        batch_size = source_inp.shape[0]
        inp = torch.cat([source_inp, target_inp], dim=0)
        feature = self.pointnet(inp)
        # batch_size x n x c
        source_f, target_f = feature[:batch_size, :, :], feature[batch_size:, :, :]

        source_corr_f = target_f.contiguous().view(-1, target_f.shape[2])[(s2t_idx+(torch.arange(batch_size)*s2t_idx.shape[1]).contiguous().view(-1, 1)).contiguous().view(-1)].contiguous().view(batch_size, target_f.shape[1], -1)
        target_corr_f = source_f.contiguous().view(-1, source_f.shape[2])[(t2s_idx+(torch.arange(batch_size)*t2s_idx.shape[1]).contiguous().view(-1, 1)).contiguous().view(-1)].contiguous().view(batch_size, source_f.shape[1], -1)

        out = self.fc(torch.cat([torch.cat([source_f, source_corr_f], dim=2), torch.cat([target_f, target_corr_f], dim=2)], dim=0))
        source_key_score, target_key = out[:batch_size, :, :].squeeze(2), out[batch_size:, :, :].squeeze(2)
        return source_key_score, target_key


if __name__ == '__main__':
    device = torch.device("cuda:0")

    net = Skr()
    net.to(device)
    batch_size = 4
    x1, x2 = torch.rand(batch_size, 717, 6).to(device), torch.rand(batch_size, 717, 6).to(device)
    key_pts1, key_pts2 = net(x1, x2)
    print(key_pts1.shape)