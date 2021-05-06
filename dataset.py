import numpy as np
import open3d as o3d
import torch
from torchvision.transforms import Compose
from modelnet40.datasets import ModelNetHdf
from modelnet40.datasets import get_transforms
from copy import deepcopy
from torch.utils import data
from descriptor import FPFH, SHOT, BallSurface
from utils import square_distance, ransac_pose_estimation, chamfer_distance, pc_normalize


def get_pc(pts, normal=None, color=None):
    pc = o3d.PointCloud()
    pc.points = o3d.Vector3dVector(pts)
    if normal is not None:
        pc.normals = o3d.Vector3dVector(normal)
    if color is not None:
        pc.colors = o3d.Vector3dVector([color]*pts.shape[0])
    return pc


class Modelnet40Pair(data.Dataset):
    def __init__(self, path, support_r=0.3, descriptor="FPFH", mode="train", noise_type="crop", categories=None, corr_pts_dis_thresh=0.06, get_raw_and_T=False, get_np=False):
        super(Modelnet40Pair, self).__init__()
        train_transforms, test_transforms = get_transforms(noise_type=noise_type)
        train_transforms, test_transforms = Compose(train_transforms), Compose(test_transforms)

        tr = train_transforms if mode == "train" else test_transforms
        self.data = ModelNetHdf(path, subset=mode, transform=tr, categories=categories)
        self.support_r = support_r
        self.dis_thresh = corr_pts_dis_thresh
        if descriptor == "FPFH":
            self.descriptor = FPFH
        if descriptor == "SHOT":
            self.descriptor = SHOT
        if descriptor == "BS":
            self.descriptor = BallSurface
        self.get_raw_and_T = get_raw_and_T
        self.get_np = get_np

    def __len__(self):
        # return 100
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        source_pts, source_normal = x["points_src"][:, :3], x["points_src"][:, 3:]
        target_pts, target_normal = x["points_ref"][:, :3], x["points_ref"][:, 3:]
        raw_pts = x["points_raw"][:, :3]
        idx = x["idx"]
        T = np.concatenate([x["transform_gt"], np.array([[0, 0, 0, 1]])], axis=0)
        source_pc = get_pc(source_pts, source_normal, [1, 0.706, 0])
        target_pc = get_pc(target_pts, target_normal, [0, 0.651, 0.929])

        # source_f, target_f = self.descriptor.get(source_pc, self.support_r), self.descriptor.get(target_pc, self.support_r)
        source_f, target_f = self.descriptor.get(source_pc, self.support_r, 1/9), self.descriptor.get(target_pc, self.support_r, 1/9)
        f_dis = square_distance(torch.Tensor(source_f).unsqueeze(0), torch.Tensor(target_f).unsqueeze(0))[0]
        source_to_target_min_idx, target_to_source_min_idx = f_dis.min(dim=1)[1].numpy(), f_dis.min(dim=0)[1].numpy()

        rotated_source_pts = np.asarray(deepcopy(source_pc).transform(T).points)
        source_corr_pts, target_corr_pts = target_pts[source_to_target_min_idx], rotated_source_pts[target_to_source_min_idx]
        source_pts_to_target_pts_dis = np.sqrt(np.sum((rotated_source_pts-source_corr_pts)**2, axis=1))
        target_pts_to_source_pts_dis = np.sqrt(np.sum((target_pts-target_corr_pts)**2, axis=1))

        dis_thresh = self.dis_thresh
        source_key_pts_idx, target_key_pts_idx = np.nonzero(source_pts_to_target_pts_dis <= dis_thresh)[0], np.nonzero(target_pts_to_source_pts_dis <= dis_thresh)[0]
        source_key_label, target_key_label = np.zeros((source_pts.shape[0], )), np.zeros((target_pts.shape[0], ))
        source_key_label[source_key_pts_idx], target_key_label[target_key_pts_idx] = 1, 1

        # 画出关键点看看
        source_pc = get_pc(source_pts, None, [1, 0.706, 0])
        target_pc = get_pc(target_pts, None, [0, 0.651, 0.929])
        source_color, target_color = np.asarray(source_pc.colors), np.asarray(target_pc.colors)
        source_color[source_key_pts_idx] = np.array([1, 0, 0])
        target_color[target_key_pts_idx] = np.array([1, 0, 0])
        # o3d.draw_geometries([source_pc, target_pc], window_name="test", width=1000, height=800)

        if self.get_np:
            return source_pts, source_normal, source_f, source_key_pts_idx, \
                   target_pts, target_normal, target_f, target_key_pts_idx, \
                   source_to_target_min_idx, target_to_source_min_idx, raw_pts, T
        source_pts, target_pts = pc_normalize(source_pts), pc_normalize(target_pts)
        if self.get_raw_and_T:
            return torch.Tensor(source_pts), torch.Tensor(source_normal), torch.Tensor(source_f), torch.Tensor(source_key_label), \
                   torch.Tensor(target_pts), torch.Tensor(target_normal), torch.Tensor(target_f), torch.Tensor(target_key_label), \
                   torch.Tensor(raw_pts), torch.Tensor(T)
        else:
            return torch.Tensor(source_pts), torch.Tensor(source_normal), torch.Tensor(source_f), torch.Tensor(source_key_label), \
                   torch.Tensor(target_pts), torch.Tensor(target_normal), torch.Tensor(target_f), torch.Tensor(target_key_label), \
                   torch.LongTensor(source_to_target_min_idx), torch.LongTensor(target_to_source_min_idx)


if __name__ == '__main__':
    path = "E:/modelnet40_ply_hdf5_2048"
    modelnet_set = Modelnet40Pair(path, mode="test", descriptor="BS",
                                  categories=["airplane", "person"], corr_pts_dis_thresh=0.05, get_np=True, support_r=0.4)
    print(len(modelnet_set))
    for i in range(len(modelnet_set)):
        source_pts, source_normal, source_f, source_key_pts_idx, target_pts, target_normal, target_f, target_key_pts_idx, raw_pts, T = modelnet_set[i]

        source_pc = get_pc(source_pts, source_normal, [1, 0.706, 0])
        target_pc = get_pc(target_pts, target_normal, [0, 0.651, 0.929])
        # 重叠部分配准
        source_overlap_pc, target_overlap_pc = o3d.PointCloud(), o3d.PointCloud()
        source_overlap_pc.points, target_overlap_pc.points = o3d.Vector3dVector(np.asarray(source_pc.points)[source_key_pts_idx]), o3d.Vector3dVector(np.asarray(target_pc.points)[target_key_pts_idx])

        ransac_T = ransac_pose_estimation(
            source_pts[source_key_pts_idx], target_pts[target_key_pts_idx],
            source_f[source_key_pts_idx], target_f[target_key_pts_idx],
            distance_threshold=0.06, max_iter=50000, max_valid=1000
        )
        icp_result = o3d.registration_icp(source_overlap_pc, target_overlap_pc, 0.04, init=ransac_T)
        icp_T = icp_result.transformation

        # 评估
        chamfer_dist = chamfer_distance(source_pts, target_pts, raw_pts, icp_T, T)
        print(chamfer_dist)
        o3d.draw_geometries([source_pc.transform(T), target_pc], window_name="test registration", width=1000, height=800)
