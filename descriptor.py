import open3d as o3d
import numpy as np
from pclpy import pcl
import math
from math import cos, sin
import matplotlib.pyplot as plt
import torch


class SHOT:
    def __init__(self, pc, r):
        self.pc, self.r = pc, r

    def get_features(self, inds):
        surface = pcl.PointCloud.PointXYZ(np.asarray(self.pc.points))
        key_points = pcl.PointCloud.PointXYZ(np.asarray(self.pc.points)[inds])
        shot = pcl.features.SHOTEstimation.PointXYZ_Normal_SHOT352_ReferenceFrame()

        # n = surface.compute_normals(self.r/self.factor)
        n = pcl.PointCloud.Normal(np.asarray(self.pc.normals)[inds])

        shot.setSearchSurface(surface)
        shot.setInputCloud(key_points)
        shot.setInputNormals(n)
        shot.setRadiusSearch(self.r)
        out = pcl.PointCloud.SHOT352()
        shot.compute(out)
        return np.asarray(out.descriptor)

    @staticmethod
    def get(pc, r):
        desc = SHOT(pc, r)
        inds = np.arange(np.asarray(pc.points).shape[0])
        return desc.get_features(inds)


class FPFH:
    def __init__(self, pc, r):
        self.pc = pc
        self.r = r

    def get_features(self, inds):
        x = pcl.PointCloud.PointXYZ(np.asarray(self.pc.points))
        key_points = pcl.PointCloud.PointXYZ(np.asarray(self.pc.points)[inds])
        # n = x.compute_normals(self.r)
        n = pcl.PointCloud.Normal(np.asarray(self.pc.normals)[inds])
        # print(type(n))
        # print(n.points)
        fpfh = pcl.features.FPFHEstimation.PointXYZ_Normal_FPFHSignature33()
        fpfh.setInputCloud(key_points)
        fpfh.setSearchSurface(x)
        fpfh.setRadiusSearch(self.r)
        fpfh.setInputNormals(n)
        out = pcl.PointCloud.FPFHSignature33()
        fpfh.compute(out)
        f = np.asarray(out.histogram)
        return f

    @staticmethod
    def get(pc, r):
        desc = FPFH(pc, r)
        inds = np.arange(np.asarray(pc.points).shape[0])
        return desc.get_features(inds)


class RCSLRF:

    def __init__(self, pcd, pcd_tree, lrf_kernel):

        self.pcd = pcd
        self.pcd_tree = pcd_tree
        self.patch_kernel = lrf_kernel

    def get(self, pt):

        _, patch_idx, _ = self.pcd_tree.search_radius_vector_3d(pt, self.patch_kernel)

        ptnn = np.asarray(self.pcd.points)[patch_idx[1:], :].T
        ptall = np.asarray(self.pcd.points)[patch_idx, :].T

        # eq. 3
        ptnn_cov = 1 / len(ptnn) * np.dot((ptnn - pt[:, np.newaxis]), (ptnn - pt[:, np.newaxis]).T)

        # if len(patch_idx) < self.patch_kernel / 2:
        #     _, patch_idx, _ = self.pcd_tree.search_knn_vector_3d(pt, self.patch_kernel)

        a, v = np.linalg.eig(ptnn_cov)
        smallest_eigevalue_idx = np.argmin(a)
        np_hat = v[:, smallest_eigevalue_idx]

        # eq. 4
        zp = np_hat if np.sum(np.dot(np_hat, pt[:, np.newaxis] - ptnn)) > 0 else - np_hat

        v = (ptnn - pt[:, np.newaxis]) - (np.dot((ptnn - pt[:, np.newaxis]).T, zp[:, np.newaxis]) * zp).T

        # 选球域的轮廓上那些点，也就是论文里的Q
        dis = np.sqrt(np.sum((ptnn - pt[:, np.newaxis])**2, axis=0))
        q_ind = (0.8*self.patch_kernel <= dis) & (dis <= self.patch_kernel)
        v = v[:, q_ind]

        pq = (ptnn - pt[:, np.newaxis])[:, q_ind]

        w = np.sign(pq.T.dot(zp[:, np.newaxis]).squeeze())*(np.power(pq.T.dot(zp[:, np.newaxis]), 2).squeeze())
        if isinstance(w, np.float64):
            w = np.array([w])
        # e.q. 5
        # print(v.shape, w.shape)
        xp = 1 / np.linalg.norm(np.dot(v, w[:, np.newaxis])) * np.dot(v, w[:, np.newaxis])
        xp = xp.squeeze()

        yp = np.cross(xp, zp)

        lRg = np.asarray([xp, yp, zp]).T
        # rotate w.r.t local frame and centre in zero using the chosen point
        ptall = (lRg.T @ (ptall - pt[:, np.newaxis])).T

        T = np.zeros((4, 4))
        T[-1, -1] = 1
        T[:3, :3] = lRg
        T[:3, -1] = pt

        return ptall, pt, T


class RCS:
    def __init__(self, pc, r, n_theta=6, nc=12):
        self.points = np.asarray(pc.points)
        self.lrf = RCSLRF(pc, o3d.geometry.KDTreeFlann(pc), lrf_kernel=r)
        self.n_theta, self.nc, self.r = n_theta, nc, r

    def get(self, pt, patch_visual=False):
        patch, _, T = self.lrf.get(pt)
        f = np.zeros(shape=(self.n_theta*self.nc))
        for i in range(self.n_theta):
            theta = i * math.pi / 6
            R = np.array([
                [cos(theta)*cos(theta), sin(theta)*cos(theta), -sin(theta)],
                [-sin(theta)*cos(theta)+sin(theta)*sin(theta)*cos(theta), cos(theta)*cos(theta)+math.pow(sin(theta), 3), sin(theta)*cos(theta)],
                [sin(theta)*sin(theta)+sin(theta)*cos(theta)*cos(theta), -sin(theta)*cos(theta)+sin(theta)*sin(theta)*cos(theta), cos(theta)*cos(theta)]
            ])
            patch_rotate = R.dot(patch.T).T
            c_pts = self.get_c_from_patch(patch_rotate[:, 0:2], patch_visual)
            f_theta = np.sqrt(np.sum(np.power(c_pts, 2), axis=1))
            f[i*self.nc:i*self.nc+self.nc] = f_theta / self.r
        return f

    def get_features(self, pts_ind, patch_visual=False):
        features = np.zeros(shape=(pts_ind.shape[0], self.n_theta*self.nc))
        center = self.points[pts_ind]
        for i, pt in enumerate(center):
            features[i, :] = self.get(pt, patch_visual)
            print("\r%d / %d" % (i + 1, pts_ind.shape[0]), end="")
        return features

    def get_c_from_patch(self, xy, patch_visual):
        thetas = np.arctan2(xy[:, 1], xy[:, 0])
        thetas[thetas < 0] = 2*math.pi+thetas[thetas < 0]
        gap = 2*math.pi/self.nc
        c_pts = np.zeros(shape=(self.nc, 2))
        # 是否要把patch展示出来，包括射线和轮廓点
        if patch_visual:
            plt.scatter(xy[:, 0], xy[:, 1], s=2)
            for i in range(self.nc):
                theta = i * gap
                x, y = 0.1*cos(theta), 0.1*sin(theta)
                plt.plot([0, x], [0, y], c="gray")

        for i in range(self.nc):
            theta = i * gap
            # print(np.abs(thetas-theta).min())
            candidate_ind = (np.abs(thetas-theta) < 0.1)
            if xy[candidate_ind].shape[0] == 0:
                continue
            c_pts[i, :] = xy[candidate_ind][np.argmax(np.sum(xy[candidate_ind]**2, axis=1))]
        # 展示轮廓点
        if patch_visual:
            plt.scatter(c_pts[:, 0], c_pts[:, 1], c='red', s=16)
            plt.show()
        return c_pts


class BSLRF:
    def __init__(self, pcd, pcd_tree, lrf_kernel):
        self.pcd = pcd
        self.pcd_tree = pcd_tree
        self.patch_kernel = lrf_kernel

    def get(self, pt):
        _, patch_idx, _ = self.pcd_tree.search_radius_vector_3d(pt, self.patch_kernel)
        ptall = np.asarray(self.pcd.points)[patch_idx, :].T

        lRg = self.get_lrf(pt)

        # rotate w.r.t local frame and centre in zero using the chosen point
        ptall = (lRg.T @ (ptall - pt[:, np.newaxis])).T

        # this is our normalisation
        ptall /= self.patch_kernel

        T = np.zeros((4, 4))
        T[-1, -1] = 1
        T[:3, :3] = lRg
        T[:3, -1] = pt

        return ptall

    def get_lrf(self, pt):
        _, patch_idx, _ = self.pcd_tree.search_radius_vector_3d(pt, self.patch_kernel)

        ptnn = np.asarray(self.pcd.points)[patch_idx[1:], :].T

        # eq. 3
        ptnn_cov = 1 / len(ptnn) * np.dot((ptnn - pt[:, np.newaxis]), (ptnn - pt[:, np.newaxis]).T)

        if len(patch_idx) < self.patch_kernel / 2:
            _, patch_idx, _ = self.pcd_tree.search_knn_vector_3d(pt, self.patch_kernel)

        # The normalized (unit “length”) eigenvectors, s.t. the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
        a, v = np.linalg.eig(ptnn_cov)
        smallest_eigevalue_idx = np.argmin(a)
        np_hat = v[:, smallest_eigevalue_idx]

        # eq. 4
        zp = np_hat if np.sum(np.dot(np_hat, pt[:, np.newaxis] - ptnn)) > 0 else - np_hat

        v = (ptnn - pt[:, np.newaxis]) - (np.dot((ptnn - pt[:, np.newaxis]).T, zp[:, np.newaxis]) * zp).T
        alpha = (self.patch_kernel - np.linalg.norm(pt[:, np.newaxis] - ptnn, axis=0)) ** 2
        beta = np.dot((ptnn - pt[:, np.newaxis]).T, zp[:, np.newaxis]).squeeze() ** 2

        # e.q. 5
        xp = 1 / (np.linalg.norm(np.dot(v, (alpha * beta)[:, np.newaxis])) + 1e-32) * np.dot(v, (alpha * beta)[:,
                                                                                                np.newaxis])
        xp = xp.squeeze()

        yp = np.cross(xp, zp)

        lRg = np.asarray([xp, yp, zp]).T
        return lRg


class BallSurface:
    def __init__(self, pc, r, space):
        self.tree = o3d.KDTreeFlann(pc)
        self.lrf = BSLRF(pc, self.tree, r)
        self.pts = np.asarray(pc.points)
        self.space = space

    def get_features(self, inds):
        key_pts = self.pts[inds]
        desc = []
        for i in range(key_pts.shape[0]):
            pt = key_pts[i]
            patch = self.lrf.get(pt)
            desc.append(self.compute(patch, self.space, self.space))
        desc = np.stack(desc, axis=0)
        return desc

    def compute(self, pc, x_theta_space=1 / 6, z_theta_space=1 / 6):
        # theta_space 是多少多少π，1/6就是间隔为π/6的意思
        # x1 y1 z1
        # x2 y2 z2
        # ...
        # xn yn zn
        pc = torch.Tensor(pc)
        x_theta = torch.atan2(pc[:, 1], pc[:, 0])
        # 0 ~ 2π
        x_theta[x_theta < 0] = 2 * math.pi + x_theta[x_theta < 0]
        # 0 ~ π
        z_theta = torch.atan2(torch.sqrt(pc[:, 0] ** 2 + pc[:, 1] ** 2), pc[:, 2])
        # x 的角度总共划分为多少个扇形
        x_theta_kind, z_theta_kind = int(2 / x_theta_space), int(1 / z_theta_space)
        x_index, z_index = torch.floor(x_theta / (x_theta_space * math.pi)), torch.floor(
            z_theta / (z_theta_space * math.pi))
        x_index, z_index = z_index.long(), z_index.long()

        x_index[x_index > x_theta_kind - 1], z_index[
            z_index > z_theta_kind - 1] = x_theta_kind - 1, z_theta_kind - 1
        x_index[x_index < 0], z_index[z_index < 0] = 0, 0

        index = x_index * z_theta_kind + z_index
        value = torch.sqrt(pc[:, 0] ** 2 + pc[:, 1] ** 2 + pc[:, 2] ** 2)

        index, value = index.numpy(), value.numpy()

        max_ind = np.lexsort([value, index])
        desc = np.zeros((x_theta_kind * z_theta_kind,))
        desc[index[max_ind]] = value[max_ind]

        return desc

    @staticmethod
    def get(pc, r, space):
        desc = BallSurface(pc, r, space)
        inds = np.arange(np.asarray(pc.points).shape[0])
        return desc.get_features(inds)
