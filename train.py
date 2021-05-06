import numpy as np
import open3d as o3d
import torch
from dataset import Modelnet40Pair
from torch.utils.data import DataLoader
from torch import nn
from sklearn.metrics import precision_recall_fscore_support
from model import Skr
from dataset import get_pc
from utils import ransac_pose_estimation, chamfer_distance, pc_normalize
from copy import deepcopy


def processbar(current, totle):
    process_str = ""
    for i in range(int(20*current/totle)):
        process_str += "█"
    while len(process_str) < 20:
        process_str += " "
    return "%s|  %d / %d" % (process_str, current, totle)


def weighted_bce_loss(prediction, gt):
    loss = nn.BCELoss(reduction='none')

    class_loss = loss(prediction, gt)

    weights = torch.ones_like(gt)
    w_negative = gt.sum() / gt.size(0)
    w_positive = 1 - w_negative

    weights[gt >= 0.5] = w_positive
    weights[gt < 0.5] = w_negative
    w_class_loss = torch.mean(weights * class_loss)

    #######################################
    # get classification precision and recall
    predicted_labels = prediction.detach().cpu().round().numpy()
    cls_precision, cls_recall, _, _ = precision_recall_fscore_support(gt.cpu().numpy(), predicted_labels, average='binary')

    return w_class_loss, cls_precision, cls_recall


def loss_fn(pred, gt):
    # batch_size x n
    loss_val, acc = 0, 0
    for i in range(pred.shape[0]):
        one_loss, one_acc, _ = weighted_bce_loss(pred[i], gt[i])
        loss_val += one_loss
        acc += one_acc
    loss_val /= pred.shape[0]
    acc /= pred.shape[0]
    return loss_val, acc


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path = "E:/modelnet40_ply_hdf5_2048"
categories = ["airplane", "person"]
corr_pts_dis_thresh = 0.05
support_r = 0.4
batch_size = 3
epoch = 251
lr_update_step = 20
param_save_path = "./params/skr-bs.pth"
lr = 0.001
min_lr = 0.00001
net = Skr()
net.to(device)
optimizer = torch.optim.Adam(params=net.parameters(), lr=lr, weight_decay=0)
# optimizer = torch.optim.SGD(params=net.parameters(), lr=lr, weight_decay=0)


modelnet_train = Modelnet40Pair(path, mode="test", descriptor="BS", categories=categories, corr_pts_dis_thresh=corr_pts_dis_thresh, support_r=support_r)
modelnet_test = Modelnet40Pair(path, mode="test", descriptor="BS", categories=categories, corr_pts_dis_thresh=corr_pts_dis_thresh, support_r=support_r)
train_loader, test_loader = DataLoader(modelnet_train, shuffle=True, batch_size=batch_size), DataLoader(modelnet_test, shuffle=False, batch_size=batch_size)


def train():
    def update_lr(optimizer, gamma=0.5):
        lr = 0
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        lr = max(lr * gamma, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print("lr update finished  cur lr: %.5f" % lr)
    max_acc = 0
    for epoch_count in range(1, epoch+1):
        loss_val, key_pts_acc, process, key_acc = 0, 0, 0, 0
        for batch_source_pts, batch_source_normal, batch_source_f, batch_source_key_pts_label, \
            batch_target_pts, batch_target_normal, batch_target_f, batch_target_key_pts_label, s2t_idx, t2s_idx in train_loader:

            source_inp = torch.cat([batch_source_pts, batch_source_normal, batch_source_f], dim=2).to(device)
            target_inp = torch.cat([batch_target_pts, batch_target_normal, batch_target_f], dim=2).to(device)

            source_key_sorce, target_key_sorce = net(source_inp, target_inp, s2t_idx, t2s_idx)
            loss, batch_acc = loss_fn(
                torch.cat([source_key_sorce, target_key_sorce], dim=1),
                torch.cat([batch_source_key_pts_label, batch_target_key_pts_label], dim=1).to(device)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val += loss.item()
            key_acc += batch_acc*batch_source_pts.shape[0]
            process += batch_source_pts.shape[0]
            print("\rprocess: %s  loss: %.5f  mean key acc: %.5f  cur key acc: %.5f" % (processbar(process, len(modelnet_train)), loss.item(), key_acc / process, batch_acc), end="")
        print("\nepoch: %d  loss: %.5f  key acc: %.5f" % (epoch_count, loss_val, key_acc / len(modelnet_train)))
        key_acc = key_acc / len(modelnet_train)
        if max_acc < key_acc:
            max_acc = key_acc
            print("save ...")
            torch.save(net.state_dict(), param_save_path)
            print("finish !!!")
        if epoch_count % lr_update_step == 0:
            update_lr(optimizer, 0.5)


def test():
    modelnet_set = Modelnet40Pair(path, mode="test", descriptor="BS", categories=categories,
                                  corr_pts_dis_thresh=corr_pts_dis_thresh, get_np=True, support_r=support_r)
    print(len(modelnet_set))
    net.load_state_dict(torch.load(param_save_path))
    net.eval()
    test_charmfer_dis = 0
    with torch.no_grad():
        for i in range(len(modelnet_set)):
            source_pts, source_normal, source_f, source_key_pts_idx, \
            target_pts, target_normal, target_f, target_key_pts_idx, \
            source_to_target_min_idx, target_to_source_min_idx, raw_pts, T = modelnet_set[i]

            source_pc = get_pc(source_pts, source_normal, [1, 0.706, 0])
            target_pc = get_pc(target_pts, target_normal, [0, 0.651, 0.929])

            source_inp = torch.cat([torch.Tensor(pc_normalize(deepcopy(source_pts))), torch.Tensor(source_normal), torch.Tensor(source_f)], dim=1).to(device)
            target_inp = torch.cat([torch.Tensor(pc_normalize(deepcopy(target_pts))), torch.Tensor(target_normal), torch.Tensor(target_f)], dim=1).to(device)
            source_to_target_min_idx, target_to_source_min_idx = torch.LongTensor(source_to_target_min_idx).unsqueeze(0), torch.LongTensor(target_to_source_min_idx).unsqueeze(0)

            source_key_sorce, target_key_sorce = net(source_inp.unsqueeze(0), target_inp.unsqueeze(0), source_to_target_min_idx, target_to_source_min_idx)
            source_key_sorce, target_key_sorce = source_key_sorce[0].cpu().numpy(), target_key_sorce[0].cpu().numpy()
            sorce_thresh = 0.3
            source_key_pts_idx_pred, target_key_pts_idx_pred = np.nonzero(source_key_sorce >= sorce_thresh)[0], np.nonzero(target_key_sorce >= sorce_thresh)[0]
            # 关键部分配准
            ransac_T = ransac_pose_estimation(
                source_pts[source_key_pts_idx_pred], target_pts[target_key_pts_idx_pred],
                source_f[source_key_pts_idx_pred], target_f[target_key_pts_idx_pred],
                distance_threshold=0.06, max_iter=500000, max_valid=100000
            )
            icp_result = o3d.registration_icp(source_pc, target_pc, 0.06, init=ransac_T)
            icp_T = icp_result.transformation

            # 评估
            chamfer_dist = chamfer_distance(source_pts, target_pts, raw_pts, icp_T, T)
            print(chamfer_dist)
            test_charmfer_dis += chamfer_dist.item()
            source_pc_key = get_pc(source_pts, None, [1, 0.706, 0])
            target_pc_key = get_pc(target_pts, None, [0, 0.651, 0.929])
            source_pc_key_color, target_pc_key_color = np.asarray(source_pc_key.colors), np.asarray(target_pc_key.colors)
            source_pc_key_color[source_key_pts_idx_pred] = np.array([1, 0, 0])
            target_pc_key_color[target_key_pts_idx_pred] = np.array([1, 0, 0])

            o3d.draw_geometries([target_pc_key], window_name="test key pts", width=1000, height=800)
            o3d.draw_geometries([source_pc_key], window_name="test key pts", width=1000, height=800)
            o3d.draw_geometries([source_pc.transform(icp_T), target_pc], window_name="test registration", width=1000, height=800)
    print("test charmfer: %.5f" % (test_charmfer_dis / len(modelnet_set)))


if __name__ == '__main__':
    # train()
    test()