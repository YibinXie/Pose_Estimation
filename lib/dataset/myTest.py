from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
from collections import OrderedDict
import logging
import os
import copy
import torch
import torchvision
import math

from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json_tricks as json
import numpy as np
import cv2

import torchvision.transforms as transforms
from dataset.JointsDataset import JointsDataset
from nms.nms import oks_nms
from nms.nms import soft_oks_nms
from core.inference import get_max_preds
from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints


logger = logging.getLogger(__name__)


class MyCOCO(Dataset):
    def __init__(self):
        self.bbox = '/home/xyb/mnt/win_d/0_MySpace/date/data/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json'
        self.root = '/home/xyb/mnt/win_d/0_MySpace/date/data/coco/'
        self.image_set = 'train2017'

        self.coco = COCO(self._get_ann_file_keypoint())

        # Deal with class names
        cats = [cat['name']
                for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        print('=> classes: {}'.format(self.classes))
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict(
            [
                (self._class_to_coco_ind[cls], self._class_to_ind[cls])
                for cls in self.classes[1:]
            ]
        )

        # Load image file names
        self.image_set_index = self._load_image_set_index()
        self.num_images = len(self.image_set_index)
        print('=> num_images: {}'.format(self.num_images))

        self.num_joints = 17+1  # 多加一个计算出的neck
        self.joints_from_coco_to_use = [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 4, 1, 3]
        self.data_format = 'jpg'
        self.image_width = 192  # 图像的宽
        self.image_height = 256  # 图像的高
        self.aspect_ratio = self.image_width * 1.0 / self.image_height  # 宽高比
        self.pixel_std = 200  # 像素标准差
        self.db = self._get_db()

        self.target_type = 'gaussian'
        self.sigma = 2
        self.heatmap_size = np.array((96, 128))
        self.paf_map_size = np.array((96, 128))
        self.image_size = np.array((192, 256))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])

        self.joint_pairs = [[1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [1, 2], [2, 3], [3, 4], [2, 16],
                            [1, 5], [5, 6], [6, 7], [5, 17], [1, 0], [0, 14], [0, 15], [14, 16], [15, 17]]
        # self.joint_pairs = [[0, 1], [1, 3], [0, 2], [2, 4], [5, 7], [7, 9], [6, 8], [8, 10], [11, 13], [13, 15],
        #                     [12, 14], [14, 16]]
        # self.joint_pairs = [[1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [1, 2], [2, 3], [3, 4],
        #                     [1, 5], [5, 6], [6, 7], [1, 0], [0, 14], [0, 15], [14, 16], [15, 17]]
        self.num_joint_pairs = len(self.joint_pairs)
        self.stride = (self.image_size // self.paf_map_size)[0]
        self.paf_thickness = 4 / self.stride

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''

        data_numpy = cv2.imread(
            image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        if data_numpy is None:
            print('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        trans = get_affine_transform(c, s, r, self.image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        if self.transform:
            input = self.transform(input)

        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        target, target_weight = self.generate_target(joints, joints_vis)

        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)
        target_map = torch.from_numpy(self.generate_paf(joints, joints_vis))

        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score
        }

        return input, target, target_weight, target_map, meta

    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)  # +0.5就可以四舍五入了
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                # 理论上不需要检查的，因为这个肯定满足的
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # Generate gaussian
                size = 2 * tmp_size + 1  # 直径
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2  # 中心
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]  # 就是g的全部了

        return target, target_weight

    # def generate_paf(self, joints, joints_vis):
    #     target_paf = np.zeros((self.num_joint_pairs * 2,
    #                            self.image_height // self.stride,
    #                            self.image_width // self.stride),
    #                           dtype=np.float32)
    #
    #     for paf_id in range(self.num_joint_pairs):
    #         keypoint_a = joints[paf_id[0]]
    #         keypoint_b = joints[paf_id[1]]
    #         if joints_vis[paf_id[0]] == 1 and joints_vis[paf_id[1]] == 1:
    #             keypoint_a /= self.stride
    #             keypoint_b /= self.stride
    #             x_ba = keypoint_b[0] - keypoint_a[0]
    #             y_ba = keypoint_b[1] - keypoint_b[0]
    #             _, h_map, w_map = target_paf[paf_id * 2:paf_id * 2 + 2]
    #             x_min = int(max(min(keypoint_a[0, keypoint_b[0]]) - self.paf_thickness), 0)
    #             x_max = int(max(min(keypoint_a[0, keypoint_b[0]]) - self.paf_thickness), w_map)
    #             y_min = int(max(min(keypoint_a[1, keypoint_b[1]]) - self.paf_thickness), 0)
    #             y_max = int(max(min(keypoint_a[1, keypoint_b[1]]) - self.paf_thickness), h_map)
    #             norm_ba = (x_ba * x_ba + y_ba * y_ba) ** 0.5
    #             if norm_ba < 1e-7:  # Same points, no paf
    #                 return
    #             x_ba /= norm_ba
    #             y_ba /= norm_ba
    #
    #             for y in range(y_min, y_max):
    #                 for x in range(x_min, x_max):
    #                     x_ca = x - keypoint_a[0]
    #                     y_ca = y - keypoint_a[1]
    #                     d = math.fabs(x_ca * y_ba - y_ca * x_ba)
    #                     if d <= self.paf_thickness:
    #                         target_paf[0, y, x] = x_ba
    #                         target_paf[1, y, x] = y_ba

    def generate_paf(self, joints, joints_vis):
        target_paf = np.zeros((self.num_joint_pairs * 2,
                               self.image_height // self.stride,
                               self.image_width // self.stride),
                              dtype=np.float32)

        for paf_id in range(self.num_joint_pairs):
            keypoint_a = joints[self.joint_pairs[paf_id][0]]  # 得到的是一个3D向量，[x, y, 0]
            keypoint_b = joints[self.joint_pairs[paf_id][1]]
            if joints_vis[self.joint_pairs[paf_id][0]][0] > 0 and joints_vis[self.joint_pairs[paf_id][1]][0] > 0:  # 两个关节都有才能连
                target_paf[paf_id * 2:paf_id * 2 + 2] = \
                    self._get_paf_maps(target_paf[paf_id * 2:paf_id * 2 + 2],
                                       keypoint_a[0], keypoint_a[1], keypoint_b[0], keypoint_b[1],
                                       self.stride, self.paf_thickness)

        return target_paf

    # def _get_paf_maps(self, paf_map, x_a, y_a, x_b, y_b, stride, thickness):
    #     x_a /= stride  # 都缩到map的图的尺寸
    #     y_a /= stride
    #     x_b /= stride
    #     y_b /= stride
    #     x_ba = x_b - x_a  # 计算向量
    #     y_ba = y_b - y_a
    #     _, h_map, w_map = paf_map.shape
    #     x_min = int(max(min(x_a, x_b) - thickness, 0))
    #     x_max = int(min(max(x_a, x_b) + thickness, w_map))
    #     y_min = int(max(min(y_a, y_b) - thickness, 0))
    #     y_max = int(min(max(y_a, y_b) + thickness, h_map))
    #     norm_ba = (x_ba * x_ba + y_ba * y_ba) ** 0.5  # 两关节点之间的肢体长度 lc,k
    #     if norm_ba < 1e-7:  # Same points, no paf
    #         return  # 理论上不会出现这种情况，因为前面进来时限制了不同关键点
    #     x_ba /= norm_ba  # 得到指示两个关节点形成的肢体方向的单位向量 v
    #     y_ba /= norm_ba
    #
    #     for y in range(y_min, y_max):
    #         for x in range(x_min, x_max):
    #             x_ca = x - x_a
    #             y_ca = y - y_a
    #             d = math.fabs(x_ca * y_ba - y_ca * x_ba)  # 返回浮点数绝对值
    #             if d <= thickness:  # 在肢体垂直方向上的长度分量小于人为设定的阈值
    #                 paf_map[0, y, x] = x_ba  # 该点p的L函数的值就是v
    #                 paf_map[1, y, x] = y_ba
    def _get_paf_maps(self, paf_map, x_a, y_a, x_b, y_b, stride, thickness):
        x_a /= stride  # 都缩到map的图的尺寸
        y_a /= stride
        x_b /= stride
        y_b /= stride
        x_ba = x_b - x_a  # 计算向量
        y_ba = y_b - y_a
        norm_ba = (x_ba * x_ba + y_ba * y_ba) ** 0.5  # 两关节点之间的肢体长度 lc,k
        if norm_ba < 1e-7:  # Same points, no paf
            return  # 理论上不会出现这种情况，因为前面进来时限制了不同关键点
        x_ba /= norm_ba  # 得到指示两个关节点形成的肢体方向的单位向量 v
        y_ba /= norm_ba
        norm_v = (x_ba * x_ba + y_ba * y_ba) ** 0.5  # 应该=1
        _, h_map, w_map = paf_map.shape
        # 先确定一个大致的矩形范围，否则就要全图点遍历了
        x_min = int(max(min(x_a, x_b) - thickness, 0))
        x_max = int(min(max(x_a, x_b) + thickness, w_map))
        y_min = int(max(min(y_a, y_b) - thickness, 0))
        y_max = int(min(max(y_a, y_b) + thickness, h_map))

        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                x_ca = x - x_a
                y_ca = y - y_a
                d = math.fabs(x_ca * y_ba - y_ca * x_ba)  # 返回浮点数绝对值，在v垂直向量方向上的投影
                if d <= thickness:  # 在肢体垂直方向上的长度分量小于人为设定的阈值
                    paf_map[0, y, x] = math.fabs(x_ba)  # 该点p的L函数的值就是v
                    paf_map[1, y, x] = math.fabs(y_ba)  # 如果是负数，等价就显示不出来了

        return paf_map

    def _get_ann_file_keypoint(self):
        prefix = 'person_keypoints' \
            if 'test' not in self.image_set else 'image_info'
        return os.path.join(
            self.root,
            'annotations',
            prefix + '_' + self.image_set + '.json'
        )

    def _load_image_set_index(self):
        image_ids = self.coco.getImgIds()
        return image_ids

    def _get_db(self):
        gt_db = self._load_coco_keypoint_annotations()
        return gt_db

    def _load_coco_keypoint_annotations(self):
        gt_db = []
        for index in self.image_set_index:
            gt_db.extend(self._load_coco_keypoint_annotation_kernal(index))
        return gt_db

    def _load_coco_keypoint_annotation_kernal(self, index):
        """
        coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
        iscrowd:
            crowd instances are handled by marking their overlaps with all categories to -1
            and later excluded in training
        bbox:
            [x1, y1, w, h]
        :param index: coco image id
        :return: db entry
        """
        im_ann = self.coco.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.coco.loadAnns(annIds)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2-x1, y2-y1]
                valid_objs.append(obj)
        objs = valid_objs

        rec = []
        for obj in objs:
            cls = self._coco_ind_to_class_ind[obj['category_id']]  # 从coco index转变到class index
            if cls != 1:
                continue

            # ignore objs without keypoints annotation
            if max(obj['keypoints']) == 0:
                continue

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)  # 多计算一个neck
            joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)
            for ipt in range(self.num_joints):
                if ipt == self.num_joints-1:
                    if joints_3d_vis[5, 0] > 0 and joints_3d_vis[6, 0] > 0:
                        joints_3d[ipt, 0] = (joints_3d[5, 0] + joints_3d[6, 0]) // 2  # 四舍五入还是向下取整
                        joints_3d[ipt, 1] = (joints_3d[5, 1] + joints_3d[6, 1]) // 2
                        joints_3d[ipt, 2] = 0
                        joints_3d_vis[ipt, 0] = 1
                        joints_3d_vis[ipt, 1] = 1
                        joints_3d_vis[ipt, 2] = 0
                else:
                    joints_3d[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
                    joints_3d[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
                    joints_3d[ipt, 2] = 0
                    t_vis = obj['keypoints'][ipt * 3 + 2]
                    if t_vis > 1:
                        t_vis = 1
                    joints_3d_vis[ipt, 0] = t_vis
                    joints_3d_vis[ipt, 1] = t_vis
                    joints_3d_vis[ipt, 2] = 0

            joints_3d_tmp = copy.deepcopy(joints_3d)
            joints_3d_vis_tmp = copy.deepcopy(joints_3d_vis)
            for ipt in range(self.num_joints):
                joints_3d[ipt, :] = joints_3d_tmp[self.joints_from_coco_to_use[ipt], :]
                joints_3d_vis[ipt, :] = joints_3d_vis_tmp[self.joints_from_coco_to_use[ipt], :]

            center, scale = self._box2cs(obj['clean_bbox'][:4])  # 获得bbox的中心和范围，也就是x,y,w,h
            rec.append({
                'image': self.image_path_from_index(index),
                'center': center,
                'scale': scale,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'filename': '',
                'imgnum': 0,
            })

        return rec

    # 得到clean bbox
    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    # 坐标生成clean bbox
    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    # 从index获取image path
    def image_path_from_index(self, index):
        """ example: images / train2017 / 000000119993.jpg """
        file_name = '%012d.jpg' % index
        if '2014' in self.image_set:
            file_name = 'COCO_%s_' % self.image_set + file_name

        prefix = 'test2017' if 'test' in self.image_set else self.image_set

        data_name = prefix + '.zip@' if self.data_format == 'zip' else prefix

        image_path = os.path.join(
            self.root, 'images', data_name, file_name)

        return image_path


def save_batch_image_with_joints(batch_image, batch_joints, batch_joints_vis,
                                 file_name='.', nrow=8, padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]

            for joint, joint_vis in zip(joints, joints_vis):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis[0]:
                    cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, [255, 0, 0], 2)
            k = k + 1
    cv2.imwrite(file_name, ndarr)


def save_batch_heatmaps(batch_image, batch_heatmaps, file_name,
                        normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: [batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros((batch_size*heatmap_height,
                           (num_joints+1)*heatmap_width,
                           3),
                          dtype=np.uint8)

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap*0.7 + resized_image*0.3
            # 这里是多余的，因为heatmap本身就有了，相当于重复画了
            cv2.circle(masked_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j+1)
            width_end = heatmap_width * (j+2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)


def save_batch_paf_maps(batch_image, batch_heatmaps, batch_paf_maps, file_name,
                        normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_paf_maps: [batch_size, num_pairs, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_paf_maps.size(0)
    num_pairs = batch_paf_maps.size(1) // 2
    paf_map_height = batch_paf_maps.size(2)
    paf_map_width = batch_paf_maps.size(3)

    grid_image = np.zeros((batch_size*paf_map_height,
                           (num_pairs+1)*paf_map_width,
                           3),
                          dtype=np.uint8)

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    joint_pairs = [[1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [1, 2], [2, 3], [3, 4], [2, 16],
                   [1, 5], [5, 6], [6, 7], [5, 17], [1, 0], [0, 14], [0, 15], [14, 16], [15, 17]]

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()
        paf_maps = batch_paf_maps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,  # 变到map的尺寸
                                   (int(paf_map_width), int(paf_map_height)))

        height_begin = paf_map_height * i
        height_end = paf_map_height * (i + 1)
        for j in range(num_pairs):
            cv2.circle(resized_image,
                       (int(preds[i][joint_pairs[j][0]][0]), int(preds[i][joint_pairs[j][0]][1])),
                       1, [0, 0, 255], 1)
            cv2.circle(resized_image,
                       (int(preds[i][joint_pairs[j][1]][0]), int(preds[i][joint_pairs[j][1]][1])),
                       1, [0, 0, 255], 1)
            for k in range(paf_map_height):
                for l in range(paf_map_width):
                    # keypoint_b = joints[self.joint_pairs[paf_id][1]-1]
                    if batch_paf_maps[i][2*j][k][l] != 0:  # 说明改点=v，以该点为起点画一个v（单位向量）
                    # if heatmaps[2*j][k][l] != 0:  # 这样应该也行
                        # 这里v的两个坐标都是小数，变成整数后和出发点相同了，画不出来
                        # 向上取整可以吗？
                        # cv2.line(resized_image,
                        #          (l, k),
                        #          (l+round(batch_paf_maps[i][2*j][k][l]), k+round(batch_paf_maps[i][2*j+1][k][l])),
                        #          (0, 255, 0), 1)
                        # cv2.circle(resized_image, (k, l), 1, (255, 0, 0), 1)  # 点点都糊成一团了
                        pass
            heatmap = np.zeros_like(heatmaps[0], dtype=np.uint8)
            heatmap2 = np.zeros_like(heatmaps[0], dtype=np.uint8)
            if np.max(paf_maps[2*j, :, :]) > 0:
                heatmap = heatmaps[joint_pairs[j][0], :, :]
                heatmap2 = heatmaps[joint_pairs[j][1], :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            colored_heatmap2 = cv2.applyColorMap(heatmap2, cv2.COLORMAP_JET)
            paf_map = paf_maps[2*j, :, :]  # 这里对吗？
            colored_paf_map = cv2.applyColorMap(paf_map, cv2.COLORMAP_JET)  # 0就是(128, 0, 0)了，就是蓝色的
            masked_image = colored_heatmap*0.7 + colored_heatmap2*0.7 \
                           + colored_paf_map*0.7 + resized_image*0.3
            for k in range(paf_map_height):
                for l in range(paf_map_width):
                    # keypoint_b = joints[self.joint_pairs[paf_id][1]-1]
                    if batch_paf_maps[i][2*j][k][l] != 0:  # 说明改点=v，以该点为起点画一个v（单位向量）
                        # cv2.line(masked_image,
                        #          (l, k),
                        #          (l+round(batch_paf_maps[i][2*j][k][l]), k+round(batch_paf_maps[i][2*j+1][k][l])),
                        #          (0, 255, 0), 1)
                        # cv2.circle(masked_image, (k, l), 1, (255, 0, 0), 1)
                        pass

            width_begin = paf_map_width * (j+1)
            width_end = paf_map_width * (j+2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image

        grid_image[height_begin:height_end, 0:paf_map_width, :] = resized_image  # 第一列是放原图的

    cv2.imwrite(file_name, grid_image)

if __name__ == '__main__':
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_dataset = MyCOCO()
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0
    )

    for i, (input, target, target_weight, target_map, meta) in enumerate(train_loader):
        if i < 10:
            save_batch_image_with_joints(
                input, meta['joints'], meta['joints_vis'],
                'myTest/train_{}_gt.jpg'.format(i)
            )
            save_batch_heatmaps(
                input, target, 'myTest/train_{}_hm_gt.jpg'.format(i)
            )
            save_batch_paf_maps(
                input, target, target_map, 'myTest/train_{}_paf_gt.jpg'.format(i)
            )
        else:
            break
