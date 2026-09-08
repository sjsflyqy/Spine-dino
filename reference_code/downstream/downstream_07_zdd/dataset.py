import glob
import os
import torch.utils.data as data
import pre_proc
import cv2
from scipy.io import loadmat
import numpy as np
import json


def rearrange_pts(pts):
    boxes = []
    for k in range(0, len(pts), 4):
        pts_4 = pts[k:k + 4, :]
        x_inds = np.argsort(pts_4[:, 0])
        pt_l = np.asarray(pts_4[x_inds[:2], :])
        pt_r = np.asarray(pts_4[x_inds[2:], :])
        y_inds_l = np.argsort(pt_l[:, 1])
        y_inds_r = np.argsort(pt_r[:, 1])
        tl = pt_l[y_inds_l[0], :]
        bl = pt_l[y_inds_l[1], :]
        tr = pt_r[y_inds_r[0], :]
        br = pt_r[y_inds_r[1], :]
        # boxes.append([tl, tr, bl, br])
        boxes.append(tl)
        boxes.append(tr)
        boxes.append(bl)
        boxes.append(br)
    return np.asarray(boxes, np.float32)


class BaseDataset(data.Dataset):
    def __init__(self, data_dir, phase, input_h=None, input_w=None, down_ratio=4):
        super(BaseDataset, self).__init__()
        self.data_dir = data_dir
        self.phase = phase
        self.input_h = input_h
        self.input_w = input_w
        self.down_ratio = down_ratio
        # self.class_name = ['__background__', 'p']
        self.num_classes = 68  # 17*4
        self.input_type = ['ap', 'lat']
        # 第一路输入
        self.img_dir = os.path.join(data_dir, self.input_type[0], 'images', self.phase)
        self.img_ids = sorted(glob.glob(self.img_dir + "/*.jpg"))
        # 第二路输入
        self.img_dir_1 = os.path.join(data_dir, self.input_type[1], 'images', self.phase)
        self.img_ids_1 = sorted(glob.glob(self.img_dir_1 + "/*.jpg"))

    def load_image(self, index, channel):
        if channel == 0:
            # image = cv2.imread(os.path.join(self.img_dir, self.img_ids[index]))
            image = cv2.imread(self.img_ids[index])
        if channel == 1:
            # image = cv2.imread(os.path.join(self.img_dir_1, self.img_ids_1[index]))
            image = cv2.imread(self.img_ids_1[index])
        return image

    def load_gt_pts(self, annopath):
        # pts = loadmat(annopath)['p2']   # num x 2 (x,y)
        with open(annopath, 'r') as load_f:
            load_dict = json.load(load_f)
        pts = []
        # dataset
        # for item in load_dict["shapes"]:
        #     label = item['label']
        #     for x in range(1,18): #17个位置   17*4
        #         x= str(x)
        #         if label in[f"{x}",f"{x}-1",f"{x}-2",f"{x}-3",f"{x}-4"]:
        #             points = item['points'][0]
        #             pts.append(points)
        # pts = rearrange_pts(np.array(pts))
        # return pts
        # dataset594
        for i in range(1, 18):
            i = str(i)
            for item in load_dict["shapes"]:
                label = item['label']
                if label == i:
                    points = item['points'][0]
                    pts.append(points)
        pts = rearrange_pts(np.array(pts))
        return pts

    '''
    channel :0 --->ap
    channel : 1 -->lat
    '''

    def load_annoFolder(self, img_id, channel):
        return os.path.join(self.data_dir, self.input_type[channel], 'labels', self.phase,
                            img_id.split("/")[-1].split(".")[0] + '.json')

    def load_annotation(self, index):
        img_id0 = self.img_ids[index]
        img_id1 = self.img_ids_1[index]

        annoFolder0 = self.load_annoFolder(img_id0, 0)
        annoFolder1 = self.load_annoFolder(img_id1, 1)
        pts0 = self.load_gt_pts(annoFolder0)
        pts1 = self.load_gt_pts(annoFolder1)
        return [pts0, pts1]

    def __getitem__(self, index):
        img_id0 = self.img_ids[index]
        img_id1 = self.img_ids_1[index]
        image0 = self.load_image(index, 0)
        image1 = self.load_image(index, 1)
        # if True:
        if self.phase == 'test':
            images0 = pre_proc.processing_test(image=image0, input_h=self.input_h, input_w=self.input_w)
            images1 = pre_proc.processing_test(image=image1, input_h=self.input_h, input_w=self.input_w)
            return {'images': [images0, images1], 'img_id': [img_id0, img_id1]}
        else:
            aug_label = False
            if self.phase == 'train':
                aug_label = True
            pts = self.load_annotation(index)  # num_obj x h x w
            out_image0, pts_2_0 = pre_proc.processing_train(image=image0,
                                                            pts=pts[0],
                                                            image_h=self.input_h,
                                                            image_w=self.input_w,
                                                            down_ratio=self.down_ratio,
                                                            aug_label=aug_label,
                                                            img_id=img_id0)

            out_image1, pts_2_1 = pre_proc.processing_train(image=image1,
                                                            pts=pts[1],
                                                            image_h=self.input_h,
                                                            image_w=self.input_w,
                                                            down_ratio=self.down_ratio,
                                                            aug_label=aug_label,
                                                            img_id=img_id1)

            data_dict0 = pre_proc.generate_ground_truth(image=out_image0,
                                                        pts_2=pts_2_0,
                                                        image_h=self.input_h // self.down_ratio,
                                                        image_w=self.input_w // self.down_ratio,
                                                        img_id=img_id0)

            data_dict1 = pre_proc.generate_ground_truth(image=out_image1,
                                                        pts_2=pts_2_1,
                                                        image_h=self.input_h // self.down_ratio,
                                                        image_w=self.input_w // self.down_ratio,
                                                        img_id=img_id1)

            return [data_dict0, data_dict1]

    def __len__(self):
        return len(self.img_ids)
