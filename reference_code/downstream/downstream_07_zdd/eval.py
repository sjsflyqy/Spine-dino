import torch
import numpy as np
from models import spinal_net
import decoder
import os
from dataset import BaseDataset
import time
import cobb_evaluate


def apply_mask(image, mask, alpha=0.5):
    """Apply the given mask to the image.
    """
    color = np.random.rand(3)
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


class Network(object):
    def __init__(self, args):
        torch.manual_seed(317)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        heads = {'hm': args.num_classes,  # cen, tl, tr, bl, br
                 'reg': 2 * args.num_classes,
                 'wh': 2 * 4, }

        self.model = spinal_net.SpineNet(heads=heads,
                                         pretrained=True,
                                         down_ratio=args.down_ratio,
                                         final_kernel=1,
                                         head_conv=256)
        self.num_classes = args.num_classes
        self.decoder = decoder.DecDecoder(K=args.K, conf_thresh=args.conf_thresh)
        self.dataset = {'spinal': BaseDataset}

    def load_model(self, model, resume):
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        print('loaded weights from {}, epoch {}'.format(resume, checkpoint['epoch']))
        state_dict_ = checkpoint['state_dict']
        model.load_state_dict(state_dict_, strict=False)
        return model

    def eval(self, args, save):
        self.model = self.load_model(self.model, "./weights_base594_spinal/model_100.pth")
        self.model = self.model.to(self.device)
        self.model.eval()

        dataset_module = self.dataset[args.dataset]
        dsets = dataset_module(data_dir=args.data_dir,
                               phase='val',
                               input_h=args.input_h,
                               input_w=args.input_w,
                               down_ratio=args.down_ratio)

        data_loader = torch.utils.data.DataLoader(dsets,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=0,
                                                  pin_memory=True)

        total_time = []
        AP_landmark_dist = []
        AP_mae_dist = []
        LAT_landmark_dist = []
        LAT_mae_dist = []

        for cnt, data_dict in enumerate(data_loader):
            begin_time = time.time()
            image1 = data_dict['images'][0].to('cuda')
            image2 = data_dict['images'][1].to('cuda')

            img_id1 = data_dict['img_id'][0][0]
            img_id2 = data_dict['img_id'][1][0]
            print('processing {}/{} image ...'.format(cnt, len(data_loader)))

            with torch.no_grad():
                output = self.model([image1, image2])
                hm0 = output[0]['hm']
                wh0 = output[0]['wh']
                reg0 = output[0]['reg']

                hm1 = output[1]['hm']
                wh1 = output[1]['wh']
                reg1 = output[1]['reg']
            torch.cuda.synchronize(self.device)
            for index, (hm, wh, reg) in enumerate(
                    zip((hm0, hm1), (wh0, wh1), (reg0, reg1))):
                pts2 = self.decoder.ctdet_decode(hm, wh, reg)  # 17, 11
                pts0 = pts2.copy()
                pts0[:, :10] *= args.down_ratio
                x_index = range(0, 10, 2)
                y_index = range(1, 10, 2)
                if index == 0:
                    ori_image = dsets.load_image(dsets.img_ids.index(img_id1), 0).copy()
                else:
                    ori_image = dsets.load_image(dsets.img_ids_1.index(img_id2), 1).copy()
                h, w, c = ori_image.shape
                pts0[:, x_index] = pts0[:, x_index] / args.input_w * w
                pts0[:, y_index] = pts0[:, y_index] / args.input_h * h
                # sort the y axis
                sort_ind = np.argsort(pts0[:, 1])
                pts0 = pts0[sort_ind]  # ndarray(17,11)
                pr_landmarks = []
                for i, pt in enumerate(pts0):  # 依次添加17节椎骨的4个点的坐标
                    pr_landmarks.append(pt[2:4])
                    pr_landmarks.append(pt[4:6])
                    pr_landmarks.append(pt[6:8])
                    pr_landmarks.append(pt[8:10])
                pr_landmarks = np.asarray(pr_landmarks, np.float32)  # [68, 2]

                end_time = time.time()
                total_time.append(end_time - begin_time)
                if index == 0:
                    gt_landmarks = dsets.load_gt_pts(dsets.load_annoFolder(img_id1, 0))
                else:
                    gt_landmarks = dsets.load_gt_pts(dsets.load_annoFolder(img_id2, 1))
                for pr_pt, gt_pt in zip(pr_landmarks, gt_landmarks):
                    if index==0:
                        AP_landmark_dist.append(np.sqrt((pr_pt[0] - gt_pt[0]) ** 2 + (pr_pt[1] - gt_pt[1]) ** 2))
                        AP_mae_dist.append(abs(pr_pt[0] - gt_pt[0]) / w + abs(pr_pt[1] - gt_pt[1]) / h)
                    else:
                        LAT_landmark_dist.append(np.sqrt((pr_pt[0] - gt_pt[0]) ** 2 + (pr_pt[1] - gt_pt[1]) ** 2))
                        LAT_mae_dist.append(abs(pr_pt[0] - gt_pt[0]) / w + abs(pr_pt[1] - gt_pt[1]) / h)

        print('AP_errordec {}'.format(np.mean(AP_landmark_dist)))
        print('AP_mae {}'.format(np.mean(AP_mae_dist)))
        print('LAT_errordec {}'.format(np.mean(LAT_landmark_dist)))
        print('LAT_mae {}'.format(np.mean(LAT_mae_dist)))
        total_time = total_time[1:]
        print('avg time is {}'.format(np.mean(total_time)))
        print('FPS is {}'.format(1. / np.mean(total_time)))
