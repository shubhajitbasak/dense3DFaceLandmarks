# https://wywu.github.io/projects/LAB/WFLW.html
# https://github.com/polarisZhao/PFLD-pytorch
import os
import random
import skimage
import numpy as np
from glob import glob
import cv2
from skimage import transform as tf
from scipy.io import loadmat

import matplotlib.pyplot as plt
from utils.plot_kp import plot_kpt_3d, plot_vertices, get_landmarks, get_vertices, plot_kpt_2d

from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms


class wlpuvDatasets(data.Dataset):
    """Represents a 2D segmentation dataset.

    Input params:
        configuration: Configuration dictionary.
    """

    def __init__(self, config):

        self.data_root = config.dataset_path
        self.datas_list = glob(os.path.join(self.data_root, '*/*.npy'))
        self.img_size = config.img_size  # 256
        self.posmap_size = config.posmap_size  # 256

        self.is_aug = config.is_aug  # True
        self.min_blur_resize = config.min_blur_resize  # 75
        self.max_noise_var = config.max_noise_var  # 0.01
        self.max_rot = config.max_rot  # 45
        self.min_scale = config.min_scale  # 0.95
        self.max_scale = config.max_scale  # 1.05
        self.max_shift = config.max_shift  # 0.0

        self.uv_kpt_ind = np.loadtxt(config.uv_kpt_ind).astype(np.int32)  # 2 x 68 get kpt
        self.face_ind = np.loadtxt(config.face_ind).astype(
            np.int32)  # get valid vertices in the pos map
        self.triangles = np.loadtxt(config.triangles).astype(np.int32)
        self.filtered_indexs = np.loadtxt(config.filtered_indexs).astype(int)
        self.filtered_kpt = np.loadtxt(config.filtered_68_kpt).astype(int)
        self.filtered_kpt_500 = np.loadtxt(config.filtered_kpt_500).astype(int)

        # indices = []
        # for iy in self.filtered_kpt:
        #     if iy in self.filtered_indexs:
        #         indices.append(np.where(self.filtered_indexs == iy)[0][0])
        #     else:
        #         print(iy)
        #
        # print(indices)

        self.resolution_op = config.resolution_op

    def data_aug(self, data_dict):
        ### image data augmentation ###
        new_img = data_dict['Image']

        angle_aug = random.random() * self.max_rot * 2 - self.max_rot
        scale_aug = random.random() * (self.max_scale - self.min_scale) + \
                    self.min_scale

        shift_aug_x = random.random() * \
                      (self.max_shift * self.posmap_size) * 2 \
                      - (self.max_shift * self.posmap_size)
        shift_aug_y = random.random() * \
                      (self.max_shift * self.posmap_size) * 2 \
                      - (self.max_shift * self.posmap_size)

        tform = tf.SimilarityTransform(
            scale=scale_aug,
            rotation=np.deg2rad(angle_aug),
            translation=(shift_aug_x, shift_aug_y))

        shift_y, shift_x = np.array(new_img.shape[:2]) / 2.
        tf_shift = tf.SimilarityTransform(translation=[-shift_x, -shift_y])
        tf_shift_inv = tf.SimilarityTransform(translation=[shift_x, shift_y])

        new_img = tf.warp(new_img, (tf_shift + (tform + tf_shift_inv)).inverse)

        # fill blank
        border_value = np.mean(new_img[:3, :], axis=(0, 1)) * 0.25 + \
                       np.mean(new_img[-3:, :], axis=(0, 1)) * 0.25 + \
                       np.mean(new_img[:, -3:], axis=(0, 1)) * 0.25 + \
                       np.mean(new_img[:, :3], axis=(0, 1)) * 0.25

        mask = np.sum(new_img.reshape(-1, 3), axis=1) == 0
        mask = mask[:, np.newaxis]
        mask = np.concatenate((mask, mask, mask), axis=1)
        border_value = np.repeat(border_value[np.newaxis, :], mask.shape[0], axis=0)
        border_value *= mask
        new_img = (new_img.reshape(-1, 3) + border_value)
        new_img = new_img.reshape(self.img_size, self.img_size, 3)
        new_img = (new_img * 255.).astype('uint8')

        # gamma correlation
        gamma = random.random() * (1.8 - 1.0) + 1.0
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype(np.uint8)
        new_img = cv2.LUT(new_img, table)

        # noise
        noise_aug = random.random() * self.max_noise_var
        new_img = (skimage.util.random_noise(
            new_img, mode="gaussian", var=noise_aug) * 255).astype(np.uint8)

        # blur
        blur_aug = random.randint(self.min_blur_resize, self.img_size)
        new_img = cv2.resize(cv2.resize(new_img, (blur_aug, blur_aug)),
                             (self.img_size, self.img_size))

        # gray
        if random.random() < 0.2:
            new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
            new_img = np.stack((new_img,) * 3, axis=-1)

        data_dict['Image'] = new_img

        # aug posmap
        posmap = data_dict['Posmap']  # 256,256,3
        vertices = np.reshape(posmap, [-1, 3]).T  # 3,65536
        z = vertices[2, :].copy() / tform.params[0, 0]  # 65536,
        vertices[2, :] = 1
        vertices = np.dot((tf_shift + (tform + tf_shift_inv)).params, vertices)  # 3,65536
        vertices = np.vstack((vertices[:2, :], z))  # 3,65536
        posmap = np.reshape(vertices.T, [self.posmap_size, self.posmap_size, 3])  # 256,256,3
        data_dict['Posmap'] = posmap

        # aug kpt_gt
        # kpt_gt = data_dict['kpt_gt']
        # kpt = kpt_gt.T
        # kpt = np.r_[kpt,[np.ones(68)]]
        # kpt = np.dot((tf_shift + (tform + tf_shift_inv)).params, kpt)
        # data_dict['kpt_gt'] = kpt.T[:, 0:2]

        return data_dict

    def __getitem__(self, index):
        # get source image as x
        # get labels as y
        # data = np.load(self.datas_list[index], allow_pickle=True).item()
        # print(self.datas_list[index])
        # mat_path = self.datas_list[index].replace('.npy', '.mat').replace('300W_LP_UV', '300W_LP')
        # mat = loadmat(mat_path)
        # kpt_gt = mat['pt2d'].transpose()

        data = {'Image': cv2.imread(self.datas_list[index].replace('.npy', '.jpg')),
                'Posmap': np.load(self.datas_list[index])}
                # 'kpt_gt': kpt_gt}
        if self.is_aug:
            data = self.data_aug(data)

        transform = transforms.ToTensor()

        data['Image'] = transform((data['Image'] / 255.).astype(np.float32))
        data['Posmap'] = (data['Posmap'] / 255.).astype(np.float32)
        data['kpt'] = get_landmarks(data['Posmap'], self.uv_kpt_ind)
        vert = get_vertices(data['Posmap'], self.resolution_op, self.face_ind)
        data['vertices_filtered'] = vert[self.filtered_indexs]
        # data['kpt_filtered'] = vert[self.filtered_kpt]
        data['kpt_filtered'] = data['vertices_filtered'][self.filtered_kpt_500]

        # data['kpt_gt'] = kpt_gt

        return data

    def __len__(self):
        # return the size of the dataset
        return len(self.datas_list)


if __name__ == '__main__':
    os.chdir('../')
    import importlib
    config = importlib.import_module("MyTest.configs.config")
    importlib.reload(config)
    cfg = config.config
    wlfwdataset = wlpuvDatasets(cfg)
    dataloader = DataLoader(wlfwdataset, batch_size=256, shuffle=True, num_workers=0, drop_last=False)

    for i, data in enumerate(dataloader):
        img = data['Image'][0] * 255
        pos = data['Posmap'][0] * 255
        kpt = data['kpt'][0] * 255
        vertices_filtered = data['vertices_filtered'][0] * 255
        kpt_filtered = data['kpt_filtered'][0] * 255
        # kpt_gt = data['kpt_gt'][0].numpy()

        # x2 = x2.numpy()
        img = img.permute(1, 2, 0).numpy().astype(np.uint8)

        kpt = kpt.numpy()
        vertices_filtered = vertices_filtered.numpy()
        kpt_filtered = kpt_filtered.numpy()
        # vertices = get_vertices(x2)
        # filtered_vertices = vertices[filtered_indexs]

        result_list = [img,
                       plot_kpt_3d(img, kpt),
                       plot_kpt_3d(img, kpt_filtered),
                       plot_vertices(img, vertices_filtered)]
                       # plot_kpt_2d(img, kpt_gt)]  # ,
        # plot_vertices(x1, filtered_vertices)]

        cv2.imshow('Input', result_list[0])
        cv2.imshow('Sparse alignment', result_list[1])
        cv2.imshow('Sparse alignment GT', result_list[2])
        cv2.imshow('Dense alignment', result_list[3])
        # cv2.imshow('Dense alignment1', result_list[4])
        cv2.moveWindow('Input', 0, 0)
        cv2.moveWindow('Sparse alignment', 500, 0)
        cv2.moveWindow('Sparse alignment GT', 1000, 0)
        cv2.moveWindow('Dense alignment', 1500, 0)
        # cv2.moveWindow('Dense alignment1', 2000, 0)

        key = cv2.waitKey(0)
        if key == ord('q'):
            exit()
