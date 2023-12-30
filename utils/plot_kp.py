import os

import numpy as np
import cv2
from scipy.io import loadmat
from skimage.transform import estimate_transform, warp

import matplotlib.pyplot as plt

end_list = np.array([17, 22, 27, 42, 48, 31, 36, 68], dtype=np.int32) - 1


def get_landmarks(pos, uv_kpt_ind):
    '''
    Args:
        pos: the 3D position map. shape = (256, 256, 3).
    Returns:
        kpt: 68 3D landmarks. shape = (68, 3).
    '''
    kpt = pos[uv_kpt_ind[1, :], uv_kpt_ind[0, :], :]
    return kpt


def get_vertices(pos, resolution_op, face_ind):
    '''
    Args:
        pos: the 3D position map. shape = (256, 256, 3).
    Returns:
        vertices: the vertices(point cloud). shape = (num of points, 3). n is about 40K here.
    '''
    all_vertices = np.reshape(pos, [resolution_op ** 2, -1])
    vertices = all_vertices[face_ind, :]

    return vertices


def plot_vertices(image, vertices, filter=1):
    image = image.copy()
    vertices = np.round(vertices).astype(np.int32)
    for i in range(0, vertices.shape[0], filter):  # sbasak01  2
        st = vertices[i, :2]
        image = cv2.circle(image, (st[0], st[1]), 1, (255, 0, 0), -1)
    return image


def plot_kpt_3d(image, kpt):
    ''' Draw 68 key points
    Args:
        image: the input image
        kpt: (68, 3).
    '''
    image = image.copy()
    kpt = np.round(kpt).astype(np.int32)
    for i in range(kpt.shape[0]):
        st = kpt[i, :2]
        image = cv2.circle(image, (st[0], st[1]), 1, (0, 0, 255), 3)
        if i in end_list:
            continue
        ed = kpt[i + 1, :2]
        image = cv2.line(image, (st[0], st[1]), (ed[0], ed[1]), (255, 255, 255), 1)
    return image


def plot_kpt_2d(image, kpt):
    ''' Draw 68 key points
    Args:
        image: the input image
        kpt: (68, 2).
    '''
    image = image.copy()
    kpt = np.round(kpt).astype(np.int32)
    for i in range(kpt.shape[0]):
        st = kpt[i]
        image = cv2.circle(image, (st[0], st[1]), 1, (0, 0, 255), 3)
        if i in end_list:
            continue
        ed = kpt[i + 1]
        image = cv2.line(image, (st[0], st[1]), (ed[0], ed[1]), (255, 255, 255), 1)
    return image


def img_resize(image):
    resolution_inp = 256
    resolution_op = 256
    print(os.getcwd())
    import dlib
    detector_path = '../data/net-data/mmod_human_face_detector.dat'
    face_detector = dlib.cnn_face_detection_model_v1(
        detector_path)

    def dlib_detect(image):
        return face_detector(image, 1)

    detected_faces = dlib_detect(image)
    if len(detected_faces) == 0:
        # print('warning: no detected face')
        return None

    d = detected_faces[
        0].rect  # # only use the first detected face (assume that each
    # input image only contains one face)
    left = d.left();
    right = d.right();
    top = d.top();
    bottom = d.bottom()
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size * 0.14])
    size = int(old_size * 1.58)

    # crop image


    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2],
                        [center[0] - size / 2, center[1] + size / 2],
                        [center[0] + size / 2, center[1] - size / 2]])
    DST_PTS = np.array([[0, 0], [0, resolution_inp - 1], [resolution_inp - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)

    image = image[:, :, ::-1]  # / 255.
    cropped_image = warp(image, tform.inverse, output_shape=(resolution_inp, resolution_inp))

    return cropped_image


if __name__ == '__main__':
    mat2 = loadmat('/home/shubhajit/Downloads/300W-3D/AFW/134212_1.mat')
    gt_kpt = mat2['pt2d'].transpose()
    gt_img = cv2.imread('/home/shubhajit/Downloads/300W-3D/AFW/134212_1.jpg')
    x1 = plot_kpt_2d(gt_img, gt_kpt)

    cropped_img = img_resize(gt_img)

    cv2.imshow('Input', gt_img)
    cv2.imshow('Sparse alignment', x1)
    cv2.imshow('Sparse alignment GT', cropped_img)
    # cv2.imshow('Dense alignment', result_list[3])
    cv2.moveWindow('Input', 0, 0)
    cv2.moveWindow('Sparse alignment', 500, 0)
    cv2.moveWindow('Sparse alignment GT', 1000, 0)
    # cv2.moveWindow('Dense alignment', 1500, 0)
    key = cv2.waitKey(0)
    if key == ord('q'):
        exit()