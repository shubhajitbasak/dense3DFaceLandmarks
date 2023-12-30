import numpy as np
import math
from math import cos, sin, atan2, asin
import cv2
from skimage.io import imread
from skimage.transform import estimate_transform, warp
from torchvision import transforms

from typing import Tuple, Union

# import mediapipe as mp

def isRotationMatrix(R):
    ''' checks if a matrix is a valid rotation matrix(whether orthogonal or not)
    '''
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def matrix2angle(R):
    ''' compute three Euler angles from a Rotation Matrix. Ref: http://www.gregslabaugh.net/publications/euler.pdf
    Args:
        R: (3,3). rotation matrix
    Returns:
        x: yaw
        y: pitch
        z: roll
    '''
    # assert(isRotationMatrix(R))

    if R[2, 0] != 1 or R[2, 0] != -1:
        x = asin(R[2, 0])
        y = atan2(R[2, 1] / cos(x), R[2, 2] / cos(x))
        z = atan2(R[1, 0] / cos(x), R[0, 0] / cos(x))

    else:  # Gimbal lock
        z = 0  # can be anything
        if R[2, 0] == -1:
            x = np.pi / 2
            y = z + atan2(R[0, 1], R[0, 2])
        else:
            x = -np.pi / 2
            y = -z + atan2(-R[0, 1], -R[0, 2])

    return x, y, z


def P2sRt(P):
    ''' decompositing camera matrix P.
    Args:
        P: (3, 4). Affine Camera Matrix.
    Returns:
        s: scale factor.
        R: (3, 3). rotation matrix.
        t2d: (2,). 2d translation.
    '''
    t2d = P[:2, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t2d


def compute_similarity_transform(points_static, points_to_transform):
    # http://nghiaho.com/?page_id=671
    p0 = np.copy(points_static).T
    p1 = np.copy(points_to_transform).T

    t0 = -np.mean(p0, axis=1).reshape(3, 1)
    t1 = -np.mean(p1, axis=1).reshape(3, 1)
    t_final = t1 - t0

    p0c = p0 + t0
    p1c = p1 + t1

    covariance_matrix = p0c.dot(p1c.T)
    U, S, V = np.linalg.svd(covariance_matrix)
    R = U.dot(V)
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1

    rms_d0 = np.sqrt(np.mean(np.linalg.norm(p0c, axis=0) ** 2))
    rms_d1 = np.sqrt(np.mean(np.linalg.norm(p1c, axis=0) ** 2))

    s = (rms_d0 / rms_d1)
    P = np.c_[s * np.eye(3).dot(R), t_final]
    return P


def estimate_pose(vertices):
    canonical_vertices = np.load('data/uv-data/canonical_vertices.npy')
    P = compute_similarity_transform(vertices, canonical_vertices)
    _, R, _ = P2sRt(P)  # decompose affine matrix to s, R, t
    pose = matrix2angle(R)
    return P, R


def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


end_list = np.array([17, 22, 27, 42, 48, 31, 36, 68], dtype=np.int32) - 1


def plot_kpt(image, kpt):
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


def plot_vertices(image, vertices, filter=1):
    image = image.copy()
    vertices = np.round(vertices).astype(np.int32)
    for i in range(0, vertices.shape[0], filter):  # sbasak01  2
        st = vertices[i, :2]
        image = cv2.circle(image, (st[0], st[1]), 1, (255, 0, 0), -1)
    return image


def plot_pose_box(image, P, kpt, color=(0, 255, 0), line_width=2):
    ''' Draw a 3D box as annotation of pose. Ref:https://github.com/yinguobing/head-pose-estimation/blob/master/pose_estimator.py
    Args:
        image: the input image
        P: (3, 4). Affine Camera Matrix.
        kpt: (68, 3).
    '''
    image = image.copy()

    point_3d = []
    rear_size = 90
    rear_depth = 0
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = 105
    front_depth = 110
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

    # Map to 2d image points
    point_3d_homo = np.hstack((point_3d, np.ones([point_3d.shape[0], 1])))  # n x 4
    point_2d = point_3d_homo.dot(P.T)[:, :2]
    point_2d[:, :2] = point_2d[:, :2] - np.mean(point_2d[:4, :2], 0) + np.mean(kpt[:27, :2], 0)
    point_2d = np.int32(point_2d.reshape(-1, 2))

    # Draw all the lines
    cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[1]), tuple(
        point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[2]), tuple(
        point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[3]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)

    return image


def get_rect_mediapipe(image):
    pass


def process_input(input, model, face_detector, cfg, cuda=True, image_info=None):
    ''' process image with crop operation.
    Args:
        input: (h,w,3) array or str(image path). image value range:1~255.
        image_info(optional): the bounding box information of faces. if None, will use dlib to detect face.
    Returns:
        pos: the 3D position map. (256, 256, 3).
    '''
    resolution_inp = 256
    resolution_op = 256

    # mp_face_detection = mp.solutions.face_detection

    if isinstance(input, str):
        try:
            image = imread(input)
        except IOError:
            print("error opening file: ", input)
            return None
    else:
        image = input

    if image.ndim < 3:
        image = np.tile(image[:, :, np.newaxis], [1, 1, 3])

    if image_info is not None:
        if np.max(image_info.shape) > 4:  # key points to get bounding box
            kpt = image_info
            if kpt.shape[0] > 3:
                kpt = kpt.T
            left = np.min(kpt[0, :]);
            right = np.max(kpt[0, :]);
            top = np.min(kpt[1, :]);
            bottom = np.max(kpt[1, :])
        else:  # bounding box
            bbox = image_info
            left = bbox[0];
            right = bbox[1];
            top = bbox[2];
            bottom = bbox[3]
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
        size = int(old_size * 1.6)
    else:
        detected_faces = face_detector(image, 1)
        if len(detected_faces) == 0:
            # print('warning: no detected face')
            return None

        d = detected_faces[
            0].rect  # # only use the first detected face (assume that each
        # input image only contains one face)

        # with mp_face_detection.FaceDetection(
        #         model_selection=1, min_detection_confidence=0.5) as face_detection:
        #     results = face_detection.process(image)

        left = d.left()
        right = d.right()
        top = d.top()
        bottom = d.bottom()
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size * 0.14])
        size = int(old_size * 1.58)

    # crop image
    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                        [center[0] + size / 2, center[1] - size / 2]])
    DST_PTS = np.array([[0, 0], [0, resolution_inp - 1], [resolution_inp - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)

    image = image[:, :, ::-1] / 255.
    cropped_image = warp(image, tform.inverse, output_shape=(resolution_inp, resolution_inp))

    transform = transforms.ToTensor()
    cropped_image = transform(cropped_image.astype(np.float32)).unsqueeze(0)

    # run our net
    if cuda == True:
        cropped_pos = model(cropped_image.cuda())
        cropped_pos *= 255.  # (1,1500)
        if cfg.heatmap3d:
            cropped_pos = cropped_pos[0]
        else:
            cropped_pos = cropped_pos.view(1, cfg.num_verts, 3)[0]  # (500,3)
        cropped_pos = cropped_pos.detach().cpu().numpy()

    else:
        cropped_pos = model(cropped_image)
        cropped_pos *= 255.  # (1,1500)
        if cfg.heatmap3d:
            cropped_pos = cropped_pos[0]
        else:
            cropped_pos = cropped_pos.view(1, cfg.num_verts, 3)[0]  # (500,3)
        cropped_pos = cropped_pos.detach().numpy()

    # restore
    cropped_vertices = cropped_pos.T  # (3,500)
    z = cropped_vertices[2, :].copy() / tform.params[0, 0]
    cropped_vertices[2, :] = 1
    vertices = np.dot(np.linalg.inv(tform.params), cropped_vertices)
    vertices = np.vstack((vertices[:2, :], z))
    # pos = np.reshape(vertices.T, [resolution_op, resolution_op, 3])

    return vertices.T


def get_landmarks(self, pos):
    '''
    Args:
        pos: the 3D position map. shape = (256, 256, 3).
    Returns:
        kpt: 68 3D landmarks. shape = (68, 3).
    '''
    kpt = pos[self.uv_kpt_ind[1, :], self.uv_kpt_ind[0, :], :]
    return kpt


def get_vertices(self, pos):
    '''
    Args:
        pos: the 3D position map. shape = (256, 256, 3).
    Returns:
        vertices: the vertices(point cloud). shape = (num of points, 3). n is about 40K here.
    '''
    all_vertices = np.reshape(pos, [self.resolution_op ** 2, -1])
    vertices = all_vertices[self.face_ind, :]

    return vertices


def get_colors_from_texture(self, texture):
    '''
    Args:
        texture: the texture map. shape = (256, 256, 3).
    Returns:
        colors: the corresponding colors of vertices. shape = (num of points, 3). n is 45128 here.
    '''
    all_colors = np.reshape(texture, [self.resolution_op ** 2, -1])
    colors = all_colors[self.face_ind, :]

    return colors


def get_colors(self, image, vertices):
    '''
    Args:
        pos: the 3D position map. shape = (256, 256, 3).
    Returns:
        colors: the corresponding colors of vertices. shape = (num of points, 3). n is 45128 here.
    '''
    [h, w, _] = image.shape
    vertices[:, 0] = np.minimum(np.maximum(vertices[:, 0], 0), w - 1)  # x
    vertices[:, 1] = np.minimum(np.maximum(vertices[:, 1], 0), h - 1)  # y
    ind = np.round(vertices).astype(np.int32)
    colors = image[ind[:, 1], ind[:, 0], :]  # n x 3

    return colors

def get_rect_mp(image, mp_face_detection):
    def _normalized_to_pixel_coordinates(
            normalized_x: float, normalized_y: float, image_width: int,
            image_height: int) -> Union[None, Tuple[int, int]]:
        """Converts normalized value pair to pixel coordinates."""

        # Checks if the float value is between 0 and 1.
        def is_valid_normalized_value(value: float) -> bool:
            return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                              math.isclose(1, value))

        if not (is_valid_normalized_value(normalized_x) and
                is_valid_normalized_value(normalized_y)):
            # to do: Draw coordinates even if it's outside of the image bounds.
            return None
        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)
        return x_px, y_px

    with mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(image)  # cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        annotated_image = image.copy()
        if results.detections:
            for detection in results.detections:
                # mp_drawing.draw_detection(annotated_image, detection)

                image_rows, image_cols, _ = annotated_image.shape

                location = detection.location_data
                relative_bounding_box = location.relative_bounding_box

                rect_start_point = _normalized_to_pixel_coordinates(
                    relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,
                    image_rows)
                rect_end_point = _normalized_to_pixel_coordinates(
                    relative_bounding_box.xmin + relative_bounding_box.width,
                    relative_bounding_box.ymin + relative_bounding_box.height, image_cols,
                    image_rows)

                left = rect_start_point[0]
                right = rect_end_point[0]
                top = rect_start_point[1]
                bottom = rect_end_point[1]

                return left, right, top, bottom
        else:
            return None


def process_input_mp(input, model, mp_face_detection, cuda=True, image_info=None):
    ''' process image with crop operation.
    Args:
        input: (h,w,3) array or str(image path). image value range:1~255.
        image_info(optional): the bounding box information of faces. if None, will use dlib to detect face.
    Returns:
        pos: the 3D position map. (256, 256, 3).
    '''
    resolution_inp = 256
    resolution_op = 256

    # mp_face_detection = mp.solutions.face_detection

    if isinstance(input, str):
        try:
            image = imread(input)
        except IOError:
            print("error opening file: ", input)
            return None
    else:
        image = input

    if image.ndim < 3:
        image = np.tile(image[:, :, np.newaxis], [1, 1, 3])

    if image_info is not None:
        if np.max(image_info.shape) > 4:  # key points to get bounding box
            kpt = image_info
            if kpt.shape[0] > 3:
                kpt = kpt.T
            left = np.min(kpt[0, :]);
            right = np.max(kpt[0, :]);
            top = np.min(kpt[1, :]);
            bottom = np.max(kpt[1, :])
        else:  # bounding box
            bbox = image_info
            left = bbox[0];
            right = bbox[1];
            top = bbox[2];
            bottom = bbox[3]
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
        size = int(old_size * 1.6)
    else:
        left, right, top, bottom = get_rect_mp(image, mp_face_detection)

        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
        size = int(old_size * 1.2)

    # crop image
    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                        [center[0] + size / 2, center[1] - size / 2]])
    DST_PTS = np.array([[0, 0], [0, resolution_inp - 1], [resolution_inp - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)

    image = image[:, :, ::-1] / 255.
    cropped_image = warp(image, tform.inverse, output_shape=(resolution_inp, resolution_inp))

    transform = transforms.ToTensor()
    cropped_image = transform(cropped_image.astype(np.float32)).unsqueeze(0)

    # run our net
    if cuda == True:
        cropped_pos = model(cropped_image.cuda())
        cropped_pos *= 255.  # (1,1500)
        cropped_pos = cropped_pos.view(1, 520, 3)[0]  # (500,3)
        cropped_pos = cropped_pos.detach().cpu().numpy()

    else:
        cropped_pos = model(cropped_image)
        cropped_pos *= 255.  # (1,1500)
        cropped_pos = cropped_pos.view(1, 520, 3)[0]  # (500,3)
        cropped_pos = cropped_pos.detach().numpy()

    # restore
    cropped_vertices = cropped_pos.T  # (3,500)
    z = cropped_vertices[2, :].copy() / tform.params[0, 0]
    cropped_vertices[2, :] = 1
    vertices = np.dot(np.linalg.inv(tform.params), cropped_vertices)
    vertices = np.vstack((vertices[:2, :], z))
    # pos = np.reshape(vertices.T, [resolution_op, resolution_op, 3])

    return vertices.T