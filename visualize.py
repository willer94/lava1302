#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : draw.py
# Author            : WangZi <wangzitju@163.com>
# Date              : 18.12.2021
# Last Modified Date: 18.12.2021
# Last Modified By  : WangZi <wangzitju@163.com>
import cv2
import numpy as np
import os
from mathutils import Quaternion
import pandas as pd
# Visualization


def read_csv(pth, style):
    data = pd.read_csv(pth, header=None)
    raw_data = data[data[0] == style]
    data = raw_data.sort_values(1).to_numpy()
    ret = {}
    for d_ in data:
        ret[d_[1]] = [d_[2:6].astype(float), d_[6:9].astype(float)]
    return ret


def _put_text(img, text, point, color, thickness):
    img = cv2.putText(img, text, point, cv2.FONT_HERSHEY_SIMPLEX,
                      0.5, color, thickness, cv2.LINE_AA)
    return img


COLORS = (
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (0, 206, 208),
    (192, 80, 77),
    (155, 187, 89),
    (128, 100, 162),
    (218, 112, 214),
    (255, 0, 255),
    (91, 74, 66),
    (147, 224, 255),
    (92, 167, 186)


)


COLORS_BGR = [item[::-1] for item in COLORS]


class Camera:
    """" Utility class for accessing camera parameters. """

    fx = 0.0176  # focal length[m]
    fy = 0.0176  # focal length[m]
    nu = 1920  # number of horizontal[pixels]
    nv = 1200  # number of vertical[pixels]
    ppx = 5.86e-6  # horizontal pixel pitch[m / pixel]
    ppy = ppx  # vertical pixel pitch[m / pixel]
    fpx = fx / ppx  # horizontal focal length[pixels]
    fpy = fy / ppy  # vertical focal length[pixels]
    k = [[fpx,   0, nu / 2],
         [0,   fpy, nv / 2],
         [0,     0,      1]]
    K = np.array(k)
    dcoef = np.zeros(5)


def draw_landmarks(img, lmks):
    img_h, img_w = img.shape[:2]
    for idx, a in enumerate(lmks):
        cv2.circle(
            img, (int(round(a[0])), int(round(a[1]))),
            11, COLORS_BGR[idx], -1, lineType=cv2.LINE_AA)
    return img


def vis_prediction_batch(batch, img, lmk, mask, output="./vis"):
    """
    \eye_imgs Bx256x256x3
    \lmks Bx106x2
    """
    if not os.path.isdir(output):
        os.makedirs(output)

    lmk, mask = lmk.cpu().detach().numpy(), mask.cpu().detach().numpy()
    for i in range(len(img)):
        image = draw_landmarks(
            img[i], lmk[i], mask[i][0])
        cv2.imwrite(f'{output}/batch_{batch}_image_{i}.png', image)


def project_pts_quat_tvec(pts, quat, tvec):
    quat = Quaternion(quat)
    Rmat = np.asarray(quat.to_matrix())
    tvec = np.asarray(tvec).reshape(3, 1)
    return project_pts(pts, Camera.K, Rmat, tvec)


def project_pts(pts, K, R, t):
    """Projects 3D points.
    :param pts: nx3 ndarray with the 3D points.
    :param K: 3x3 ndarray with an intrinsic camera matrix.
    :param R: 3x3 ndarray with a rotation matrix.
    :param t: 3x1 ndarray with a translation vector.
    :return: nx2 ndarray with 2D image coordinates of the projections.
    """
    assert (pts.shape[1] == 3)
    P = K.dot(np.hstack((R, t)))
    pts_h = np.hstack((pts, np.ones((pts.shape[0], 1))))
    pts_im = P.dot(pts_h.T)
    pts_im /= pts_im[2, :]
    return pts_im[:2, :].T
