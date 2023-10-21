import numpy as np
from mathutils import Quaternion

from render_depth import load_obj, render_depth
from utils.utils import Camera


class Render(object):
    def __init__(self):

        self.model = load_obj("./mesh20000.obj")
        self.im_size = (640, 400)
        self.K = Camera.K.copy()

        # self.K[:2, 2] = self.K[:2, 2] / 3
        self.K[:, 2] = self.K[:, 2] / 3
        self.nc, self.fc = 0.01, 15

    def __call__(self, quat, tvec):
        cam_k = self.K.copy()
        # cam_k[:2, :] = cam_k[:2, :] / 3

        quat = Quaternion(quat)
        Rmat = np.asarray(quat.to_matrix())
        tvec = np.asarray(tvec).reshape(3, 1)

        depth = render_depth(
            self.model, self.im_size, cam_k, Rmat, tvec * 3, self.nc, self.fc
        )

        return depth
