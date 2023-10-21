import cv2
import numpy as np
from mathutils import Matrix, Quaternion

from geometry import W_PT as landmarks
from geometry import line_idx, vertex
from visualize import draw_landmarks, project_pts_quat_tvec, read_csv


def plot_lines(img, pts, line_idx):
    img = np.ones_like(img) * 255
    img = np.asarray(img, dtype=np.uint8)
    pts = np.asarray(pts / 3, dtype=np.int)
    for idx in line_idx:
        start_pt, end_pt = pts[idx[0]], pts[idx[1]]
        cv2.line(
            img,
            (start_pt[0], start_pt[1]),
            (end_pt[0], end_pt[1]),
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
    return img


rot = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
mat = Matrix.Rotation(30 / 180 * np.pi, 3, "X")
quat = Matrix.to_quaternion(mat)
quat.rotate(Matrix.Rotation(10 / 180 * np.pi, 3, "Y").to_quaternion())

img_dir = "./speedplus_small/sunlamp/images"
pose_file = "./csv_files/lightbox_ex_submission_1953_2022_03_30_21_34_58.csv"

img_name = "img000217.jpg"
pose = read_csv(pose_file, "sunlamp")
pose_000017 = pose[img_name]

vertex = np.asarray(vertex)
# vertex_2d = project_pts_quat_tvec(vertex, *pose_000017)
vertex_2d = project_pts_quat_tvec(vertex, quat, pose_000017[1])
keypts = project_pts_quat_tvec(np.asarray(landmarks), quat, pose_000017[1])

img = cv2.imread(f"{img_dir}/{img_name}")
img = plot_lines(img, vertex_2d, line_idx)
img = draw_landmarks(img, keypts / 3)

cv2.imshow("win", img)
cv2.waitKey(0)
cv2.destroyWindow("win")

cv2.imwrite("model.png", img)
