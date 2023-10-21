import logging

import numpy as np
from glumpy import app, gl, gloo
from glumpy.log import log

# Set backend (http://glumpy.readthedocs.io/en/latest/api/app-backends.html)
app.use("glfw")

# Set logging level
# log.setLevel(logging.WARNING) # ERROR, WARNING, DEBUG, INFO
log.setLevel(logging.ERROR)

# Depth vertex shader
# Ref: https://github.com/julienr/vertex_visibility/blob/master/depth.py
# -------------------------------------------------------------------------------
# Getting the depth from the depth buffer in OpenGL is doable, see here:
#   http://web.archive.org/web/20130416194336/http://olivers.posterous.com/linear-depth-in-glsl-for-real
#   http://web.archive.org/web/20130426093607/http://www.songho.ca/opengl/gl_projectionmatrix.html
#   http://stackoverflow.com/a/6657284/116067
# But it is hard to get good precision, as explained in this article:
# http://dev.theomader.com/depth-precision/
#
# Once the vertex is in view space (view * model * v), its depth is simply the
# Z axis. So instead of reading from the depth buffer and undoing the projection
# matrix, we store the Z coord of each vertex in the color buffer. OpenGL allows
# for float32 color buffer components.
_depth_vertex_code = """
uniform mat4 u_mv;
uniform mat4 u_mvp;
attribute vec3 a_position;
attribute vec3 a_color;
varying float v_eye_depth;
void main() {
    gl_Position = u_mvp * vec4(a_position, 1.0);
    vec3 v_eye_pos = (u_mv * vec4(a_position, 1.0)).xyz; // Vertex position in eye coords.
    // OpenGL Z axis goes out of the screen, so depths are negative
    v_eye_depth = -v_eye_pos.z;
}
"""

# Depth fragment shader
# -------------------------------------------------------------------------------
_depth_fragment_code = """
varying float v_eye_depth;
void main() {
    gl_FragColor = vec4(v_eye_depth, 0.0, 0.0, 1.0);
}
"""


def draw_depth(shape, vertex_buffer, index_buffer, mat_model, mat_view, mat_proj):

    program = gloo.Program(_depth_vertex_code, _depth_fragment_code)
    program.bind(vertex_buffer)
    program["u_mv"] = _compute_model_view(mat_model, mat_view)
    program["u_mvp"] = _compute_model_view_proj(mat_model, mat_view, mat_proj)

    # Frame buffer object
    color_buf = np.zeros((shape[0], shape[1], 4), np.float32).view(gloo.TextureFloat2D)
    depth_buf = np.zeros((shape[0], shape[1]), np.float32).view(gloo.DepthTexture)
    fbo = gloo.FrameBuffer(color=color_buf, depth=depth_buf)
    fbo.activate()

    # OpenGL setup
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glClearColor(0.0, 0.0, 0.0, 0.0)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glViewport(0, 0, shape[1], shape[0])

    # Keep the back-face culling disabled because of objects which do not have
    # well-defined surface (e.g. the lamp from the dataset of Hinterstoisser)
    gl.glDisable(gl.GL_CULL_FACE)
    # gl.glEnable(gl.GL_CULL_FACE)
    # gl.glCullFace(gl.GL_BACK) # Back-facing polygons will be culled

    # Rendering
    program.draw(gl.GL_TRIANGLES, index_buffer)
    # program.draw(gl.GL_POINTS, index_buffer)

    # Retrieve the contents of the FBO texture
    depth = np.zeros((shape[0], shape[1], 4), dtype=np.float32)
    gl.glReadPixels(0, 0, shape[1], shape[0], gl.GL_RGBA, gl.GL_FLOAT, depth)
    depth.shape = shape[0], shape[1], 4
    depth = depth[::-1, :]
    depth = depth[:, :, 0]  # Depth is saved in the first channel

    fbo.deactivate()
    del color_buf, depth_buf

    return depth


def load_obj(path):
    """
    only load vertex and faces from .obj model, since we only render depth image
    :param path:
    :return: model{"pts": np.ndarray, "faces": ndarray}
    """

    model = {}
    f = open(path, "r")
    lines = f.readlines()
    vt = np.asarray(
        [
            l.rstrip().lstrip("v").lstrip().split()
            for l in lines
            if l.split(" ")[0] == "v"
        ],
        dtype=np.float64,
    )
    model["pts"] = vt

    faces_ = [l.rstrip().lstrip("f").lstrip() for l in lines if l.split(" ")[0] == "f"]

    # .obj: face index from 1 while python from 0, so we need to minus 1
    faces = np.asarray(
        [[int(item.split("/")[0]) - 1 for item in f.split(" ")] for f in faces_],
        dtype=np.float64,
    )
    model["faces"] = faces

    f.close()
    return model


# Functions to calculate transformation matrices
# Note that OpenGL expects the matrices to be saved column-wise
# (Ref: http://www.songho.ca/opengl/gl_transform.html)
# -------------------------------------------------------------------------------
# Model-view matrix
def _compute_model_view(model, view):
    return np.dot(model, view)


# Model-view-projection matrix


def _compute_model_view_proj(model, view, proj):
    return np.dot(np.dot(model, view), proj)


# Normal matrix (Ref: http://www.songho.ca/opengl/gl_normaltransform.html)


def _compute_normal_matrix(model, view):
    return np.linalg.inv(np.dot(model, view)).T


# Conversion of Hartley-Zisserman intrinsic matrix to OpenGL projection matrix
# -------------------------------------------------------------------------------
# Ref:
# 1) https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL
# 2) https://github.com/strawlab/opengl-hz/blob/master/src/calib_test_utils.py


def _compute_calib_proj(K, x0, y0, w, h, nc, fc, window_coords="y_down"):
    """
    :param K: Camera matrix.
    :param x0, y0: The camera image origin (normally (0, 0)).
    :param w: Image width.
    :param h: Image height.
    :param nc: Near clipping plane.
    :param fc: Far clipping plane.
    :param window_coords: 'y_up' or 'y_down'.
    :return: OpenGL projection matrix.
    """
    depth = float(fc - nc)
    q = -(fc + nc) / depth
    qn = -2 * (fc * nc) / depth

    # Draw our images upside down, so that all the pixel-based coordinate
    # systems are the same
    if window_coords == "y_up":
        proj = np.array(
            [
                [2 * K[0, 0] / w, -2 * K[0, 1] / w, (-2 * K[0, 2] + w + 2 * x0) / w, 0],
                [0, -2 * K[1, 1] / h, (-2 * K[1, 2] + h + 2 * y0) / h, 0],
                # This row is standard glPerspective and sets near and far planes
                [0, 0, q, qn],
                [0, 0, -1, 0],
            ]
        )  # This row is also standard glPerspective

    # Draw the images right side up and modify the projection matrix so that OpenGL
    # will generate window coords that compensate for the flipped image coords
    else:
        assert window_coords == "y_down"
        proj = np.array(
            [
                [2 * K[0, 0] / w, -2 * K[0, 1] / w, (-2 * K[0, 2] + w + 2 * x0) / w, 0],
                [0, 2 * K[1, 1] / h, (2 * K[1, 2] - h + 2 * y0) / h, 0],
                # This row is standard glPerspective and sets near and far planes
                [0, 0, q, qn],
                [0, 0, -1, 0],
            ]
        )  # This row is also standard glPerspective
    return proj.T


def render_depth(model, im_size, K, R, t, clip_near=100, clip_far=2000):
    assert {"pts", "faces"}.issubset(set(model.keys()))
    shape = (im_size[1], im_size[0])

    # --------------------------------------------- Create buffers ----------------------------------------
    colors = np.ones((model["pts"].shape[0], 3), np.float32) * 0.5

    vertices_type = [
        ("a_position", np.float32, 3),
        ("a_color", np.float32, colors.shape[1]),
    ]
    vertices = np.array(list(zip(model["pts"], colors)), vertices_type)

    vertex_buffer = vertices.view(gloo.VertexBuffer)
    index_buffer = model["faces"].flatten().astype(np.uint32).view(gloo.IndexBuffer)

    # --------------------------------------------- Model matrix ------------------------------------------
    mat_model = np.eye(4, dtype=np.float32)  # From object space to world space

    # View matrix (transforming also the coordinate system from OpenCV to
    # OpenGL camera space)
    mat_view = np.eye(4, dtype=np.float32)  # From world space to eye space
    mat_view[:3, :3], mat_view[:3, 3] = R, t.squeeze()
    yz_flip = np.eye(4, dtype=np.float32)
    yz_flip[1, 1], yz_flip[2, 2] = -1, -1
    mat_view = yz_flip.dot(mat_view)  # OpenCV to OpenGL camera system
    mat_view = mat_view.T  # OpenGL expects column-wise matrix format

    # Projection matrix
    mat_proj = _compute_calib_proj(K, 0, 0, im_size[0], im_size[1], clip_near, clip_far)

    # --------------------------------------------- Create window ------------------------------------------
    # config = app.configuration.Configuration()
    # Number of samples used around the current pixel for multisample
    # anti-aliasing (max is 8)
    # config.samples = 8
    # config.profile = "core"
    # window = app.Window(config=config, visible=False)
    window = app.Window(visible=False)

    global depth
    depth = None

    @window.event
    def on_draw(dt):
        window.clear()

        # Render depth image
        global depth
        depth = draw_depth(
            shape, vertex_buffer, index_buffer, mat_model, mat_view, mat_proj
        )

    app.run(framecount=0)  # The on_draw function is called framecount+1 times
    window.close()
    gloo.VertexBuffer.delete(vertex_buffer)
    gloo.IndexBuffer.delete(index_buffer)

    return depth
