import gym
import numpy as np
from gym import error, spaces, utils
from gym.envs.mujoco import mujoco_env
import glfw

import six
import math

from skimage import color
from skimage import transform
import tkinter # all fine

from mujoco_py.builder import cymj

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

class MujocoHeadCam():
    """
    deprecated
    just use:
    img = self.sim.render(camera_name='head', width=100, height=100, depth=False)
    """


    head_cam_viewer = None

    def get_head_cam_image(self, width = 32, height = 20, device_id = 0):
        #img = self.render_head_cam()

        #gray = color.rgb2gray(img)  # convert to gray (now become 0-1)
        #gray_resized = transform.resize(gray, (84, 84))  # resize

        #return img
        #data, width, height = self.get_head_cam_viewer().get_image()
        #return np.fromstring(data, dtype='uint8').reshape(height, width, 3)[::-1, :, :]

    #def render_head_cam(self):
        """
        old
        self.get_head_cam_viewer().render()
        data, width, height = self.get_head_cam_viewer().get_image()
        return np.fromstring(data, dtype='uint8').reshape(height, width, 3)[::-1, :, :]
        """

        # window size used for old mujoco-py:
        #width, height = 16, 16
        self.get_head_cam_viewer().render(width, height, device_id)
        data = self.get_head_cam_viewer().read_pixels(width, height, depth=False)

        # the right was that not work
        # data = self.sim.render(width=width1, height=height1, camera_name='head', depth=False, mode='offscreen', device_id=0)

        # original image is upside-down, so flip it
        return data[::-1, :, :]

    def get_head_cam_viewer(self):

        if self.head_cam_viewer is None:
            #self.head_cam_viewer = mujoco_py.MjViewer(init_width=100, init_height=100, go_fast=True)

            #self.head_cam_viewer = mujoco_py.MjViewer(self.sim)

            #self.head_cam_viewer = cymj.MjRenderContextWindow(self.sim)
            #glfw.window_hint(glfw.VISIBLE, 0)
            self.head_cam_viewer = cymj.MjRenderContextOffscreen(self.sim, 0)
            #size = glfw.get_window_size(self.head_cam_viewer.opengl_context.window)
            #glfw.set_window_size(self.head_cam_viewer.opengl_context.window, 800, 600)
            #glfw.set_window_pos(self.head_cam_viewer.opengl_context.window, 500, 500)

            self.head_cam_viewer_setup()

            # hides it but does not give images?
            #glfw.hide_window(self.head_cam_viewer.opengl_context.window)

        return self.head_cam_viewer

    def head_cam_viewer_close(self):
        if self.get_head_cam_viewer() is not None:
            #glfw.window_should_close()
            glfw.set_window_size_callback(self.head_cam_viewer.opengl_context.window, None)

            glfw.destroy_window(self.head_cam_viewer.opengl_context.window)
            #self.get_head_cam_viewer().finish()
            #self.head_cam_viewer = None

    def head_cam_viewer_setup(self):
        self.head_cam_viewer.cam.fixedcamid = 0
        self.head_cam_viewer.cam.distance = self.model.stat.extent * 1.25
        self._hide_overlay = True
        #1+1

    #def render(self, mode='human', close=False):
    #    self.render_head_cam(mode='human', close=False)
    #    return super().render(mode)


    """
    def reset(self):
        ob = super().reset()

        if self.head_cam_viewer is not None:
            self.head_cam_viewer_setup()

        return ob
    """


