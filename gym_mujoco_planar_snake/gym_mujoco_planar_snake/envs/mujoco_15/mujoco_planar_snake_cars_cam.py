import multiprocessing as mp
import os
import math

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.robotics import rotations
from mujoco_py import MjViewer, MjSim, MjRenderPool, MjRenderContextOffscreen
import copy

from gym_mujoco_planar_snake.envs.mujoco_15.mujoco_head_cam import MujocoHeadCam
from gym_mujoco_planar_snake.benchmark.tracks_generator import TracksGenerator
from skimage import color
from skimage import transform

from gym_mujoco_planar_snake.benchmark.plot_data import plot_head_cam as phc


class MujocoPlanarSnakeCarsEnv(mujoco_env.MujocoEnv, MujocoHeadCam, utils.EzPickle):

    def __init__(self):

        file = self.get_mjcf_file()

        self.target_sliders = ['target_slider_x', 'target_slider_y']
        self.target_sliders_idx = None
        self.geom_target_ball_name = 'target_ball_geom'
        self.geom_target_ball_idx = None

        self.camera_target_name = 'track_ball2'
        self.camera_target_idx = None

        self.sliders = ['slider_x', 'slider_y', 'slider_z']
        self.sliders_idx = None

        self.joints = ['joint01', 'joint02', 'joint03', 'joint04', 'joint05', 'joint06', 'joint07', 'joint08']
        self.joints_idx = None

        #self.wheels = ['joint_wheel_l_00', 'joint_wheel_r_00']
        self.wheels = list(np.array([['joint_wheel_l_0'+str(x), 'joint_wheel_r_0'+str(x)] for x in range(9)]).flat)
        self.joint_wheels_idx = None

        self.sensors_actuatorfrc = ["sensor_actuatorfrc_joint01", "sensor_actuatorfrc_joint02", "sensor_actuatorfrc_joint03", "sensor_actuatorfrc_joint04", "sensor_actuatorfrc_joint05", "sensor_actuatorfrc_joint06", "sensor_actuatorfrc_joint07", "sensor_actuatorfrc_joint08"]
        self.sensors_actuatorfrc_idx = None

        self.sensor_velocimeter = "sensor_velocimeter"
        self.sensor_velocimeter_idx = None

        self.sensor_accelerometer = "sensor_accelerometer"
        self.sensor_accelerometer_idx = None

        self.geom_head_name = 'head'
        self.geom_head_idx = None

        self.body_cars_names = ['car{}'.format(s) for s in range(1,10)]
        self.body_cars_idx = None

        self.joint_head_pos_name = 'hinge_z'
        self.joint_head_pos_idx = None

        self.camera_head_name = 'head'
        self.camera_head_idx = None

        self.camera_head_site = 'site_head_camera'
        self.camera_head_site_idx = None

        self.head_target_dist = 2

        self.wheels_pos = []

        frame_skip = 4 # 4

        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, file, frame_skip)

    def get_mjcf_file(self):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        file = os.path.join(ROOT_DIR, '../assets', 'planar_snake_cars_servo.xml')
        return file

    def init_idx_values(self):
        self.target_sliders_idx = list(
            map(lambda x: self.model.joint_names.index(x), self.target_sliders))
        self.sliders_idx = list(map(lambda x: self.model.joint_name2id(x), self.sliders))
        self.joints_idx = list(map(lambda x: self.model.joint_name2id(x), self.joints))
        self.joint_wheels_idx = list(map(lambda x: self.model.joint_name2id(x), self.wheels))

        self.sensors_actuatorfrc_idx = list(map(lambda x: self.model.sensor_name2id(x), self.sensors_actuatorfrc))
        self.sensor_velocimeter_idx = self.model.sensor_name2id(self.sensor_velocimeter)
        self.sensor_accelerometer_idx = self.model.sensor_name2id(self.sensor_accelerometer)
        self.geom_head_idx = self.sim.model.geom_name2id(self.geom_head_name)
        self.body_cars_idx = [self.sim.model.body_name2id(x) for x in self.body_cars_names]

        self.joint_head_pos_idx = self.model.joint_name2id(self.joint_head_pos_name)
        self.camera_head_idx = self.model.camera_names.index(self.camera_head_name)
        self.camera_target_idx = self.model.camera_names.index(self.camera_target_name)
        self.geom_target_ball_idx = self.sim.model.geom_name2id(self.geom_target_ball_name)


    def get_xy_sliders_index(self):
        x_joint_index = self.sliders_idx[0]
        y_joint_index = self.sliders_idx[1]
        return [x_joint_index, y_joint_index]

    def get_sensor_actuatorfrcs(self):
        return self.sim.data.sensordata[self.sensors_actuatorfrc_idx]

    def get_sencor_head_velocity(self):
        return self.sim.data.sensordata[self.sensor_velocimeter_idx]

    def get_joint_positions(self):
        qpos = self.sim.data.qpos
        joints_pos = qpos[self.joints_idx].flat
        return list(joints_pos)

    def get_joint_velocities(self):
        qvel = self.sim.data.qvel
        joints_vel = qvel[self.joints_idx].flat
        return list(joints_vel)

    def get_body_pos(self):
        head_x = self.sim.data.qpos.flat[self.sliders_idx[0]]
        head_y = self.sim.data.qpos.flat[self.sliders_idx[1]]
        return head_x, head_y

    def get_head_pos(self):
        pos = self.data.get_geom_xpos(self.geom_head_name)
        return pos[0], pos[1]

        #x, y= self.get_head_cam_pos() #also possible
        #x,y = self.get_body_pos()
        #return x, y

    def get_target_pos(self):
        #pos = np.array(self.get_body_com('target_ball'))
        #target_x = pos[0]
        #target_y = pos[1]

        pos = self.data.get_geom_xpos(self.geom_target_ball_name)
        return pos[0], pos[1]

        #target_x, target_y, z = self.sim.data.cam_xpos[self.camera_target_idx]
        #target_x, target_y, z = self.sim.data.qpos[:][self.geom_target_ball_idx]

        #target_x = self.sim.data.qpos.flat[self.target_sliders_idx[0]]
        #target_y = self.sim.data.qpos.flat[self.target_sliders_idx[1]]
        #return target_x, target_y

    # deprecated
    #def get_score(self):
    #    raise NotImplementedError("method needs to be defined by sub-class")

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 3
        self.viewer.go_fast = True

    def reset_model(self):
        # TODO only change snake joints
        # self.set_state(
        #    self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
        #    self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        # )

        self.set_state(
            self.init_qpos,
            self.init_qvel
        )

        self.move_ball()

        return self.get_obs()

    def close(self):
        print('close')
        self.head_cam_viewer_close()
        super().close()

    def move_ball(self):
        raise NotImplementedError("method needs to be defined by sub-class")

    def set_ball(self, pos):
        # is called every step

        target_slider_x_idx, target_slider_y_idx = self.target_sliders_idx

        # move ball_target
        qpos = self.sim.data.qpos.flat[:]
        qvel = self.sim.data.qvel.flat[:]

        qpos[target_slider_x_idx], qpos[target_slider_y_idx] = pos

        self.set_state(qpos, qvel)

    def calc_distance(self):
        head_x, head_y = self.get_head_pos()
        target_x, target_y = self.get_target_pos()
        dist = math.sqrt(((target_x - head_x) ** 2) + ((target_y - head_y) ** 2))

        return dist

    def quat_to_euler(self, quat):
        return rotations.quat2euler(quat)

    def get_head_euler_angles(self):
        head_xquat = self.sim.data.body_xquat[self.body_cars_idx[0]]
        head_euler_angle = self.quat_to_euler(head_xquat)
        return head_euler_angle

    def get_orientation_degree_z_angle(self):
        cars = [self.sim.data.body_xquat[x] for x in self.body_cars_idx]
        euler_theta_all = [self.quat_to_euler(xquat) for xquat in cars]
        degrees_z_theta_all = [math.degrees(a[2]) for a in euler_theta_all]

        theta_all = np.mean(degrees_z_theta_all)

        return theta_all

    def get_head_degrees_z_angle(self):
        return math.degrees(self.get_head_euler_angles()[2])

    def get_head_euler_z_angle(self):
        return self.get_head_euler_angles()[2]

    def get_head_cam_pos(self):
        x, y, z = self.sim.data.cam_xpos[self.camera_head_idx]

        return x, y

    def get_target_z_degree_angle(self):
        # head pos to target pos angele
        x_head_cam, y_head_cam = self.get_head_cam_pos()

        target_pos = self.get_target_pos()

        opposite = target_pos[1] - y_head_cam
        adjacent = target_pos[0] - x_head_cam

        return math.degrees(math.atan(opposite / adjacent))

    def get_head_to_target_degree_angle(self):
        # total angle
        head_angle_z = self.get_head_degrees_z_angle()
        # target_angle_z = 0
        target_angle_z = self.get_target_z_degree_angle()
        return head_angle_z - target_angle_z

    def get_head_to_target_degree_angle_difference(self):
        angle_difference = abs(self.get_head_to_target_degree_angle())
        return angle_difference

    def get_orientation_to_target_degree_angle(self):
        # total angle
        orientation_angle_z = self.get_orientation_degree_z_angle()
        # target_angle_z = 0
        target_angle_z = self.get_target_z_degree_angle()
        return orientation_angle_z - target_angle_z

    def calculate_energy_usage_normalized(self):

        # kinetic and potential energy
        #energy = self.unwrapped.data.energy

        #actuator_force = self.unwrapped.data.actuator_force  # same as sensor
        sensordata = self.get_sensor_actuatorfrcs()
        actuator_force = sensordata
        actuator_forcerange = self.unwrapped.model.actuator_forcerange

        # force_max
        force_max = actuator_forcerange.max()

        # kp not needed
        #actuator_gainprm = self.unwrapped.model.actuator_gainprm

        # gear
        actuator_gear = self.unwrapped.model.actuator_gear.max()

        # joint velocitys
        joints_vel = self.get_joint_velocities()

        max_joint_vel = 25.0

        # actuator torque
        actuator_torques = np.array(actuator_force) * actuator_gear

        # sum force
        #force_sum = np.sum(np.abs(actuator_force)) * actuator_gear
        #energy_norm = force_sum / actuator_force.size / force_max

        actuator_energy = np.mean(np.abs(actuator_torques * joints_vel))
        actuator_energies_max = force_max * actuator_gear * max_joint_vel

        actuators_energy_normalized = actuator_energy / actuator_energies_max

        return actuators_energy_normalized, actuator_energy

    def calculate_total_energy_usage_sec(self):
        # gear
        actuator_gear = self.unwrapped.model.actuator_gear.max()

        sensordata = self.get_sensor_actuatorfrcs()
        actuator_force = sensordata

        # joint velocitys
        joints_vel = self.get_joint_velocities()

        # actuator torque
        actuator_torques = np.array(actuator_force) * actuator_gear

        # P = torque * angular_velocity
        energie_array = np.abs(actuator_torques * joints_vel)
        sum_all_actuator_energy = np.sum(energie_array)

        return sum_all_actuator_energy, energie_array


    def get_obs(self):
        raise NotImplementedError("method needs to be defined by sub-class")

    def step(self):
        raise NotImplementedError("method needs to be defined by sub-class")


class MujocoPlanarSnakeCarsAngleEnv(MujocoPlanarSnakeCarsEnv):
    '''
    Environment with a head to target angle in observation space
    Used for power velocity experiment
    Careful density of model should be 600. Check the mujoco_planar_snake_cars_servo.xml file line 32
    '''

    # velocities that will be used during the training
    #target_v_array= [0.1, 0.3, 0.5, 0.7] #long snake
    #target_v_array = [0.05, 0.075, 0.1, 0.125] # working
    #target_v_array = [0.025, 0.075, 0.125, 0.175, 0.225] # good with 600
    #target_v_array = [0.05, 0.1, 0.15, 0.20, 0.25, 0.3] # medium
    #target_v_array = [0.025, 0.05, 0.1, 0.15, 0.20, 0.25, 0.3]  # worse
    #target_v_array = [0.05, 0.1, 0.15, 0.20, 0.25, 0.3] # meh 0.3 is not easy

    # final
    target_v_array = [0.05, 0.1, 0.15, 0.20, 0.25] #really good, linear

    target_v = 0.042 # dummy
    update = 0

    def __init__(self):
        super().__init__()

    def get_target_velocity(self):
        if 'target_v' in self.unwrapped.metadata:
            return self.unwrapped.metadata['target_v']
        else:
            return None

    def reset_model(self):

        #if self.target_v >= 0.8:
        #    self.target_v = -0.15

        #self.target_v += 0.05
        #self.target_v += 0.2

        if self.get_target_velocity() is not None:
            self.target_v = self.get_target_velocity()

        else:
            """
            self.target_v = self.target_v_array[self.update % len(self.target_v_array)]
            """
            if self.update >= 100:
                self.target_v = self.target_v_array[self.update % len(self.target_v_array)]
            else:
                self.target_v = 0.1 # default for first 100 episodes

            if 'target_v' in self.unwrapped.metadata:
                self.target_v = self.unwrapped.metadata['target_v']

        self.update += 1

        print(self.target_v)
        return super().reset_model()

    def get_obs(self):
        # joint angles
        joints_pos = self.get_joint_positions()

        # joint velocitys
        joints_vel = self.get_joint_velocities()


        angle = self.get_head_to_target_degree_angle()
        #orientation = self.get_orientation_to_target_degree_angle()

        sensor_velocity = self.get_sencor_head_velocity()

        #target_v_delta = self.target_v - sensor_velocity


        #add energy in OBs
        sensor_actuatorfrcs = self.get_sensor_actuatorfrcs()
        sensor_actuatorfrcs = np.abs(sensor_actuatorfrcs)
        actuator_forcerange = self.unwrapped.model.actuator_forcerange
        # gear
        actuator_gear = self.unwrapped.model.actuator_gear.max()
        # max
        force_max = actuator_forcerange.max()

        joint_velocities = self.get_joint_velocities()
        #joint_powers = np.abs(np.multiply(sensor_actuatorfrcs, joint_velocities))

        #actuators_energy_normalized = np.array(sensor_actuatorfrcs) * actuator_gear / force_max

        #total_energy, energy_array = self.calculate_total_energy_usage_sec()
        #target_v_norm = self.target_v / np.array(self.target_v_array).max()

        # torque
        sensor_actuatorfrcs = np.array(sensor_actuatorfrcs) * actuator_gear


        # final
        ob = np.concatenate([joints_pos, joints_vel, [angle], [sensor_velocity], sensor_actuatorfrcs, [self.target_v]])

        return ob

    def step(self, a):
        


        # first init
        if self.sliders_idx == None:
            self.init_idx_values()

        # careful position before step
        self.move_ball()


        distbefore = self.calc_distance()
        self.do_simulation(a, self.frame_skip)
        distafter = self.calc_distance()


        angle_difference = self.get_head_to_target_degree_angle_difference() # no + or - , no left right

        # normalize it use squared
        max_angle = 60
        angle_difference_normalized = angle_difference**2 / max_angle**2  if angle_difference < max_angle else 1.0


        # efficiency
        energy0 = self.unwrapped.data.energy #kinetic and potential energy
        sensor_actuatorfrcs = self.get_sensor_actuatorfrcs()
        mean_actuatorfrcs = np.mean(np.abs(sensor_actuatorfrcs))
        joint_velocities = self.get_joint_velocities()

        #self.get_sensor_actuatorfrcs()
        power_normalized, power = self.calculate_energy_usage_normalized()

        # efficiency
        distance = distbefore - distafter
        velocity = distance / self.dt # in m/s
        joint_powers = np.abs(np.multiply(sensor_actuatorfrcs, joint_velocities))

        # watt per sec
        total_power_sec, power_array_sec = self.calculate_total_energy_usage_sec()


        # see old git version for all the parameter tries
        #new 15
        # works great, min max must be trained
        a1 = 0.2
        a2 = 0.2
        rew_v = (1.0 - (np.abs(self.target_v - velocity) / a1)) ** (1 / a2)


        # see old git version for all the parameter tries
        #new 15
        b1 = 0.6
        rew_p = np.abs(1.0 - power_normalized) ** (b1 ** (-2.0))
        reward = rew_v * rew_p


        ob = self.get_obs()

        return ob, reward, False, dict(reward=reward,
                                       power=power,
                                       power_normalized=power_normalized,
                                       velocity=velocity,
                                       distance_delta=distance,
                                       joint_powers=joint_powers,
                                       joint_velocities= joint_velocities,
                                       abs_joint_velocities=np.abs(joint_velocities),
                                       max_joint_velocities=np.max(np.abs(joint_velocities)),
                                       actuatorfrcs=sensor_actuatorfrcs,
                                       mean_actuatorfrcs=mean_actuatorfrcs,
                                       energy0=energy0,
                                       total_power_sec=total_power_sec,
                                       power_array_sec=power_array_sec,
                                       target_v=self.target_v,
                                       angle_difference_normalized=angle_difference_normalized,
                                       sensor_head_velocity=self.get_sencor_head_velocity(),
                                       head_x=self.sim.data.qpos.flat[self.sliders_idx[0]],
                                       head_y=self.sim.data.qpos.flat[self.sliders_idx[1]],
                                       joints_pos=self.get_joint_positions(),
                                       joint_head_pos=self.sim.data.qpos[self.joint_head_pos_idx],
                                       )


class MujocoPlanarSnakeCarsCamEnv(MujocoPlanarSnakeCarsEnv, MujocoHeadCam):
    '''
        Environment with camera image in observation space
        Used for target tracking experiment
        Careful density of model should be 800. Check the mujoco_planar_snake_cars_servo.xml file line 32
    '''

    target_v = 0.42 #dummy ... has no effect (only for easier logging)
    # results to 3 dist in min and max direction

    # set target distance!
    # final
    target_distance = 4.0 # 4.5  # 4
    #target_distance = 2.5

    previous_distance = None

    update = 0
    #step_target = 0
    #step_i = 0

    def __init__(self):
        super().__init__()
        MujocoHeadCam.__init__(self)


    def reset_model(self):
        ret = super().reset_model()

        x = self.get_head_pos()[0] + self.target_distance
        self.set_ball([x, 0])

        self.update += 1
        return ret

    last_row = None
    def get_obs(self):
        # joint angles
        joints_pos = self.get_joint_positions()

        # joint velocitys
        joints_vel = self.get_joint_velocities()

        sensor_velocity = self.get_sencor_head_velocity()

        #angle_t = self.get_head_to_target_degree_angle()

        #norm_angle_t = (angle_t + 180.0) / 360.0

        #orientation = self.get_orientation_degree_z_angle()

        # not really compareable?
        # angle from head and orientation?
        #norm_orientation_angle = (angle_t - orientation + 180) / 360

        # dist
        #dist = self.calc_distance()


        # image
        #img = self.get_head_cam_image(16, 10, self.camera_head_idx)

        img = self.get_head_cam_image(32, 20, self.camera_head_idx)

        #img = self.get_head_cam_image(1680, 1050, self.camera_head_idx)
        #img = self.get_head_cam_image(8, 5, self.camera_head_idx)
        #img = self.get_head_cam_image(16, 5, self.camera_head_idx)
        #img = self.get_head_cam_image(64, 40, self.camera_head_idx)

        # rgb to gray
        img_gray = color.rgb2gray(img)

        # select row
        #img_gray = img_gray[4] # 16x10
        img_gray = img_gray[9] # 32x20
        # img_gray = img_gray[19] # 64x40


        # with Camera
        # ob = np.concatenate([joints_pos, joints_vel, [sensor_velocity], [dist_diff], img_gray])
        # ob = np.concatenate([joints_pos, joints_vel, [sensor_velocity], [dist_diff], [angle], img_gray])
        # ob = np.concatenate([joints_pos, joints_vel, [sensor_velocity], img_gray])

        # final
        ob = np.concatenate([joints_pos, joints_vel, [sensor_velocity], img_gray])


        # with fixed distance
        #ob = np.concatenate([joints_pos, joints_vel, [sensor_velocity], [norm_angle_t], [norm_orientation_angle]])

        return ob

    def step(self, a):
        # first init
        if self.sliders_idx == None:
            self.init_idx_values()


        # careful position before step
        target_x0, target_y0 = self.get_target_pos()
        self.move_ball()

        distbefore = self.calc_distance()
        head_x0, head_y0 = self.get_head_pos()


        # do sim
        self.do_simulation(a, self.frame_skip)

        distafter = self.calc_distance()
        head_x1, head_y1 = self.get_head_pos()
        target_x1, target_y1 = self.get_target_pos()


        angle_difference = self.get_head_to_target_degree_angle_difference()
        angle_t = self.get_head_to_target_degree_angle()
        head_angle_z = self.get_head_degrees_z_angle()
        target_angle_z = self.get_target_z_degree_angle()

        # normalize it use squared
        max_angle = 60
        angle_difference_normalized = angle_difference ** 2 / max_angle ** 2 if angle_difference < max_angle else 1.0

        # efficiency
        energy0 = self.unwrapped.data.energy  # kinetic and potential energy
        sensor_actuatorfrcs = self.get_sensor_actuatorfrcs()
        mean_actuatorfrcs = np.mean(np.abs(sensor_actuatorfrcs))
        joint_velocities = self.get_joint_velocities()

        # self.get_sensor_actuatorfrcs()
        power_normalized, power = self.calculate_energy_usage_normalized()

        # efficiency
        distance_delta = distbefore - distafter
        velocity_to_target = distance_delta / self.dt  # in m/s wrong with moving target
        velocity = math.sqrt(((head_x0-head_x1)**2)+((head_y0-head_y1)**2)) / self.dt
        #print(velocity, head_x0, head_x1)
        
        joint_powers = np.abs(np.multiply(sensor_actuatorfrcs, joint_velocities))

        target_distance_delta = math.sqrt(((target_x0 - target_x1) ** 2) + ((target_y0 - target_y1) ** 2))
        #print(target_velocity, target_x0, target_x1)

        # watt per sec
        total_power_sec, power_array_sec = self.calculate_total_energy_usage_sec()



        # DISTANCE

        # TODO from track generator
        target_dist_max = 6.0  # 7.5  # 8
        target_dist_min = 2.0
        distance_range = (target_dist_max - target_dist_min)/2


        # check old git versions for all reward tries

        # test
        # t6
        # final
        diff_before = np.abs(self.target_distance - distbefore)
        diff_after = np.abs(self.target_distance - distafter)
        #reward = (1 - diff_after / distance_range) / self.dt - (1 - diff_before / distance_range) / self.dt
        # same as
        reward = (diff_before - diff_after) / distance_range #* self.dt


        ob = self.get_obs()


        #  which one???? head or cam pos?
        # almost same as self.get_head_cam_pos()
        #head_x = self.sim.data.qpos.flat[self.sliders_idx[0]],
        #head_y = self.sim.data.qpos.flat[self.sliders_idx[1]]

        head_x, head_y = self.get_head_pos()
        #cam_x, cam_y = self.get_head_cam_pos()

        #print(head_x, head_x2, cam_x, head_y, head_y2, cam_y)

        # same as old variant but with function call
        target_x, target_y = self.get_target_pos()

        img = ob[-32:]

        return ob, reward, False, dict(reward=reward,
                                       power=power,
                                       power_normalized=power_normalized,
                                       velocity=velocity,#velocity_to_target, # velocity
                                       distance_delta=distance_delta,
                                       head_target_distance=distafter,
                                       head_target_distance_diff=self.target_distance - distafter,
                                       target_distance=self.target_distance,
                                       target_distance_delta=target_distance_delta,
                                       joint_powers=joint_powers,
                                       joint_velocities=joint_velocities,
                                       obs_img=img,
                                       abs_joint_velocities=np.abs(joint_velocities),
                                       max_joint_velocities=np.max(np.abs(joint_velocities)),
                                       actuatorfrcs=sensor_actuatorfrcs,
                                       mean_actuatorfrcs=mean_actuatorfrcs,
                                       energy0=energy0,
                                       total_power_sec=total_power_sec,
                                       target_v=self.target_v,
                                       #injured_joint=self.injured_joint,
                                       head_angle_z=head_angle_z,
                                       target_angle_z=target_angle_z,
                                       orientation=self.get_orientation_degree_z_angle(),
                                       angle_t=angle_t,
                                       angle_difference=angle_difference,
                                       angle_difference_normalized=angle_difference_normalized,
                                       sensor_head_velocity=self.get_sencor_head_velocity(),
                                       head_x=head_x,
                                       head_y=head_y,
                                       target_x=target_x,
                                       target_y=target_y,
                                       joints_pos=self.get_joint_positions(),
                                       joints_pos_mean=np.mean(self.get_joint_positions()),
                                       joint_head_pos=self.sim.data.qpos[self.joint_head_pos_idx],
                                       )


# target tracks

class MujocoPlanarSnakeCarsAngleLineEnv(MujocoPlanarSnakeCarsAngleEnv):
    # Follow straight line

    def move_ball(self):
        x = self.sim.data.qpos.flat[self.sliders_idx[0]]
        x += self.head_target_dist
        y = 0
        self.set_ball([x, y])
        return [x, y]

class MujocoPlanarSnakeCarsCamLineEnv(MujocoPlanarSnakeCarsCamEnv):
    # Follow straight line

    def move_ball(self):
        x = self.sim.data.qpos.flat[self.sliders_idx[0]]
        x += self.head_target_dist
        y = 0
        self.set_ball([x, y])
        return [x, y]

class MujocoPlanarSnakeCarsCamLineDistanceEnv(MujocoPlanarSnakeCarsCamEnv):
    #Follow straight line distance

    tracks_generator = TracksGenerator()
    def move_ball(self):
        head_x, head_y = self.get_head_pos()
        target_x, target_y = self.get_target_pos()
        #x += self.head_target_dist

        x, y = self.tracks_generator.gen_line_step(head_x, head_y, target_x, target_y, self.dt)

        self.set_ball([x, y])
        return [x, y]


class MujocoPlanarSnakeCarsAngleWaveEnv(MujocoPlanarSnakeCarsAngleEnv):
    """
    Follow straight for some units then do a sinus
    """

    def move_ball(self):
        x = self.sim.data.qpos.flat[self.sliders_idx[0]]
        x += self.head_target_dist
        start_sin_at = 5
        period = 0.25
        amplitude = 3

        y = 0
        if x >= start_sin_at:
            y = amplitude * np.sin(period * (x - start_sin_at))

        self.set_ball([x, y])
        return [x, y]

class MujocoPlanarSnakeCarsCamWaveEnv(MujocoPlanarSnakeCarsCamEnv):
    '''
    Follow straight for some units then do a sinus
    '''

    def move_ball(self):
        x = self.sim.data.qpos.flat[self.sliders_idx[0]]
        x += self.head_target_dist
        start_sin_at = 5
        period = 0.25
        amplitude = 3

        y = 0
        if x >= start_sin_at:
            y = amplitude * np.sin(period * (x - start_sin_at))

        self.set_ball([x, y])
        return [x, y]

class MujocoPlanarSnakeCarsCamWaveDistanceEnv(MujocoPlanarSnakeCarsCamEnv):
    #Follow straight for some units then do a sinus

    tracks_generator = TracksGenerator()
    def move_ball(self):
        head_x, head_y = self.get_head_pos()
        target_x, target_y = self.get_target_pos()
        #x += self.head_target_dist

        x, y = self.tracks_generator.gen_wave_step(head_x, head_y, target_x, target_y, self.dt)

        self.set_ball([x, y])
        return [x, y]


class MujocoPlanarSnakeCarsAngleZigzagEnv(MujocoPlanarSnakeCarsAngleEnv):
    #Follow straight for some units then do a zigzag

    def move_ball(self):
        x = self.sim.data.qpos.flat[self.sliders_idx[0]]
        x += self.head_target_dist
        y = 0
        start_sin_at = 5 #8

        c = 10
        d = 0.1

        def a():
            return c * (-1 + 2 *math.fmod(math.floor(d*x), 2))

        def b():
            return - c * math.fmod(math.floor(d*x),2)

        if x >= start_sin_at:
            y = (d*x - math.floor(d*x)) * a() + b() + c/2

        self.set_ball([x, y])

        return [x, y]

class MujocoPlanarSnakeCarsCamZigzagEnv(MujocoPlanarSnakeCarsCamEnv):
    """
    Follow straight for some units then do a zigzag
    """

    def move_ball(self):
        x = self.sim.data.qpos.flat[self.sliders_idx[0]]
        x += self.head_target_dist
        y = 0
        start_sin_at = 5  # 8

        c = 10
        d = 0.1

        def a():
            return c * (-1 + 2 * math.fmod(math.floor(d * x), 2))

        def b():
            return - c * math.fmod(math.floor(d * x), 2)

        if x >= start_sin_at:
            y = (d * x - math.floor(d * x)) * a() + b() + c / 2

        self.set_ball([x, y])

        return [x, y]

class MujocoPlanarSnakeCarsCamZigzagDistanceEnv(MujocoPlanarSnakeCarsCamEnv):
    #Follow straight for some units then do a zigzag
    tracks_generator = TracksGenerator()

    def move_ball(self):
        head_x, head_y = self.get_head_pos()
        target_x, target_y = self.get_target_pos()

        x, y = self.tracks_generator.gen_zigzag_step(head_x, head_y, target_x, target_y, self.dt)

        self.set_ball([x, y])
        return [x, y]


class MujocoPlanarSnakeCarsAngleCircleEnv(MujocoPlanarSnakeCarsAngleEnv):
    """
    Follow straight for some units then do a zigzag
    """

    def move_ball(self):
        x_head = self.sim.data.qpos.flat[self.sliders_idx[0]]
        y_head = self.sim.data.qpos.flat[self.sliders_idx[1]]
        start_sin_at = 10 #8

        radius = start_sin_at

        # before circle
        if math.sqrt(x_head**2 + y_head**2) <= radius - self.head_target_dist:
            x = x_head + self.head_target_dist
            y = 0
            #print('before circle')
        else:
            #a = math.degrees(math.atan(y_head / x_head))
            alpha_head = math.degrees(math.atan2(y_head , x_head))

            #print('a', a)

            # angle alpha at a known distance
            # Law of cosines
            # b^2 = a^2 + c^2 - 2ac * cos beta
            a = self.head_target_dist
            b = radius
            c = radius
            cos_alpha = (b**2 + c**2 - a**2) / (2 * b * c)
            beta = math.degrees(math.acos(cos_alpha))
            #print('beta',beta) # 11.478

            alpha_target = alpha_head + beta

            x = math.cos(math.radians(alpha_target)) * radius
            y = math.sin(math.radians(alpha_target)) * radius


        self.set_ball([x, y])

        return [x, y]

class MujocoPlanarSnakeCarsCamCircleEnv(MujocoPlanarSnakeCarsCamEnv):
    """
    Follow straight for some units then do a zigzag
    """

    def move_ball(self):
        x_head = self.sim.data.qpos.flat[self.sliders_idx[0]]
        y_head = self.sim.data.qpos.flat[self.sliders_idx[1]]
        start_sin_at = 10 #8

        radius = start_sin_at

        # before circle
        if math.sqrt(x_head**2 + y_head**2) <= radius - self.head_target_dist:
            x = x_head + self.head_target_dist
            y = 0
            #print('before circle')
        else:
            #a = math.degrees(math.atan(y_head / x_head))
            alpha_head = math.degrees(math.atan2(y_head , x_head))

            #print('a', a)

            # angle alpha at a known distance
            # Law of cosines
            # b^2 = a^2 + c^2 - 2ac * cos beta
            a = self.head_target_dist
            b = radius
            c = radius
            cos_alpha = (b**2 + c**2 - a**2) / (2 * b * c)
            beta = math.degrees(math.acos(cos_alpha))
            #print('beta',beta) # 11.478

            alpha_target = alpha_head + beta

            x = math.cos(math.radians(alpha_target)) * radius
            y = math.sin(math.radians(alpha_target)) * radius

            #print(x,y)


        """
        a 89.77782898950618
        -2.550716196152078 9.669221627757194
        a -89.92577274584525
        2.600701941061671 -9.65589713148178
        """

        self.set_ball([x, y])

        return [x, y]

class MujocoPlanarSnakeCarsCamCircleDistanceEnv(MujocoPlanarSnakeCarsCamEnv):
    #TODO not  working
    #Follow straight for some units then do a zigzag
    tracks_generator = TracksGenerator()

    def move_ball(self):
        head_x, head_y = self.get_head_pos()
        target_x, target_y = self.get_target_pos()

        x, y = self.tracks_generator.gen_circle_step(head_x, head_y, target_x, target_y, self.dt)

        self.set_ball([x, y])
        return [x, y]


class MujocoPlanarSnakeCarsAngleRandomEnv(MujocoPlanarSnakeCarsAngleEnv):
    """
    Follow straight for some units then do a zigzag
    """

    do_random = False
    current_rand = 0
    stepnr = 0

    def move_ball(self):
        x_head = self.sim.data.qpos.flat[self.sliders_idx[0]]
        y_head = self.sim.data.qpos.flat[self.sliders_idx[1]]
        # x += self.head_target_dist

        head_degree_angle = self.get_head_degrees_z_angle()

        start_random_at = 0   # 8
        radius = self.head_target_dist
        max_degree_target_change = 20
        degree_target_steps_duration = 80

        # before random
        if not self.do_random:
            self.do_random = x_head >= start_random_at
            x = x_head + self.head_target_dist
            y = 0
            # print('before random')
        else:

            if self.stepnr % degree_target_steps_duration == 0:
                self.current_rand = self.np_random.randint(-max_degree_target_change/4, max_degree_target_change)
                #print(self.current_rand)

            beta = head_degree_angle + self.current_rand

            x = x_head + math.cos(math.radians(beta)) * radius
            y = y_head + math.sin(math.radians(beta)) * radius

        self.set_ball([x, y])
        self.stepnr += 1

        return [x, y]

class MujocoPlanarSnakeCarsCamRandomEnv(MujocoPlanarSnakeCarsCamEnv):
    """
    Follow straight for some units then do a zigzag
    """

    do_random = False
    current_rand = 0
    stepnr = 0

    def move_ball(self):
        x_head = self.sim.data.qpos.flat[self.sliders_idx[0]]
        y_head = self.sim.data.qpos.flat[self.sliders_idx[1]]
        # x += self.head_target_dist

        head_degree_angle = self.get_head_degrees_z_angle()
        #target_degree_angle_z = self.get_head_to_target_z_degree_angle()

        start_random_at = 0   # 8
        radius = self.head_target_dist
        max_degree_target_change = 20
        degree_target_steps_duration = 80

        # before random
        if not self.do_random:
            self.do_random = x_head >= start_random_at
            x = x_head + self.head_target_dist
            y = 0
            # print('before random')
        else:

            if self.stepnr % degree_target_steps_duration == 0:
                self.current_rand = self.np_random.randint(-max_degree_target_change/4, max_degree_target_change)
                #print(self.current_rand)

            beta = head_degree_angle + self.current_rand

            x = x_head + math.cos(math.radians(beta)) * radius
            y = y_head + math.sin(math.radians(beta)) * radius

        self.set_ball([x, y])
        self.stepnr += 1

        return [x, y]

class MujocoPlanarSnakeCarsCamRandomDistanceEnv(MujocoPlanarSnakeCarsCamEnv):
    """
    Follow straight for some units then do a zigzag
    """
    current_rand = 0
    track_seed = 5

    tracks_generator = None

    def reset_model(self):
        self.tracks_generator = None

        return super().reset_model()

    def seed(self, seed=None):
        #print("seed", seed)
        #self.track_seed = seed

        return super().seed(seed)

    def move_ball(self):
        head_x, head_y = self.get_head_pos()
        target_x, target_y = self.get_target_pos()

        if self.tracks_generator is None:
            self.tracks_generator = TracksGenerator()

            self.track_seed += 1

        x, y = self.tracks_generator.gen_random_step(head_x, head_y, target_x, target_y, self.dt, seed=self.track_seed, ignore_head=False)

        self.set_ball([x, y])
        return [x, y]



class MujocoPlanarSnakeCarsAngleCursorEnv(MujocoPlanarSnakeCarsAngleEnv):
    """
    Control target with mouse. enjoy only
    """

    def __init__(self):
        super().__init__()

    def get_score(self):
        target_pos = self.get_body_com('target_ball')
        return target_pos

    def move_ball(self):
        import glfw
        x_head = self.sim.data.qpos.flat[self.sliders_idx[0]]
        y_head = self.sim.data.qpos.flat[self.sliders_idx[1]]

        the_viewer = self._get_viewer()
        # c = glfw._GLFWcursorposfun
        # glfw.set_cursor_pos_callback(the_viewer.opengl_context.window, c)

        window_size = glfw.get_window_size(the_viewer.opengl_context.window)
        cursor_pos = glfw.get_cursor_pos(the_viewer.opengl_context.window)
        cursor_pos = [cursor_pos[0], -cursor_pos[1]]

        # norm
        norm_cursor_pos = np.divide(cursor_pos, window_size)
        # center
        norm_center_cursor_pos = [norm_cursor_pos[0] - 0.5, norm_cursor_pos[1] + 0.5]

        # free cursor
        """
        """
        # scale to zoom and viewangle
        x = norm_center_cursor_pos[0] * 22
        y = norm_center_cursor_pos[1] * 12

        x += x_head
        y += y_head

        # cursor on target radius
        """
        radius = self.head_target_dist
        alpha_head = math.degrees(math.atan2(norm_center_cursor_pos[1], norm_center_cursor_pos[0]))

        # angle alpha at a known distance
        # Law of cosines
        # b^2 = a^2 + c^2 - 2ac * cos beta
        a = 0
        b = radius
        c = radius
        cos_alpha = (b ** 2 + c ** 2 - a ** 2) / (2 * b * c)
        beta = math.degrees(math.acos(cos_alpha))
        # print('beta',beta) # 11.478

        alpha_target = alpha_head + beta

        x = math.cos(math.radians(alpha_target)) * radius + x_head
        y = math.sin(math.radians(alpha_target)) * radius + y_head
        """

        self.set_ball([x, y])
        return [x, y]

class MujocoPlanarSnakeCarsCamCursorEnv(MujocoPlanarSnakeCarsCamEnv):
    """
    Control target with mouse. enjoy only
    """

    def __init__(self):
        super().__init__()

    def get_score(self):
        target_pos = self.get_body_com('target_ball')
        return target_pos

    def move_ball(self):
        import glfw
        x_head = self.sim.data.qpos.flat[self.sliders_idx[0]]
        y_head = self.sim.data.qpos.flat[self.sliders_idx[1]]


        the_viewer = self._get_viewer()
        #c = glfw._GLFWcursorposfun
        #glfw.set_cursor_pos_callback(the_viewer.opengl_context.window, c)

        window_size = glfw.get_window_size(the_viewer.opengl_context.window)
        cursor_pos = glfw.get_cursor_pos(the_viewer.opengl_context.window)
        cursor_pos = [cursor_pos[0], -cursor_pos[1]]

        # norm
        norm_cursor_pos = np.divide(cursor_pos, window_size)
        # center
        norm_center_cursor_pos = [norm_cursor_pos[0] - 0.5, norm_cursor_pos[1] + 0.5]


        # free cursor
        """
        """
        # scale to zoom and viewangle
        x = norm_center_cursor_pos[0] * 22
        y = norm_center_cursor_pos[1] * 12
        
        x += x_head
        y += y_head

        # cursor on target radius
        """
        radius = self.head_target_dist
        alpha_head = math.degrees(math.atan2(norm_center_cursor_pos[1], norm_center_cursor_pos[0]))

        # angle alpha at a known distance
        # Law of cosines
        # b^2 = a^2 + c^2 - 2ac * cos beta
        a = 0
        b = radius
        c = radius
        cos_alpha = (b ** 2 + c ** 2 - a ** 2) / (2 * b * c)
        beta = math.degrees(math.acos(cos_alpha))
        # print('beta',beta) # 11.478

        alpha_target = alpha_head + beta

        x = math.cos(math.radians(alpha_target)) * radius + x_head
        y = math.sin(math.radians(alpha_target)) * radius + y_head
        """

        self.set_ball([x, y])
        return [x, y]