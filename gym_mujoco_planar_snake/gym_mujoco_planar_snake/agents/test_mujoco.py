#import multiprocessing as mp
import gym

import numpy as np
import gym_mujoco_planar_snake

#from mujoco_py import MjSim, MjRenderPool, load_model_from_path
#from mujoco_py.mjviewer import MjViewer

import time

import matplotlib.pyplot as plt
from skimage import color
from skimage import transform

from baselines import logger
import os.path as osp

def test():
    # 15
    #env = gym.make('Humanoid-v2')
    #env = gym.make('Mujoco-planar-snake-cars-cam-line-v1')

    env = gym.make('Mujoco-planar-snake-cars-angle-line-v1')

    #env = gym.make('Mujoco-planar-snake-cars-cam-dist-wave-v1')

    ob = env.reset()
    #print(ob)

    print("actionspace", env.action_space)
    print("observationspace", env.observation_space)

    step = 0
    img_plot = None
    dict_list_infos = None

    #sim = MjSim(env.model)
    #viewer = MjViewer(sim)

    #action = env.action_space.sample()

    #sensors = ["sensor_actuatorfrc_rot01z", "sensor_jointpos_rot01z", "sensor_jointvel_rot01z", "sensor_actuatorfrc_rot02z"]
    #sensors = ["sensor_actuatorfrc_joint01", "sensor_actuatorfrc_joint02", "sensor_actuatorfrc_joint03"]
    #sensor_idx = list(map(lambda x: env.unwrapped.model.sensor_name2id(x), sensors))

    #logger.configure()
    #test_log_dir = osp.join(logger.get_dir(), '../../')
    #print('test_log_dir',test_log_dir)


    max_reward = 0
    max_power = 0
    max_joint_power = 0
    max_joint_velocitys = 0
    max_actuatorfrcs = 0
    max_energy0 = 0

    minus = True
    action = env.action_space.sample()
    while True:
        if step % 10 == 0:
            action = env.action_space.sample()

            #action[1:] = 0
            #action[0] = .68
            #action[0] = 1.5
            #action[0] = 0.0
            #action[:] = 0
            #action[4] = 1.5
            """
            if minus:
                action[0] = -1.5
            else:
                action[0] = 1.5
            minus = not(minus)
            """


        #print('action', action)

        #action[:] = 0

        # head_cam angle 90, head 0-1.5 -> 0- 90... if 1 then on 60d
        #action[0] = 1.0

        #action[0] = -1.5
        #action[1] = 1.5
        #action[4] = 1.5
        #action[5] = 1.5

        #action[:] = 0
        #action[3:] = 0.0

        #action[:6] = 0
        #action[0:] = 0.0

        #action[-2] = 1
        #action[-1] = 2



        ob, reward, done, info = env.step(action)




        # kinetic and potential energy
        #sensordata = env.unwrapped.data.sensordata
        # qfrc_actuator = actuatorfrcs
        energy = env.unwrapped.data.energy



        #print('qfrc_actuator', env.unwrapped.data.qfrc_actuator)




        if max_power < info['power']:
            max_power = info['power']

        if max_joint_power < np.max(info['actuatorfrcs']):
            max_joint_power = np.max(info['actuatorfrcs'])

        if max_joint_velocitys < np.max(info['joint_velocities']):
            max_joint_velocitys = np.max(info['joint_velocities'])

        if max_actuatorfrcs < np.max(info['actuatorfrcs']):
            max_actuatorfrcs = np.max(info['actuatorfrcs'])

        if max_energy0 < energy[0]:
            max_energy0 = energy[0]

        """
        print('--------------------------')
        #print("sesordata " + str(sensordata), "energy " + str(energy))
        print("energy " + str(energy))
        #print("ob " + str(ob[0:1]))


        vs = list([float(info[x]) for x in ['reward', 'velocity', 'power', 'mean_actuatorfrcs', 'sensor_head_velocity']])
        print(
            'reward {d[0]:.4f}, velocity {d[1]:.4f}, power {d[2]:.4f}, mean_actuatorfrcs {d[3]:.4f}, sensor_head_velocity{d[4]:.4f}'.format(
                d=vs))

        print("joint_powers" + str(info['joint_powers']))

        print("max_actuatorfrcs " + str(max_actuatorfrcs),
              "max power " + str(max_power),
              "max_energy0 " + str(max_energy0),
              "max_joint_power " + str(max_joint_power),
              "max_joint_velocitys "+ str(max_joint_velocitys))
            
        """


        #print(ob, reward, done, info)
        #print(len(ob))
        #print(info)
        #print(action[8])

        """
        img = ob[-1*16:] * 1
        #img = img.reshape(16, 5, 1)[::-1, :, :]
        img = img.reshape(1, 16, 1)
        #img = img[::-1, :, :]# original image is upside-down, so flip it
        img = img[:, :, 0]
        """
        
        #[0.38380157  0.38380157  0.38380157  0.38380157  0.38380157  0.38380157
        #0.38380157  0.38380157  0.38380157  0.38380157  0.38380157  0.38380157
        #0.38380157  0.38380157  0.38380157  0.38380157]

        #print("-----------------------------------------", info['joints_pos'][0])
        #print("------------angle_difference ", info['angle_difference'])
        #print("------------angle_difference_normalized ", info['angle_difference_normalized'])
        #print("------------angle_t ", info['angle_t'])

        """
        img = info['obs_img']
        #img = img.reshape(1, 16, 1)
        #img = img.reshape(1, 32, 1)
        img = img.reshape(1, 32, 1)
        img = img[:, :, 0]

        # hue, saturation, lightness
        #img_hsv = img_hsv[:, :, 1]
        print(img)

        if step%10==0:
            print(str(time.time()-action)+" s delta")
            #print(ob[-16:])
            #img = img * 256
            plt.imshow(img, cmap='gray', animated=True, label="aaa2"+str(step))
            plt.title('step: ' + str(step))
            plt.colorbar()
            plt.savefig('aa2.png')
            plt.clf()
            #plt.ion()

        """
        env.render()
        #env.env.get_sim().render()
        #env.sim.render()
        #img_plot.set_data(img)
        #plt.draw()
        #plt.draw()

        step += 1
        #print(step)

        #rgb_array = env.render(mode="rgb_array")

        #if step % 250 == 0:
        #    env.reset()
            # env.render()


if __name__ == '__main__':
#    mp.freeze_support()
#    mp.set_start_method('spawn')

    test()