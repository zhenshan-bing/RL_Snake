#!/usr/bin/env python
from baselines import bench
from baselines.common import set_global_seeds
import gym, logging
from baselines import logger
import tensorflow as tf

from collections import Counter
from sklearn.model_selection import ParameterGrid, GridSearchCV

from gym_mujoco_planar_snake.benchmark.info_collector import InfoCollector, InfoDictCollector

import numpy as np

import os
import os.path as osp

from gym_mujoco_planar_snake.common.model_saver_wrapper import ModelSaverWrapper
import gym_mujoco_planar_snake.benchmark.plots as import_plots
#from gym_mujoco_planar_snake.common.my_observation_wrapper import MyObservationWrapper

import math
from pprint import pprint

"""
 In the gait equation, the frequency is for changing speed, the amplitude is to change the "S" shape, 
 the amplitude bias is used to steer the snake, and the phase is to set up the numbers of cycles. 
 You can find more details in "Parameterized and Scripted Gaits for Modular Snake Robots".
"""

# also for benchmark
# run untill done
#def run_environment_episode(env, max_timesteps, render, lambda_deg=120, alpha_deg=100, w_para=2.0, y_para=0.0):
#def run_environment_episode(env, max_timesteps, render, lambda_deg=80, alpha_deg=80, w_para=1.5, y_para=0.1):
#def run_environment_episode(env, max_timesteps, render, lambda_deg=140, alpha_deg=40, w_para=1.5, y_para=0.2):
# best
#def run_environment_episode(env, max_timesteps, render, lambda_deg=100, alpha_deg=80, w_para=1.0, y_para=0.2):
#def run_environment_episode(env, max_timesteps, render, lambda_deg=140, alpha_deg=40, w_para=1.9, y_para=0.2):
#def run_environment_episode(env, max_timesteps, render, lambda_deg=120, alpha_deg=55, w_para=0.5, y_para=0.1):
#def run_environment_episode(env, max_timesteps, render, lambda_deg=120, alpha_deg=55, w_para=0.5, y_para=0.2):
def run_environment_episode(env, max_timesteps, render, lambda_deg=110, alpha_deg=80, w_para=0.75, y_para=0.2):
#def run_environment_episode(env, max_timesteps, render, lambda_deg=120, alpha_deg=90, w_para=2.0, y_para=0.2):
#def run_environment_episode(env, max_timesteps, render, lambda_deg=120, alpha_deg=90, w_para=0.5, y_para=0.1):
    number_of_timestep = 0
    done = False

    # set seed
    seed=1
    set_global_seeds(seed)
    env.seed(seed)

    # pos
    # orientation =


    # Number of Joints
    K = 8

    # Car Joint Handles
    #carJoints = {}
    #for i in range(K):
    #    carJoints[i] = sim.getObjectHandle('Car_joint_',(i))

    # Time since simulation Start
    t = 0

    # Lenght of one Snake - Car - Module
    m = 0.35

    # linear reduction parameters(set y = 0 and z = 1 for disabling)
    y = y_para # 0.5, 0.4
    z = 1 - y

    # Motion Parameters
    # lam = lambda (number of waves, 80=2, 60 =1, 120=3, 140=4)
    lam = lambda_deg * math.pi / 180 # 60 , 90
    # Turning Radius (how much bending)
    a = alpha_deg * math.pi / 180 # 60, 80



    # angular frequency
    w = w_para * math.pi # 1.5

    # for the start
    p = -1 # -0.3

    # Calculate the mean effective lenght of the snake
    #Direction of the n-th Module relative to the head
    #theta = {0}
    theta = [0] * (K+1)

    #Mean Direction of the Snake
    snakeDir = 0 # TODO to ball

    for i in range(K):
        amp = a * ((i - 1) * y / K + z)
        theta[i + 1] = theta[i] + amp * math.cos( lam * (i - 1))
        snakeDir = snakeDir + theta[i + 1]


    snakeDir = snakeDir / (K + 1)

    # Mean effective lenght of the snake
    l = 0
    for i in range(K+1):
        l = l + m * math.cos(theta[i] - snakeDir)


    # turn radius TODO
    # set
    b = 0


    obs = env.reset()

    # info_collector = InfoCollector(env, {'lambda_deg':lambda_deg, 'alpha_deg':alpha_deg, 'w_para':w_para, 'y_para':y_para})
    info_collector = InfoCollector(env, {'env':env, 'seed':seed})

    while (not done) and number_of_timestep < max_timesteps:

        #t_delta = env.unwrapped.model.opt.timestep * env.unwrapped.frame_skip #0.05
        t_delta = env.unwrapped.dt
        #print(t_delta, env.unwrapped.model.opt.timestep , env.unwrapped.frame_skip)

        t = t + t_delta

        # direction
        angle_difference = env.unwrapped.get_head_to_target_degree_angle()
        b = -angle_difference / 50 # seems working ok, 100

        # speed
        dis_difference = env.unwrapped.calc_distance() - 4
        w = w + dis_difference * 0.025*math.pi  # 0.01 works ok
        #print("-----Distance Difference------: ", dis_difference)
        #print("------------- w --------------: ", w)

        action = np.zeros(8)#env.action_space.sample() # setJointTargetPosition

        # Calculate and set the Target angle phi for each carJoint except the first
        # without head
        #for i in range(1,K):
        #    amp = a * ((i - 1) * y / K + z)
        #    phi = b + amp * math.cos(w * t - lam * (i - 1))
        #    #sim.setJointTargetPosition(carJoints[i], -phi * (1 - math.exp(p * t)))
        #    action[i] = -phi * (1 - math.exp(p * t))

        # with head
        for i in range(1,K+1):
            # print("-----i is: ", i)
            amp = a * ((i) * y / K + z)

            #amp = np.pi/2 # aka alpha
            #lam = 2*np.pi/(K+1) # aka beta

            phi = b*1 + amp * math.cos(w * t - lam * (i))
            #sim.setJointTargetPosition(carJoints[i], -phi * (1 - math.exp(p * t)))
            action[i-1] = -phi * (1 - 1*math.exp(p * t))


        # Head Orientation Compensation
        """
        theta = 0
        snakeDir = 0
        JointPosition = obs[0:8]

        # print(angle_difference)
        snakeDir = -angle_difference/6 # TODO

        for i in range(1,K):
            #phi = sim.getJointPosition(carJoints[i])
            phi = JointPosition[i]
            theta = theta + phi
            snakeDir = snakeDir + theta

        snakeDir = snakeDir / (K+1)

        #sim.setJointTargetPosition(carJoints[1], -snakeDir * (1 - math.exp(p * t)))
        action[0] = -snakeDir * (1 - math.exp(p * t))
        #"""

        obs, reward, done, info = env.step(action)
        #print(info["joint_powers"])

        info['seed'] = seed
        info['env'] = env.spec.id
        info_collector.add_info(info)


        #render
        if render:
            env.render()

        number_of_timestep += 1

    return done, number_of_timestep, info_collector



def evaluate_power_velocity(env_id):
    env = gym.make(env_id)

    # more steps
    # env._max_episode_steps = env.spec.max_episode_steps * 2
    # obs = env.reset()

    print("actionspace", env.action_space)
    print("observationspace", env.observation_space)

    gym.logger.setLevel(logging.WARN)

    #lambda_deg = [60, 80, 100, 120, 140]
    #lambda_deg = [42]
    #alpha_deg = [70, 80, 90, 100]
    #alpha_deg = [42]
    #w_para = [0.5, 0.75, 1.0, 1.5, 2.0]
    #y_para = [0.1]

    # final
    lambda_deg = np.arange(40, 190, 10)  # 15
    alpha_deg = np.arange(40, 130, 10)  # 9
    w_para = np.arange(0.25, 3.25, 0.25)  # 12
    y_para = np.arange(0.1, 0.5, 0.1)  # 4

    # lambda_deg = np.arange(40, 190, 40)  # 15
    # alpha_deg = np.arange(40, 130, 40)  # 9
    # w_para = np.arange(0.25, 3.25, 1)  # 12
    # y_para = np.arange(0.1, 0.5, 0.2)  # 4


    grid = ParameterGrid(param_grid={'lambda_deg': lambda_deg, 'alpha_deg': alpha_deg, 'w_para': w_para, 'y_para':y_para})
    paras = list(grid)

    info_dict_collector = InfoDictCollector(env)



    for i, para in enumerate(paras):
        # run one episode
        #print("----------Episode " + "%s"%i + "----------")
        render = True

        lambda_deg = para['lambda_deg']
        alpha_deg = para['alpha_deg']
        w_para = para['w_para']
        y_para = para['y_para']

        done, number_of_timesteps, info_collector = \
            run_environment_episode(env, env._max_episode_steps, render, lambda_deg, alpha_deg, w_para, y_para)

        mean_dict = info_collector.get_mean_dict_200(['velocity'])
        # mean_dict_joints_power = info_collector.get_mean_dict_200_joint_power(['joint_powers'])
        # print(mean_dict_joints_power)

        print('run {}/{} para: {}, velocity: {}'.format(i,len(paras),para, mean_dict['velocity']))

        info_dict_collector.add_info_collector2(info_collector)



    info_dict_collector.print_rank2()



def evaluate_target_tracking(env_id, render_flag):

    seed = [1]
    # envs
    eval_env_id = ['Mujoco-planar-snake-cars-cam-dist-line-v1',
                   'Mujoco-planar-snake-cars-cam-dist-wave-v1',
                   'Mujoco-planar-snake-cars-cam-dist-zigzag-v1',
                   'Mujoco-planar-snake-cars-cam-dist-random-v1',
                   ]

    grid = ParameterGrid(param_grid={'eval_env_id': eval_env_id, 'seed': seed})
    paras = list(grid)

    info_dict_collector = InfoDictCollector(None)

    render = render_flag
    print("===============================================================", render_flag)

    with tf.device('/cpu'):

        for i, para in enumerate(paras):
            eval_env_id = para['eval_env_id']
            seed = int(para['seed'])

            env = gym.make(eval_env_id)
            env._max_episode_steps = env.spec.max_episode_steps * 3


            # lambda_deg = 110
            # alpha_deg = 50
            # w_para = 0.25*np.pi
            # y_para = 0.3

            lambda_deg = 110
            alpha_deg = 60
            w_para = 2
            y_para = 0.5*0     

               

            done, number_of_timesteps, info_collector = \
                    run_environment_episode(env, env._max_episode_steps, render, lambda_deg, alpha_deg, w_para, y_para)

            info_dict_collector.add_info_collector(info_collector)

            env.close()

    modelversion = 666
    info_dict_collector.following_eval_save(modelversion)

    # print(info_collector.sensor_head_velocity)
    # pprint(dir(info_collector))
    # print(info_collector.dict_list_infos['head_y'])
    # info_dict_collector.following_eval_save()

    import_plots.evaluate_target_tracking()

def enjoy(env_id):

    env = gym.make(env_id)

    # more steps
    #env._max_episode_steps = env.spec.max_episode_steps * 2

    #obs = env.reset()

    print("actionspace", env.action_space)
    print("observationspace", env.observation_space)

    gym.logger.setLevel(logging.WARN)

    sum_info = ''

    while True:
        # run one episode
        render = True #True # TODO

        done, number_of_timesteps, info_collector = run_environment_episode(env, env._max_episode_steps, render)
        #done, number_of_timesteps, list_infos, dict_list_infos = run_environment_episode_hirose(env, env._max_episode_steps, render)

        print('timesteps: %d' % (number_of_timesteps))

        info_collector.episode_info_print()
        #print("INFO", info_collector)



def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument('--num-timesteps', type=int, default=int(2e6))#1e6
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))  # 1e6

    # grid
    parser.add_argument('--evaluate_power_velocity', type=bool, default=False)  # 1e6

    parser.add_argument('--evaluate_target_tracking', type=bool, default=False)  # 1e6

    # env
    #parser.add_argument('--env', help='environment ID', default='Mujoco-planar-snake-cars-angle-v1')
    parser.add_argument('--env', help='environment ID', default='Mujoco-planar-snake-cars-angle-line-v1')
    #parser.add_argument('--env', help='environment ID', default='Mujoco-planar-snake-cars-angle-zigzag-v1')
    #parser.add_argument('--env', help='environment ID', default='Mujoco-planar-snake-cars-angle-circle-v1')
    #parser.add_argument('--env', help='environment ID', default='Mujoco-planar-snake-cars-angle-random-v1')

    # render
    parser.add_argument('--render', help='render simulation', type=bool, default=False)

    args = parser.parse_args()
    logger.configure()

    # TODO CUDA off -> CPU only!
    os.environ['CUDA_VISIBLE_DEVICES'] = ''


    # Evaluate power velocity
    if args.evaluate_power_velocity:
        print("----------First----------")
        evaluate_power_velocity(args.env)

    elif args.evaluate_target_tracking:
        print("----------Second----------")
        print(args.render)
        evaluate_target_tracking(args.env, args.render)
    
    # Enjoy this controller    
    else:
        print("----------Third----------")
        evaluate_target_tracking(args.env)
        # enjoy(args.env)
        

if __name__ == '__main__':
    main()
