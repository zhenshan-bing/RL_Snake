#!/usr/bin/env python
from baselines import bench
import gym, logging
from baselines import logger

from collections import Counter
from sklearn.model_selection import ParameterGrid, GridSearchCV

from gym_mujoco_planar_snake.benchmark.info_collector import InfoCollector, InfoDictCollector

import numpy as np

import os
import os.path as osp

from gym_mujoco_planar_snake.common.model_saver_wrapper import ModelSaverWrapper
#from gym_mujoco_planar_snake.common.my_observation_wrapper import MyObservationWrapper

import math
import csv

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
#def run_environment_episode(env, max_timesteps, render, lambda_deg=110, alpha_deg=80, w_para=0.75, y_para=0.2):
#def run_environment_episode(env, max_timesteps, render, lambda_deg=120, alpha_deg=90, w_para=2.0, y_para=0.2):
def run_environment_episode(env, max_timesteps, render, lambda_deg=120, alpha_deg=90, w_para=0.5, y_para=0.1):
    number_of_timestep = 0
    done = False

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
    p = -0.5 # -0.3

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

    info_collector = InfoCollector(env, {'lambda_deg':lambda_deg, 'alpha_deg':alpha_deg, 'w_para':w_para, 'y_para':y_para})

    while (not done) and number_of_timestep < max_timesteps:

        #t_delta = env.unwrapped.model.opt.timestep * env.unwrapped.frame_skip #0.05
        t_delta = env.unwrapped.dt
        #print(t_delta, env.unwrapped.model.opt.timestep , env.unwrapped.frame_skip)

        t = t + t_delta

        # direction
        angle_difference = env.unwrapped.get_head_to_target_degree_angle()
        b = -angle_difference / 100 # seems working ok


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
            amp = a * ((i) * y / K + z)

            #amp = np.pi/2 # aka alpha
            #lam = 2*np.pi/(K+1) # aka beta

            phi = b + amp * math.cos(w * t - lam * (i))
            #sim.setJointTargetPosition(carJoints[i], -phi * (1 - math.exp(p * t)))
            action[i-1] = -phi * (1 - math.exp(p * t))


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
        #action[0] = -snakeDir * (1 - math.exp(p * t))
        """

        obs, reward, done, info = env.step(action)

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


    grid = ParameterGrid(param_grid={'lambda_deg': lambda_deg, 'alpha_deg': alpha_deg, 'w_para': w_para, 'y_para':y_para})
    paras = list(grid)

    info_dict_collector = InfoDictCollector(env)



    for i, para in enumerate(paras):
        # run one episode
        render = False

        lambda_deg = para['lambda_deg']
        alpha_deg = para['alpha_deg']
        w_para = para['w_para']
        y_para = para['y_para']

        done, number_of_timesteps, info_collector = \
            run_environment_episode(env, env._max_episode_steps, render, lambda_deg, alpha_deg, w_para, y_para)

        mean_dict = info_collector.get_mean_dict_200(['velocity'])

        print('run {}/{} para: {}, velocity: {}'.format(i,len(paras),para, mean_dict['velocity']))

        info_dict_collector.add_info_collector2(info_collector)



    info_dict_collector.print_rank2()



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




from bayes_opt import BayesianOptimization

def beyas_power_velocity(env_id):
    env = gym.make(env_id)
    print("actionspace", env.action_space)
    print("observationspace", env.observation_space)

    gym.logger.setLevel(logging.WARN)
    
    for i in range(0, 12):
        w_para = (i*0.25, (i+1)*0.25)
        # print(w_para)

        pbounds={'lambda_deg':(40,190),'alpha_deg':(40,130),'alpha_deg':(40,130),'y_para':(0.1,0.5),'w_para':w_para}

        def black_box(lambda_deg, alpha_deg, y_para, w_para):
            done, number_of_timesteps, info_collector = \
                run_environment_episode(env, env._max_episode_steps, False, lambda_deg, alpha_deg, w_para, y_para)

            mean_dict = info_collector.get_mean_dict_200(['total_power_sec','velocity'])
            print('e:',mean_dict['total_power_sec'] / 3.6 )

            # fields=[mean_dict['total_power_sec'] / 3.6, mean_dict['velocity']]

            file_exists = osp.isfile('./bayes_opt_data/bayes_opt_data_'+str(i)+'.csv')
            with open('./bayes_opt_data/bayes_opt_data_'+str(i)+'.csv', 'a') as f:
                headers = ['total_power_sec', 'velocity']
                writer = csv.DictWriter(f, delimiter=',', lineterminator='\n',fieldnames=headers)
                if not file_exists:
                    writer.writeheader()  # file doesn't exist yet, write a header

                writer.writerow({'total_power_sec': mean_dict['total_power_sec'] / 3.6, 'velocity': mean_dict['velocity']})

            return mean_dict['velocity']

        
        optimizer=BayesianOptimization(f=black_box,pbounds=pbounds,random_state=1)
        optimizer.maximize(init_points=10,n_iter=100)


def beyas_power_velocity_test(env_id):
    env = gym.make(env_id)
    print("actionspace", env.action_space)
    print("observationspace", env.observation_space)

    gym.logger.setLevel(logging.WARN)

    alpha_deg = 118.6
    lambda_deg = 99.13
    w_para = 0.9212
    y_para = 0.1134


    done, number_of_timesteps, info_collector = \
        run_environment_episode(env, env._max_episode_steps, False, lambda_deg, alpha_deg, w_para, y_para)

    mean_dict = info_collector.get_mean_dict_200(['energy0','velocity'])
    print('Energy:',mean_dict['energy0'], 'velocity:',mean_dict['velocity'])

    return mean_dict['velocity']
    



def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument('--num-timesteps', type=int, default=int(2e6))#1e6
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))  # 1e6

    # grid
    parser.add_argument('--evaluate_power_velocity', type=bool, default=False)  # 1e6
    parser.add_argument('--bayes', type=bool, default=True)  # 1e6

    # env
    #parser.add_argument('--env', help='environment ID', default='Mujoco-planar-snake-cars-angle-v1')
    parser.add_argument('--env', help='environment ID', default='Mujoco-planar-snake-cars-angle-line-v1')
    #parser.add_argument('--env', help='environment ID', default='Mujoco-planar-snake-cars-angle-zigzag-v1')
    #parser.add_argument('--env', help='environment ID', default='Mujoco-planar-snake-cars-angle-circle-v1')
    #parser.add_argument('--env', help='environment ID', default='Mujoco-planar-snake-cars-angle-random-v1')


    args = parser.parse_args()
    logger.configure()

    # TODO CUDA off -> CPU only!
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    if args.evaluate_power_velocity:
        #if args.beyas:
            beyas_power_velocity(args.env)
            # beyas_power_velocity_test(args.env)
        #else:
            #evaluate_power_velocity(args.env)
    else:
        enjoy(args.env)

if __name__ == '__main__':
    main()
