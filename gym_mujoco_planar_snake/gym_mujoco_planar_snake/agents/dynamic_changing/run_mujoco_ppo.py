#!/usr/bin/env python
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import gym, logging
from baselines import logger
import tensorflow as tf
import numpy as np

from sklearn.model_selection import ParameterGrid

import os
import os.path as osp

from gym_mujoco_planar_snake.common.model_saver_wrapper import ModelSaverWrapper
from gym_mujoco_planar_snake.common import my_tf_util
from gym_mujoco_planar_snake.benchmark.info_collector import InfoCollector, InfoDictCollector

import  gym_mujoco_planar_snake.benchmark.plots as import_plots

def prepare_env(env_id, seed, num_cpu):
    sess = U.make_session(num_cpu=num_cpu)
    sess.__enter__()
    set_global_seeds(seed)
    env = gym.make(env_id)
    return env

def get_latest_model_file(model_dir):
    return get_model_files(model_dir)[0]

def get_model_files(model_dir):
    list = [x[:-len(".index")] for x in os.listdir(model_dir) if x.endswith(".index")]
    list.sort(key=str.lower, reverse=True)

    files = [osp.join(model_dir, ele) for ele in list]
    return files

def get_model_dir(env_id, name):
    model_dir = osp.join(logger.get_dir(), '../../models')
    model_dir = ModelSaverWrapper.gen_model_dir_path(model_dir, env_id, name)
    logger.log("model_dir: %s" % model_dir)
    return model_dir

def policy_fn(name, ob_space, ac_space):
    from baselines.ppo1 import mlp_policy
    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                hid_size=64, num_hid_layers=2)

    #from baselines.ppo1 import cnn_policy
    #return cnn_policy.CnnPolicy(name=name, ob_space=ob_space, ac_space=ac_space)


# also for benchmark
# run untill done
def run_environment_episode(env, pi, seed, model_file, max_timesteps, render, stochastic=True):
    number_of_timestep = 0
    done = False

    # load model
    my_tf_util.load_state(model_file)

    # set seed
    set_global_seeds(seed)
    env.seed(seed)

    obs = env.reset()

    # TODO!!!
    #obs[-1] = target_v

    #info_collector = InfoCollector(env, {'target_v': target_v})
    info_collector = InfoCollector(env, {'env':env, 'seed':seed})

    # injured_joint_pos = [None, 7, 5, 3, 1]
    # injured_joint_pos = [None, 7, 6, 5, 4, 3, 2, 1, 0]

    #######################################################
    friction_index = {4,5,8,9,12,13,16,17,20,21,24,25,28,29,32,33,36,37}
    mu_friction, sigma_friction = 0, 0.1
    wheel_friction = np.random.normal(mu_friction, sigma_friction, 1)
    wheel_friction = np.append(wheel_friction, [0, 0])
    wheel_friction = wheel_friction + [0.8, 0, 1e-4]



    for i in friction_index:
        env.unwrapped.sim.model.geom_friction[i] = wheel_friction
        # print(env.unwrapped.sim.model.geom_friction[i])
        # pass
    # print((env.unwrapped.sim.model.geom_friction))
    mu, sigma = 0, 0.2
    s = np.random.normal(mu, sigma, 3)
    s = np.append(s,0)
    s = s + [0, 0, 0, 1]
    # print(s)
    # env.unwrapped.sim.model.geom_rgba[1] = s
    # print("############## color ", env.unwrapped.sim.model.geom_rgba[1], "#################")
    #######################################################

    while (not done) and number_of_timestep < max_timesteps:

        # TODO!!!
        #obs[-1] = target_v

        action = pi.act(stochastic, obs)

        action = action[0]  # TODO check

        """
        if number_of_timestep % int(max_timesteps / len(injured_joint_pos)) == 0:
            index = int(number_of_timestep / int(max_timesteps / (len(injured_joint_pos))))
            print("index 1: ", index)
            index = min(index, len(injured_joint_pos)-1)
            print("index 2", index)

            print("number_of_timestep", number_of_timestep, index, max_timesteps)
            env.unwrapped.metadata['injured_joint'] = injured_joint_pos[index]
        #"""

        obs, reward, done, info = env.step(action)

        info['seed'] = seed
        info['env'] = env.spec.id

        # add info
        info_collector.add_info(info)

        #render
        if render:
            env.render()

        number_of_timestep += 1

    return done, number_of_timestep, info_collector



def evaluate_power_velocity(env_id):

    #target_v = np.arange(0.05, 0.8, 0.05) long
    #target_v = np.arange(0.025, 0.275, 0.025)
    #target_v = np.arange(0.025, 0.175, 0.025)
    #target_v = np.arange(0.025, 0.175, 0.005)
    #target_v = np.arange(0.025, 0.225, 0.005)
    #target_v = np.arange(0.025, 0.305, 0.005)
    target_v = np.arange(0.025, 0.255, 0.005)

    seed = [1]
    #seed = np.arange(1, 4, 1)

    grid = ParameterGrid(param_grid={'target_v': target_v, 'seed': seed})
    paras = list(grid)

    render = False

    with tf.device('/cpu'):
        sess = U.make_session(num_cpu=1)
        sess.__enter__()
        env = gym.make(env_id)
        gym.logger.setLevel(logging.WARN)

        info_dict_collector = InfoDictCollector(env)

        # init load
        model_dir = get_model_dir(env_id, 'ppo')
        model_files = get_model_files(model_dir)
        model_file = model_files[0]
        # model_file = model_files[75]
        logger.log("load model_file: %s" % model_file)
        pi = policy_fn('pi', env.observation_space, env.action_space)


        for i, para in enumerate(paras):
            # run one episode

            target_v = para['target_v']
            seed = int(para['seed'])

            env.unwrapped.metadata['target_v'] = target_v

            done, number_of_timesteps, info_collector = \
                run_environment_episode(env, pi, seed, model_file, env._max_episode_steps, render, stochastic=True)

            print('run {}/{} para: {}, timesteps: {}'.format(i,len(paras),para, number_of_timesteps))

            info_dict_collector.add_info_collector2(info_collector)

    info_dict_collector.print_rank2()



def evaluate_target_tracking(env_id):

    # there is somehow a bug somewhere. somehow always one run fails... therefore run everything twice
    seed = [1,2]
    max_timesteps = 3000000

    # model select
    #
    #modelverion_in_k_ts = 2000
    modelverion_in_k_ts = 3000 # good
    modelverion_in_k_ts = 2510  # better
    model_index = int(max_timesteps/1000/10 - modelverion_in_k_ts /10)

    #TOdo last saved model
    model_index = 0 # uncomment fo select model by modelverion_in_k_ts


    # envs
    eval_env_id = ['Mujoco-planar-snake-cars-cam-dist-line-v1',
                   'Mujoco-planar-snake-cars-cam-dist-wave-v1',
                   'Mujoco-planar-snake-cars-cam-dist-zigzag-v1',
                   'Mujoco-planar-snake-cars-cam-dist-random-v1',
                   ]

    grid = ParameterGrid(param_grid={'eval_env_id': eval_env_id, 'seed': seed})
    paras = list(grid)

    render = False


    info_dict_collector = InfoDictCollector(None)

    # init load
    model_dir = get_model_dir(env_id, 'ppo')
    model_files = get_model_files(model_dir)
    model_file = model_files[model_index]
    # model_file = model_files[75]
    logger.log("load model_file: %s" % model_file)

    sess = U.make_session(num_cpu=1)
    sess.__enter__()
    gym.logger.setLevel(logging.WARN)

    env = gym.make(env_id)
    pi = policy_fn('pi', env.observation_space, env.action_space)
    env.close()

    with tf.device('/cpu'):

        for i, para in enumerate(paras):
            eval_env_id = para['eval_env_id']
            seed = int(para['seed'])

            env = gym.make(eval_env_id)

            # 3000 timesteps, default for evaluation
            env._max_episode_steps = env._max_episode_steps * 3


            done, number_of_timesteps, info_collector = \
                run_environment_episode(env, pi, seed, model_file, env._max_episode_steps, render, stochastic=False)

            print('run {}/{} para: {}, timesteps: {}'.format(i, len(paras), para, number_of_timesteps))

            info_dict_collector.add_info_collector(info_collector)

            env.close()


    modelversion = modelverion_in_k_ts
    info_dict_collector.following_eval_save(modelversion)

    # plot
    import_plots.evaluate_target_tracking()




def enjoy(env_id, seed):

    with tf.device('/cpu'):
        sess = U.make_session(num_cpu=1)
        sess.__enter__()

        env = gym.make(env_id)
        #env = gym.make('Mujoco-planar-snake-cars-cam-v1')
        #env = gym.make('Mujoco-planar-snake-cars-cam-dist-zigzag-v1')
        #env = gym.make('Mujoco-planar-snake-cars-cam-dist-random-v1')
        #env = gym.make('Mujoco-planar-snake-cars-cam-dist-line-v1')
        #env = gym.make('Mujoco-planar-snake-cars-cam-dist-circle-v1')
        #env = gym.make('Mujoco-planar-snake-cars-cam-dist-wave-v1')

        check_for_new_models = True


        # more steps
        #env._max_episode_steps = env.spec.max_episode_steps * 3
        #obs = env.reset()


        max_timesteps = 3000000

        # model_index = 254 # 251 # best

        # Select model file .....

        #check_for_new_models = False
        #
        # modelverion_in_k_ts = 2000
        modelverion_in_k_ts = 3000  # good
        modelverion_in_k_ts = 2510  # better

        model_index = int(max_timesteps / 1000 / 10 - modelverion_in_k_ts / 10)

        # TOdo last saved model
        model_index = 0



        print("actionspace", env.action_space)
        print("observationspace", env.observation_space)

        gym.logger.setLevel(logging.WARN)

        # init load
        model_dir = get_model_dir(env_id, 'ppo')
        model_files = get_model_files(model_dir)
        #model_file = get_latest_model_file(model_dir)
        print('available models: ', len(model_files))
        model_file = model_files[model_index]
        #model_file = model_files[75]
        logger.log("load model_file: %s" % model_file)

        sum_info = None
        pi = policy_fn('pi', env.observation_space, env.action_space)


        while True:
            # run one episode

            # TODO specify target velocity
            # only takes effect in angle envs
            #env.unwrapped.metadata['target_v'] = 0.05
            env.unwrapped.metadata['target_v'] = 0.15
            #env.unwrapped.metadata['target_v'] = 0.25

            #env._max_episode_steps = env._max_episode_steps * 3

            done, number_of_timesteps, info_collector = run_environment_episode(env, pi, seed, model_file, env._max_episode_steps, render=True, stochastic=False)



            info_collector.episode_info_print()

            check_model_file = get_latest_model_file(model_dir)
            if check_model_file != model_file and check_for_new_models:
                model_file = check_model_file
                logger.log('loading new model_file %s' % model_file)

            print('timesteps: %d, info: %s' % (number_of_timesteps, str(sum_info)))

            """
            # print head cam
            if number_of_timestep % 100 == 0:
                img = obs[-1 * 16:] * 1
                # img = img.reshape(16, 5, 1)[::-1, :, :]
                img = img.reshape(1, 16, 1)
                # img = img[::-1, :, :]# original image is upside-down, so flip it
                img = img[:, :, 0]

                plt.imshow(img, cmap='gray', animated=True, label="aaa" + str(number_of_timestep))
                plt.title('step: ' + str(number_of_timestep))
                plt.colorbar()
                plt.savefig('aa.png')
                plt.clf()
            """


def generate_expert_traj(env_id, seed):

    target_v = np.arange(0.025, 0.255, 0.005)

    seed = [1]

    grid = ParameterGrid(param_grid={'target_v': target_v, 'seed': seed})
    paras = list(grid)

    render = False

    with tf.device('/cpu'):
        sess = U.make_session(num_cpu=1)
        sess.__enter__()

        env = gym.make(env_id)

        model_index = 0

        gym.logger.setLevel(logging.WARN)
        model_dir = get_model_dir(env_id, 'ppo')
        model_files = get_model_files(model_dir)
        model_file = model_files[model_index]
        logger.log("load model_file: %s" % model_file)

        pi = policy_fn('pi', env.observation_space, env.action_space)

        for i, para in enumerate(paras):
            # run one episode
            target_v = para['target_v']
            seed = int(para['seed'])

            env.unwrapped.metadata['target_v'] = target_v

            done, number_of_timesteps, info_collector = \
                run_environment_episode(env, pi, seed, model_file, env._max_episode_steps, render, stochastic=True)


def train_ppo1(env_id, num_timesteps, sfs, seed):
    from baselines.ppo1 import pposgd_simple
    # config = tf.ConfigProto()
    # config.intra_op_parallelism_threads = 50
    # config.inter_op_parallelism_threads = 50
    sess = U.make_session(num_cpu=1)
    sess.__enter__()
    set_global_seeds(seed)
    env = gym.make(env_id)

    model_dir = get_model_dir(env_id, 'ppo')



    # monitor tensorboard
    log_dir = osp.join(logger.get_dir(), 'log_ppo')
    logger.log("log_dir: %s" % log_dir)
    env = bench.Monitor(env, log_dir)

    env = ModelSaverWrapper(env, model_dir, sfs)

    env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    pposgd_simple.learn(env, policy_fn,
                        max_timesteps=num_timesteps,
                        timesteps_per_actorbatch=2048,
                        clip_param=0.2, # TODO 0.2
                        entcoeff=0.0,
                        optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
                        gamma=0.99, lam=0.95,
                        schedule='linear', # TODO linear
                        )
    env.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument('--seed', help='RNG seed', type=int, default=1)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)

    parser.add_argument('--num-timesteps', type=int, default=int(1e6))  # 1e6

    #parser.add_argument('--train', help='do training or load model', type=bool, default=True)
    parser.add_argument('--train', help='do training or load model', type=bool, default=False)


    # env
    # TODO choose environment

    # target fixed (deprecated)
    #parser.add_argument('--env', help='environment ID', default='Mujoco-planar-snake-cars-cam-v1')
    #parser.add_argument('--env', help='environment ID', default='Mujoco-planar-snake-cars-cam-line-v1')
    #parser.add_argument('--env', help='environment ID', default='Mujoco-planar-snake-cars-cam-zigzag-v1')
    #parser.add_argument('--env', help='environment ID', default='Mujoco-planar-snake-cars-cam-line-v1')
    #parser.add_argument('--env', help='environment ID', default='Mujoco-planar-snake-cars-cam-wave-v1')

    #velocity - power test
    parser.add_argument('--env', help='environment ID', default='Mujoco-planar-snake-cars-angle-line-v1')

    # target tracking
    #parser.add_argument('--env', help='environment ID', default='Mujoco-planar-snake-cars-cam-dist-wave-v1')
    #parser.add_argument('--env', help='environment ID', default='Mujoco-planar-snake-cars-cam-dist-line-v1')
    #parser.add_argument('--env', help='environment ID', default='Mujoco-planar-snake-cars-cam-dist-zigzag-v1')
    #parser.add_argument('--env', help='environment ID', default='Mujoco-planar-snake-cars-cam-dist-random-v1')



    #save_frequency_steps
    parser.add_argument('--sfs', help='save_frequency_steps', default=10000) # for mujoco

    # grid search
    parser.add_argument('--evaluate_power_velocity', help='evaluate_power_velocity by grid search', type=bool, default=False)


    # run_environment_episode
    parser.add_argument('--evaluate_target_tracking', help='evaluate_target_tracking by grid search', type=bool, default=False)


    args = parser.parse_args()
    logger.configure()

    if args.evaluate_power_velocity:
        evaluate_power_velocity(args.env)

    elif args.evaluate_target_tracking:
        evaluate_target_tracking(args.env)

    elif args.train:
        # CUDA off -> CPU only!
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

        train_ppo1(args.env, num_timesteps=args.num_timesteps, sfs=args.sfs, seed=args.seed)

    else:
        # CUDA off -> CPU only!
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

        enjoy(args.env, seed=args.seed)

if __name__ == '__main__':
    main()
