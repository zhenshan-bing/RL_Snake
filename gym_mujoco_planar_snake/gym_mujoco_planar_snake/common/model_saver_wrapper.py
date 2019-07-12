from gym.core import ObservationWrapper
#from baselines.common import tf_util as U
from gym_mujoco_planar_snake.common import my_tf_util as U
from baselines import logger
import os.path as osp
import time

class ModelSaverWrapper(ObservationWrapper):

    def __init__(self, env, model_dir, save_frequency_steps):
        ObservationWrapper.__init__(self, env=env)

        self.save_frequency_steps = save_frequency_steps
        self.total_steps = 0
        self.total_steps_save_counter = 0
        self.total_episodes = 0
        self.str_time_start = time.strftime("%Y%m%d-%H%M")

        self.model_dir = model_dir

    def gen_model_dir_path(folder_path, env_id, algorithm_name):
        dir = folder_path
        if osp.isdir(dir):
            dir = osp.join(dir, '%s' % env_id)
            dir = osp.join(dir, '%s' % algorithm_name)
        else:
            raise RuntimeError("Not a folder")

        return dir

    def reset(self, **kwargs):

        self.total_episodes += 1

        if self.total_steps_save_counter >= self.save_frequency_steps or self.total_steps == 1:
            # save
            file_name = osp.join(self.model_dir, '%s-%0*d' % (self.str_time_start, 9, self.total_steps))
            U.save_state(file_name)
            logger.log('Saved model to: ' + file_name)

            self.total_steps_save_counter = 0

        return self.env.reset(**kwargs)

    def step(self, action):
        

        self.total_steps += 1
        self.total_steps_save_counter += 1
        return self.env.step(action)

    def observation(self, observation):
        return observation