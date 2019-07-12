
import numpy as np
import pandas as pd
from gym_mujoco_planar_snake.benchmark import plots
from time import gmtime, strftime
import itertools
import os

from baselines import logger
import os.path as osp

class InfoCollector():

    def __init__(self, env, paras):
        self.dict_list_infos = None
        self.env = env
        self.paras = paras

    def add_info(self, info):
        # dict_list_infos
        if self.dict_list_infos is None:
            self.dict_list_infos = dict.fromkeys(info, 42)
            # reference problem
            for key in self.dict_list_infos.keys():
                self.dict_list_infos[key] = []

        # add values
        [self.dict_list_infos[k].append(v) for k, v in info.items()]

    def get_list_dict(self,columns):
        return {x:list(self.dict_list_infos[x]) for x in columns}

    def get_mean_dict(self, columns):
        menas_dict = {x:float(np.mean(self.dict_list_infos[x])) for x in columns}
        return menas_dict

    def get_sum_dict(self, columns):
        sum_dict = {x:float(np.sum(self.dict_list_infos[x])) for x in columns}
        return sum_dict

    def get_first_dict(self, columns):
        first_dict = {x:str(self.dict_list_infos[x][0]) for x in columns}
        return first_dict

    def get_mean_dict_200(self, columns):
        menas_dict = {x:float(np.mean(self.dict_list_infos[x][200:])) for x in columns}
        return menas_dict

    # Write by Zhenshan Bing
    def get_mean_dict_200_joint_power(self, columns):
        # print(np.mean(self.dict_list_infos['joint_powers'][200:], axis=0))
        temp = np.mean(self.dict_list_infos['joint_powers'][200:], axis=0)
        # menas_dict = {'joint_powers': temp}
        # menas_dict = {'joint_powers':float(np.mean(self.dict_list_infos['joint_powers'][200:], axis=0))}
        return temp

    def get_sum_dict_200(self, columns):
        sum_dict = {x:float(np.sum(self.dict_list_infos[x][200:])) for x in columns}
        return sum_dict

    def episode_info_print(self):
        paras = ['reward', 'velocity', 'power', 'power_normalized', 'total_power_sec', 'mean_actuatorfrcs', 'abs_joint_velocities', 'energy0',
                 'sensor_head_velocity', 'target_v']
        means_v = list([float(np.mean(self.dict_list_infos[x])) for x in paras])

        cum = list([float(np.sum(self.dict_list_infos[x])) for x in ['power', 'distance_delta']])

        max = list([float(np.max(np.abs(self.dict_list_infos[x]))) for x in ['max_joint_velocities']])


        print(
            'mean reward {d[0]:06.4f}, '
            'mean velocity {d[1]:06.4f}, '
            'mean power {d[2]:06.4f}, '
            'energy normalized {d[3]:06.4f}, '
            'mean actuatorfrcs {d[4]:06.4f}, '
            'mean abs_joint_velocities {d[5]:06.4f},'
            'mean energy0 {d[6]:06.4f}, '
            'mean sensor_head_velocity {d[7]:06.4f}, '
            'mean_target_v {d[8]:06.4f}, '
            'sum power {d2[0]:06.4f}, '
            'sum distance_delta {d2[1]:06.4f} '
            'max_joint_velocities {d3[0]:06.4f} '.format(d=means_v, d2=cum, d3=max))

        return means_v

    def save_csv(self, dir, fname= None):
        df = pd.DataFrame.from_dict(self.dict_list_infos)

        print("-------HERE------") 

        if fname is None:
            fname = 'data_run_env_{}_seed_{}'.format(self.paras['env'].spec.id, self.paras['seed'])

        df.to_csv('{}/{}.csv'.format(dir, fname))

    def plot(self, dir=None):
        my_plots = plots.Plots(self.env, dir)
        # todo

        my_plots.plot_head_trace(self.get_list_dict(['head_x', 'head_y']))
        #my_plots.plot_snake_gait(self.get_list_dict(['head_x', 'head_y', 'joints_pos', 'joint_head_pos']))

class InfoDictCollector():
    def __init__(self, env):
        self.info_collector_set = set()
        self.env = env

        self.all_mean_dict = None


    def add_info_collector(self, info_collector):
        self.info_collector_set.add(info_collector)



    def add_info_collector2(self, info_collector):
        self.info_collector_set.add(info_collector)

        columns = ['reward', 'velocity', 'power_normalized', 'target_v', 'total_power_sec']
        sum_columns = ['power', 'distance_delta']

        mean_dict = info_collector.get_mean_dict_200(columns)  # 200
        mean_dict.update(info_collector.paras)

        #   Add by Zhenshan Bing  #
        mean_dict['joint_powers'] = info_collector.get_mean_dict_200_joint_power(['joint_powers'])
        print(mean_dict['joint_powers'])
        # Finish by Zhenshan Bing #
        
        mean_dict['id'] = info_collector.env.unwrapped.spec.id

        mean_dict['v_e_ratio'] = mean_dict['velocity'] / mean_dict['power_normalized']

        sum_dict = info_collector.get_sum_dict_200(sum_columns)

        mean_dict.update(sum_dict)

        product = itertools.product([1.0], [0.8, 0.6, 0.5, 0.2])
        for (b, v_rate) in product:
            rew_v = 1 - np.abs(mean_dict['target_v'] - mean_dict['velocity']) ** (1 / b) / mean_dict[
                                                                                               'target_v'] ** (
                                                                                           1 / b)

            reward = v_rate * rew_v + (1.0 - v_rate) * (1.0 - mean_dict['power_normalized'])
            mean_dict['rew_b_{}_v_rate_{}'.format(b, v_rate)] = reward

        if self.all_mean_dict is None:
            mean_dict = {key: [value] for key, value in mean_dict.items()}
            self.all_mean_dict = mean_dict.copy()
        else:
            {self.all_mean_dict[key].append(value) for key, value in mean_dict.items()}


    # velocity and power
    def print_rank2(self):
        df = pd.DataFrame.from_dict(self.all_mean_dict)
        df = df.sort_values(['reward'], ascending=False)

        print('results')

        #self.dir = "/home/chris/openai_logdir/power_velocity/"  # TODO
        logger.configure()
        self.dir = osp.join(logger.get_dir(), '../../power_velocity')

        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        fname = 'grid_search_results_{}'.format(strftime("%Y-%m-%d_%H:%M:%S", gmtime()))

        df.to_csv('{}/{}.csv'.format(self.dir, fname))
        print(df.to_string())

        # plot
        my_plots = plots.Plots(self.env, self.dir)
        my_plots.plot_velocity_power_diagram(df, fname)
        my_plots.plot_energy_distance_diagram(df, fname)






    def following_eval_save(self, modelversion = None):

        #self.dir = "/home/chris/openai_logdir"  # TODO
        logger.configure()
        self.dir = osp.join(logger.get_dir(), '../../')

        if modelversion is None:
            dname = 'following_eval_results_{}'.format(strftime("%Y-%m-%d_%H:%M:%S", gmtime()))
        else:
            dname = 'following_eval_results_{}_{}'.format(strftime("%Y-%m-%d_%H:%M:%S", gmtime()), modelversion)
        newdir = self.dir + '/' + dname

        if not os.path.exists(newdir):
            os.makedirs(newdir)

        columns = ['reward', 'velocity', 'angle_difference_normalized', 'head_target_distance']
        sum_columns = ['distance_delta']
        #first_columns = ['injured_joint', 'seed']
        first_columns = ['seed', 'env']


        all_mean_dict = None

        for info_colletor in self.info_collector_set:
             
            info_colletor.save_csv(newdir)

            mean_dict = info_colletor.get_mean_dict(columns)
            mean_dict.update(info_colletor.paras)
            #mean_dict['id'] = info_colletor.env.unwrapped.spec.id

            #mean_dict['injured_joint'] = '{}'.format(info_colletor)
            sum_dict = info_colletor.get_sum_dict(sum_columns)
            fist_dict = info_colletor.get_first_dict(first_columns)
            mean_dict.update(sum_dict)
            mean_dict.update(fist_dict)


            if all_mean_dict is None:
                mean_dict = {key: [value] for key, value in mean_dict.items()}
                all_mean_dict = mean_dict.copy()
            else:
                {all_mean_dict[key].append(value) for key, value in mean_dict.items()}



        df = pd.DataFrame.from_dict(all_mean_dict)
        #df = df.sort_values(['reward'], ascending=False)
        csv_file = '{}/{}.csv'.format(newdir, 'data')
        df.to_csv(csv_file)
        print('save file {}'.format(csv_file))
        print(df.to_string())

        #return newdir


        print('results')

        # plot
        #my_plots = plots.Plots(self.env, dir = newdir)
        #my_plots.plot_distance_bar_diagram(df)
        #my_plots.plot_energy_distance_diagram(df, fname)


        # plot head tracks
        #bla = 0
        #for info_colletor in self.info_collector_set:
            # plot head track
        #    info_colletor.plot(dir=newdir + "/{}".format(bla))
        #    bla += 1
            # my_plots.plot_head_trace(info_colletor[['head_x', 'head_y']])

