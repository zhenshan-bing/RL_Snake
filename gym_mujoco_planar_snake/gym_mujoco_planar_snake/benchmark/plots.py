import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches
import os, fnmatch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from baselines import logger
import os
import os.path as osp

matplotlib.style.use('ggplot')

class Plots():

    def __init__(self, env, dir=None):
        if dir == None:
            #self.dir = "/home/chris/openai_logdir/" #TODO
            logger.configure()
            self.dir = osp.join(logger.get_dir(), '../../')
        else:
            self.dir = dir

        if env != None:
            self.env_name = env.unwrapped.spec.id
        else:
            self.env_name = 'test'

    def plot_head_trace(self, list_infos):

        data_x = list_infos['head_x']
        data_y = list_infos['head_y']

        plt.figure(figsize=(10, 4))

        plt.plot(data_x, data_y)
        #plt.show()
        plt.savefig(self.dir + self.env_name + '_plot_head_trace.pdf', dpi=600, bbox_inches='tight')
        plt.close('all')


    def plot_joint_angles(self, list_infos):
        joints_pos = [[],[],[],[],[],[],[],[]]
        for info in list_infos:
            for i in range(7):
                y = info['head_y']
                joints_pos[i] = info['joints_pos' + i]

        plt.figure(figsize=(10, 4))
        plt.plot(data_x, data_y)
        plt.savefig(self.dir + self.env_name + '_plot_joint_angles.pdf', dpi=600, bbox_inches='tight')
        #plt.show()

        plt.close('all')


    def plot_snake_gait(self, list_infos):

        size = 256, 256
        dpi = 300

        nrows = 10
        ncols = 1

        t_start = 200
        steps = 3
        timesteps = list(range(t_start, t_start+nrows*steps, steps))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)

        #for ax, t in zip(axes.flat[:], timesteps):
        for row in range(len(timesteps)):
            t = timesteps[row]
            ax1 = axes[0]
            ax = axes[row]

            joint_pos = list_infos[t]['joints_pos']
            joint_head_pos = list_infos[t]['joint_head_pos']

            head_x = list_infos[t]['head_x']
            head_y = list_infos[t]['head_y']

            m = 1

            # to degrees [-1.5,1.5] -> [-90,90]
            joint_pos = np.array(joint_pos) * 90 / 1.5
            joint_head_pos = joint_head_pos * 90 / 1.5

            # calculate endpoints
            # start from head
            joints_x, joints_y = calculate_gait_endpoints([head_x, head_y], joint_head_pos, joint_pos, m)


            #ax.set_title('t: %s' % t, fontsize=6)
            #plt.setp(axes[0].get_xticklabels(), fontsize=3)
            #ax.plot(joints_x, joints_y, '-o', linewidth=1, solid_joinstyle='round')

            ax.set_title('t: %s' % t, fontsize=6)
            plt.setp(ax.get_yticklabels(), fontsize=6, visible=True)

            plt.setp(ax.get_xticklabels(), fontsize=6, visible=True)


            ax.plot(joints_x, joints_y, marker='o', markersize=3, color='b', linestyle='-', linewidth=1, solid_joinstyle='round')
            ax.grid(color='grey', alpha=0.75, linestyle='-', linewidth=0.5, axis='both')


        plt.ylabel('y')
        plt.xlabel('x')
        #plt.ylim((-0.75, 0.75))

        # ax.figure(figsize=(10, 4))
        #fig.tight_layout()

        fig.subplots_adjust(hspace=0)

        #plt.show()
        plt.savefig(self.dir + self.env_name + '_snake_gait.png', dpi=600, bbox_inches='tight')
        plt.close('all')


    # Path Following Control and Poincaré Map Analysis, Liljeback
    # a stability analysis tool

    def plot_velocity_power_diagram(self, df, fname):
        #{x: float(np.mean(self.dict_list_infos[x][200:])) for x in columns}

        #v = [x.get_mean_dict_200('velocity') for x in info_collector_set]
        #e = [x.get_mean_dict_200('power_normalized') for x in info_collector_set]

        # somewhere at
        # 6cm/s
        # 143.4 mWh/m

        df['total_power_mWh'] = df['total_power_sec'] / 3600 * 1000


        #df.plot(x='velocity', y='power_normalized', c='w_para', kind='scatter', xlim = [0.0, 1.0], ylim=[0.0, 1.0])
        #df.plot(x='velocity', y='power_normalized', kind='scatter', xlim=[0.0, 1.0], ylim=[0.0, 1.0])
        #ax = df.plot(x='velocity', y='power', kind='scatter', xlim=[0.0, 1.0], ylim=[0.0, 10000.0], figsize=(8,6))
        ax = df.plot(x='velocity', y='total_power_mWh', kind='scatter', xlim=[0, 0.3], ylim=[0.0, 50.0],figsize=(8, 5))#, figsize=(9,6))

        ax.set_xlabel("Velocity [m/s]")
        ax.set_ylabel("Power [mW]")
        #ax.set_ylabel("energy [joules/s]")

        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(18)


        #plt.show()fname
        plt.savefig(self.dir + '/' + fname + '_velocity_power_diagram.pdf', dpi=600, bbox_inches='tight')
        plt.savefig(self.dir + '/' + fname + '_velocity_power_diagram.png', dpi=600, bbox_inches='tight')


    def plot_energy_distance_diagram(self, df, fname):
        #, xlim=[0.0, 1.0], ylim=[0.0, 1.0]
        # Ws to mWh
        df['total_power_mWh'] = df['total_power_sec'] / 3600 * 1000

        ax = df.plot(x='distance_delta', y='total_power_mWh', kind='scatter', xlim=[0.0, 10.0], ylim=[0.0, 50.0], figsize=(8, 5))

        ax.set_xlabel("Distance [m]")
        ax.set_ylabel("Energy [mW]")

        plt.savefig(self.dir + '/' + fname + '_energy_distance_diagram.pdf', dpi=600, bbox_inches='tight')
        plt.savefig(self.dir + '/' + fname + '_energy_distance_diagram.png', dpi=600, bbox_inches='tight')

    def plot_distance_bar_diagram(self, df):
        df2 = df.set_index(['injured_joint', 'id'])
        df2.sort_index(inplace=True)
        df2 = df2.groupby(['injured_joint', 'id']).mean()
        df2 = df2['distance_delta'].unstack()

        df2.plot(kind='bar')
        plt.savefig(self.dir + '/distance_diagram.pdf', dpi=600, bbox_inches='tight')

    def plot_hist_joint_power_diagram(self, df, fname):
        # power_0_0_5 = [0.76876904 1.20128084 0.79702089 0.80390111 0.41385619 1.12608577 1.22071898 4.69168658]
        # power_0_2_5 = [ 6.94321449  7.41863576 11.17242784  6.54425896  5.43228331  3.98019011 8.67987413  8.47925477]


        # vel_interval = np.array(0, 0.25, 0.01)
        pass










def calculate_gait_endpoints(head_pos, joint_head_pos, joint_pos, m=0.5):
    joints_x = [head_pos[0]]
    joints_y = [head_pos[1]]
    a = joint_head_pos

    for i in range(8):

        angle_rad = np.radians(a)
        a += joint_pos[i]
        # ankatete
        x = np.cos(angle_rad) * m
        # gegenkatete
        y = np.sin(angle_rad) * m

        joints_x.append(joints_x[-1] - x)
        joints_y.append(joints_y[-1] - y)

    return joints_x, joints_y


def plot_snake_track(data, maindir, dir, data_run, ax = None):
    # data for gait at t
    length = len(data)
    gaits_x = []
    gaits_y = []
    gatis_head_angle = []

    # for t in np.arange(0, 999, 150):
    for t in range(length):
        data_t = data.loc[t, :]
        joint_head_pos = data_t['joint_head_pos']
        joint_pos = str(data_t['joints_pos'])
        joint_pos = joint_pos.replace(']', "")
        joint_pos = joint_pos.replace('[', "")
        joint_pos = [float(x) for x in joint_pos.split(', ')]
        head_pos = [data_t['head_x'], data_t['head_y']]
        m = 0.5
        joint_pos = np.array(joint_pos) * 90 / 1.5
        joint_head_pos = joint_head_pos * 90 / 1.5
        gait_x, gait_y = calculate_gait_endpoints(head_pos, joint_head_pos, joint_pos, m)

        gaits_x.append(gait_x)
        gaits_y.append(gait_y)
        gatis_head_angle.append(joint_head_pos)

    # head and track
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    #plt.figure(figsize=(10,3))
    ax.plot(data['target_x'], data['target_y'], label='Track', color='blue')

    #center_x = [np.mean(x) for x in gaits_x]
    #center_y = [np.mean(y) for y in gaits_y]
    #plt.plot(center_x, center_y, label='snake_center', linestyle='-', color='yellow')

    plt.plot(data['head_x'], data['head_y'], label='Head module', linestyle='-', color='green')

    # plt.plot(data['target_x'], data['target_y']+1, label='target', color='blue', linestyle='--')
    # plt.plot(data['target_x'], data['target_y']-1, label='target', color='blue', linestyle='--')

    #ax = plt.gca() #aa

    # for i in range(len(gaits_x),100):
    # for i in np.arange(0, 999, 150):
    for i in [0, 999, 1999, 2999]:

        if i == 0:
            ax.plot(gaits_x[i], gaits_y[i], color="red", label='Movement')
        else:
            ax.plot(gaits_x[i], gaits_y[i], color="red")

        e1 = patches.Ellipse((data['target_x'][i], data['target_y'][i]), 0.2, 0.2, angle=gatis_head_angle[i],
                             linewidth=2,
                             fill=True, zorder=10, color='red')
        ax.add_patch(e1)

    ax.axis('equal')
    ax.legend()
    # plt.title('Position')
    plt.xlabel('x')
    plt.ylabel('y')

    # aspect ratio
    #ax = plt.gca()
    #ax.set_aspect(aspect=0.2)

    filename = '{}/{}/plot_head_trace_{}.png'.format(maindir, dir, data_run)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    filename = '{}/{}/plot_head_trace_{}.pdf'.format(maindir, dir, data_run)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close('all')
    plt.clf()

def plot_line_hist_diff(data, maindir, dir, data_run):

    aim_distance = data['target_distance'][0]
    length = len(data['head_target_distance'])
    x = range(length)
    y = data['head_target_distance']

    fig, ax = plt.subplots(figsize=(10,4))


    plt.plot(x, [aim_distance] * length, linestyle='--', color="red")
    ax.plot(x, y, label='target_distance', color="blue")

    plt.xlabel('Timestep')
    plt.ylabel('Distance between head and target')

    divider = make_axes_locatable(ax)
    axHisty = divider.append_axes("right", 1.4, pad=0.1, sharey=ax)
    axHisty.yaxis.set_tick_params(labelleft=False)

    binwidth = 0.2 #2 #0.5 #0.1
    binwidth = 2

    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax / binwidth) + 1) * binwidth
    bins = np.arange(-lim, lim + binwidth, binwidth)
    axHisty.hist(np.array(y), bins=bins, orientation='horizontal', color="blue")

    # Todo
    axHisty.set_xticks([0, length/2, length])

    plt.xlabel('Histogram')
    ax.set_ylim([aim_distance - 2.2, aim_distance + 2.2])

    # aspect ratio
    #ax.set_aspect(aspect=0.5)

    # plt.title('Target distance')
    filename = '{}/{}/plot_target_distance_{}.png'.format(maindir, dir, data_run)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    filename = '{}/{}/plot_target_distance_{}.pdf'.format(maindir, dir, data_run)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close('all')
    plt.clf()


def plot_head_trace_figure(data_list, maindir, dir):
    fig, ax = plt.subplots(4, 1, sharex='all', sharey='row', squeeze=False, figsize=(8,11))
    # fig, ax = plt.subplots(3, 2, squeeze=False)
    # fig.suptitle("Tracks", fontsize=16)

    #plt.setp(ax[1,1], xticks=[0, 10, 20, 30, 40, 50], yticks=[-10, 0, 10])

    ax[0, 0].set_ylabel("y [m]")
    ax[1, 0].set_ylabel("y [m]")
    ax[2, 0].set_ylabel("y [m]")
    ax[3, 0].set_ylabel("y [m]")
    ax[3, 0].set_xlabel("x [m]")

    ax[0, 0].axis('equal')
    ax[1, 0].axis('equal')

    start, end = ax[0, 0].get_xlim()
    plt.setp(ax[0, 0], xticks=[-10, 0, 10, 20, 30, 40, 50], yticks=[-10, -5, 0, 5, 10])
    plt.setp(ax[1, 0], xticks=[-10, 0, 10, 20, 30, 40, 50], yticks=[-10, -5, 0, 5, 10])
    plt.setp(ax[2, 0], xticks=[-10, 0, 10, 20, 30, 40, 50], yticks=[-10, -5, 0, 5, 10])
    plt.setp(ax[3, 0], xticks=[-10, 0, 10, 20, 30, 40, 50], yticks=[-10, -5, 0, 5, 10])

    data_order = [3,0,2,1]
    data_title = ['Line (1)', 'Zigzag (2)', 'Wave (3)', 'Random (4)']

    for d in range(4):
        data = data_list[data_order[d]]

        length = len(data)
        gaits_x = []
        gaits_y = []
        gatis_head_angle = []

        # for t in np.arange(0, 999, 150):
        for t in range(length):
            data_t = data.loc[t, :]
            joint_head_pos = data_t['joint_head_pos']
            joint_pos = str(data_t['joints_pos'])
            joint_pos = joint_pos.replace(']', "")
            joint_pos = joint_pos.replace('[', "")
            joint_pos = [float(x) for x in joint_pos.split(', ')]
            head_pos = [data_t['head_x'], data_t['head_y']]
            m = 0.5
            joint_pos = np.array(joint_pos) * 90 / 1.5
            joint_head_pos = joint_head_pos * 90 / 1.5
            gait_x, gait_y = calculate_gait_endpoints(head_pos, joint_head_pos, joint_pos, m)

            gaits_x.append(gait_x)
            gaits_y.append(gait_y)
            gatis_head_angle.append(joint_head_pos)

        # plt.figure(figsize=(10,3))
        ax[d,0].plot(data['target_x'], data['target_y'], label='Track', color='blue')

        # center_x = [np.mean(x) for x in gaits_x]
        # center_y = [np.mean(y) for y in gaits_y]
        # plt.plot(center_x, center_y, label='snake_center', linestyle='-', color='yellow')

        ax[d, 0].plot(data['head_x'], data['head_y'], label='Head module trace', linestyle='-', color='green')

        # plt.plot(data['target_x'], data['target_y']+1, label='target', color='blue', linestyle='--')
        # plt.plot(data['target_x'], data['target_y']-1, label='target', color='blue', linestyle='--')

        # ax = plt.gca() #aa

        # for i in range(len(gaits_x),100):
        # for i in np.arange(0, 999, 150):
        for i in [0, 1000, 2000, 2999]:

            if i == 0:
                ax[d, 0].plot(gaits_x[i], gaits_y[i], color="red", label='Movement')
            else:
                ax[d, 0].plot(gaits_x[i], gaits_y[i], color="red")

            e1 = patches.Ellipse((data['target_x'][i], data['target_y'][i]), 0.2, 0.2, angle=gatis_head_angle[i],
                                 linewidth=2,
                                 fill=True, zorder=10, color='red')
            ax[d, 0].add_patch(e1)

        ax[d, 0].axis('equal')
        ax[d, 0].legend(loc=2)
        ax[d,0].set_title(data_title[d], size=11)
        # aspect ratio
        # ax = plt.gca()
        # ax.set_aspect(aspect=0.2)


    # plt.setp([ax[2,0].get_xticklabels() ], visible=True)
    # ax[1, 1].set_xticklabels([ax[2, 0].get_xticklabels()], visible=True)
    # ax[2,1].set_xticklabels(["-20", "-10", "0", "10", "20"])
    # ax[1, 1].get_xticklabels().set_visible(True)

    # plt.setp(ax[2, 1].get_xticklabels(), visible=True)
    # ax[1, 1] = ax[2, 1]
    # ax[2, 1].set_visible(False)

    # plt.setp(ax[1, 1], visible=True)

    plt.tight_layout()

    for ax in ax.flatten():
        for tk in ax.get_yticklabels():
            tk.set_visible(True)
        for tk in ax.get_xticklabels():
            tk.set_visible(True)

    filename = '{}/{}/plot_head_traces.png'.format(maindir, dir)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    filename = '{}/{}/plot_head_traces.pdf'.format(maindir, dir)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close('all')
    plt.clf()


from matplotlib import gridspec
def plot_head_distance_figure(data_list, maindir, dir):
    fig, ax = plt.subplots(4, 2, sharex='col', sharey='row', squeeze=False, figsize=(8, 11))
    ax[0, 0].set_ylabel("Distance [m]")
    ax[1, 0].set_ylabel("Distance [m]")
    ax[2, 0].set_ylabel("Distance [m]")
    ax[3, 0].set_ylabel("Distance [m]")
    ax[3, 0].set_xlabel("Time Step")


    #ax[0, 0].axis('equal')
    #ax[1, 0].axis('equal')

    start, end = ax[0, 0].get_xlim()
    for row in range(4):
        #plt.setp(ax[row, 0], xticks=[0, 1000, 2000, 3000], yticks=[2, 3, 4, 5, 6])
        #plt.setp(ax[row, 1], xticks=[0, 1000], yticks=[2, 3, 4, 5, 6])
        plt.ylim((2, 6))
        #plt.setp(ax[row, 0], yticks=[2, 3, 4, 5, 6])


    data_order = [3, 0, 2, 1]
    data_title = ['Line (1)', 'Zigzag (2)', 'Wave (3)', 'Random (4)']


    gs = gridspec.GridSpec(4, 2, width_ratios=[6, 1])



    for d in range(4):
        data = data_list[data_order[d]]

        aim_distance = data['target_distance'][0]
        length = len(data['head_target_distance'])
        x = range(length)
        y = data['head_target_distance']

        #ax[d, 0].set_title("Line (1)", size=11, loc='center')

        # line
        ax0 = plt.subplot(gs[d*2])
        ax0.plot(x, [aim_distance] * length, linestyle='--', color="red")
        ax0.plot(x, y, label='target_distance', color="blue")
        ax0.set_ylim([aim_distance - 2.2, aim_distance + 2.2])
        ax0.set_xlim([-50, 3050])

        ax0.set_ylabel("Distance [m]")
        if d ==3:
            ax0.set_xlabel("Time Step")

        #data_title[d]
        plt.title(data_title[d], x=0.61, size=11, loc='center', horizontalalignment='center', verticalalignment='center')

        # ax.axis('equal')
        #ax[d,0].legend(loc=2)


        # hist
        ax1 = plt.subplot(gs[(d)*2+1])
        ax1.hist(np.array(y), bins=20, orientation='horizontal', color="blue")
        # ax1.set_xticks([0, length / 2, length])

        if d == 3:
            ax1.set_xlabel('Histogram')
        ax1.set_ylim([aim_distance - 2.2, aim_distance + 2.2])
        ax1.yaxis.set_tick_params(labelleft=False)
        ax1.set_xlim([0, 610])# todo

        plt.tight_layout()

    #for ax in ax.flatten():
    #    for tk in ax.get_yticklabels():
    #        tk.set_visible(True)
    #    for tk in ax.get_xticklabels():
    #        tk.set_visible(True)

    filename = '{}/{}/plot_target_distance.png'.format(maindir, dir)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    filename = '{}/{}/plot_target_distance.pdf'.format(maindir, dir)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close('all')
    plt.clf()



def plot_head_distance_figure2(data_list, maindir, dir):
    fig = plt.figure(figsize=(8,11))
    outer = gridspec.GridSpec(4, 1, wspace=0.2, hspace=0.2)
    #fig, ax = plt.subplots(4, 1, sharex='all', sharey='row', squeeze=False, )
    # fig, ax = plt.subplots(3, 2, squeeze=False)
    # fig.suptitle("Tracks", fontsize=16)

    #plt.setp(ax[1,1], xticks=[0, 10, 20, 30, 40, 50], yticks=[-10, 0, 10])

    for i in range(4):
        inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[i], wspace=0.2, hspace=0.2)

        data_order = [3, 0, 2, 1]
        data_title = ['Line (1)', 'Zigzag (2)', 'Wave (3)', 'Random (4)']

        data = data_list[data_order[i]]

        aim_distance = data['target_distance'][0]
        length = len(data['head_target_distance'])
        x = range(length)
        y = data['head_target_distance']

        #ax[0, 0].set_ylabel("Distance [m]")
        #ax[1, 0].set_ylabel("Distance [m]")
        #ax[2, 0].set_ylabel("Distance [m]")
        #ax[3, 0].set_ylabel("Distance [m]")
        #ax[3, 0].set_xlabel("Time Step [m]")

        #ax[0, 0].axis('equal')
        #ax[1, 0].axis('equal')

        #start, end = ax[0, 0].get_xlim()
        #plt.setp(ax[0, 0], xticks=[-10, 0, 10, 20, 30, 40, 50], yticks=[-10, -5, 0, 5, 10])
        #plt.setp(ax[1, 0], xticks=[-10, 0, 10, 20, 30, 40, 50], yticks=[-10, -5, 0, 5, 10])
        #plt.setp(ax[2, 0], xticks=[-10, 0, 10, 20, 30, 40, 50], yticks=[-10, -5, 0, 5, 10])
        #plt.setp(ax[3, 0], xticks=[-10, 0, 10, 20, 30, 40, 50], yticks=[-10, -5, 0, 5, 10])

        plt.xlabel('Timestep')
        plt.ylabel('Distance')


        # aspect ratio
        # ax.set_aspect(aspect=0.5)


        #ax.set_title(data_title[i], size=11)


        # line
        ax = plt.Subplot(fig, inner[0])

        ax.plot(x, [aim_distance] * length, linestyle='--', color="red")
        ax.plot(x, y, label='target_distance', color="blue")

        #ax.axis('equal')
        ax.legend(loc=2)

        fig.add_subplot(ax)




        # hist
        ax = plt.Subplot(fig, inner[1])
        #divider = make_axes_locatable(ax)
        #axHisty = divider.append_axes("right", 1.4, pad=0.1, sharey=ax)
        ax.yaxis.set_tick_params(labelleft=False)

        binwidth = 0.2  # 2 #0.5 #0.1
        binwidth = 2

        xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
        lim = (int(xymax / binwidth) + 1) * binwidth
        bins = np.arange(-lim, lim + binwidth, binwidth)
        ax.hist(np.array(y), bins=bins, orientation='horizontal', color="blue")

        #ax.set_xticks([0, length / 2, length])

        plt.xlabel('Histogram')
        #ax.set_ylim([aim_distance - 2.2, aim_distance + 2.2])




        plt.tight_layout()
        fig.add_subplot(ax)


        # plt.setp([ax[2,0].get_xticklabels() ], visible=True)
        # ax[1, 1].set_xticklabels([ax[2, 0].get_xticklabels()], visible=True)
        # ax[2,1].set_xticklabels(["-20", "-10", "0", "10", "20"])
        # ax[1, 1].get_xticklabels().set_visible(True)

        # plt.setp(ax[2, 1].get_xticklabels(), visible=True)
        # ax[1, 1] = ax[2, 1]
        # ax[2, 1].set_visible(False)

        # plt.setp(ax[1, 1], visible=True)


        #plt.tight_layout()


        #for ax in ax.flatten():
        #    for tk in ax.get_yticklabels():
        #        tk.set_visible(True)
        #    for tk in ax.get_xticklabels():
        #        tk.set_visible(True)

    filename = '{}/{}/plot_target_distance.png'.format(maindir, dir)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    filename = '{}/{}/plot_target_distance.pdf'.format(maindir, dir)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close('all')
    plt.clf()


def save_distance_std(data_list, maindir, dir):
    data_order = [3, 0, 2, 1]
    data_title = ['Line (1)', 'Zigzag (2)', 'Wave (3)', 'Random (4)']


    df = pd.concat(data_list)

    df_g = df.groupby('env')
    #res = df_g.describe(include='head_target_distance, velocity')
    res = df_g.describe()

    filename = '{}/{}/describe.csv'.format(maindir, dir)
    res.to_csv(filename)

    #for d in range(4):
    #    data = data_list[data_order[d]]
    #    print(data['head_target_distance'].describe())



def evaluate_target_tracking(dir= None):

    if dir is None:
        #maindir = '/home/chris/openai_logdir/'
        logger.configure()
        maindir = osp.join(logger.get_dir(), '../../')


        pattern = "following_eval_results*"
        dirs = [d for d in os.listdir(maindir) if fnmatch.fnmatch(d, pattern)]
        dirs.sort()
        dir = dirs[-1]

    data_runs = [d for d in os.listdir(maindir+dir) if fnmatch.fnmatch(d, "data_run*.csv")]


    data_list = [pd.read_csv(maindir + dir + '/' + data_run) for data_run in data_runs]

    #stats
    save_distance_std(data_list, maindir, dir)

    #
    plot_head_trace_figure(data_list, maindir, dir)

    #
    for data_run in data_runs:
        # plot_snake_track
        data = pd.read_csv(maindir + dir + '/' + data_run)

        #hotfix for bug
        m = np.mean(data['velocity'])
        if m < 0.01:
            print(m)
            #continue

        plot_snake_track(data, maindir, dir, data_run)



    #dist fig
    plot_head_distance_figure(data_list, maindir, dir)

    """
    for data_run in data_runs:
        # distance
        data = pd.read_csv(maindir + dir + '/' + data_run)

        m = np.mean(data['velocity'])
        if m < 0.01:
            print(m)
            #continue

        # TODO
        plot_line_hist_diff(data, maindir, dir, data_run)
    """


    # angle Absolut
    """
    plt.figure(figsize=(10, 4))
    plt.plot(range(1000), data['head_angle_z'], label='head_angle_z')
    plt.plot(range(1000), data['target_angle_z'], label='target_angle_z')
    plt.plot(range(1000), data['orientation'], label='orientation')

    #plt.plot(range(1000), data['angle_t'], label='angle_t')
    #plt.plot(range(1000), data['joints_pos_mean']*90, label='joints_pos_mean_angle')
    
    plt.legend()
    plt.axis('equal')
    #plt.title('Absolut angle')
    plt.xlabel('Timestep')
    plt.ylabel('Angle (degree)')
    filename = '{}/{}/plot_absolut_angle_{}.png'.format(maindir, dir, data_run)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    filename = '{}/{}/plot_absolut_angle_{}.pdf'.format(maindir, dir, data_run)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close('all')
    plt.clf()
    """

    # angle relative
    """
    plt.figure(figsize=(10, 4))
    plt.plot(range(1000), data['angle_t'], label='angle_t')
    plt.xlabel('Timestep')
    plt.ylabel('Angle (degree)')


    # plt.plot(range(1000), data['joints_pos_mean']*90, label='joints_pos_mean_angle')
    plt.legend()
    plt.axis('equal')
    #plt.title('Relative angle')
    axes = plt.gca()
    axes.set_ylim([-220.0, 220.0])
    filename = '{}/{}/plot_relative_angle_{}.png'.format(maindir, dir, data_run)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    filename = '{}/{}/plot_relative_angle_{}.pdf'.format(maindir, dir, data_run)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close('all')
    plt.clf()
    """

    """
    plt.figure(figsize=(10, 4))
    plt.plot(range(1000), data['target_distance_diff'], label='target_distance_diff')
    plt.legend()
    # plt.axis('equal')
    axes = plt.gca()
    # axes.set_xlim([xmin, xmax])
    axes.set_ylim([-4.0, 4.0])
    plt.title('Target distance')
    filename = '{}/{}/plot_target_distance_{}.png'.format(maindir, dir, data_run)
    plt.savefig(filename, dpi=300)
    filename = '{}/{}/plot_target_distance_{}.pdf'.format(maindir, dir, data_run)
    plt.savefig(filename, dpi=300)
    plt.close('all')
    plt.clf()
    """


def evaluate_locomotion_control():
    """
    Plot the power_velocity results

    Combines the grid search result from the traditional controller and the grid search result from the poo controller.
    TODO set file names!
    :return:
    """
    # TODO
    eq_fname = 'grid_search_results_2018-05-29_17:36:03'

    #eq_data = pd.read_csv('/home/chris/openai_logdir/' + eq_fname + '.csv')
    logger.configure()
    logdir = osp.join(logger.get_dir(), '../../')
    eq_data = pd.read_csv( logdir + '/' + eq_fname + '.csv')

    eq_data['controller'] = 15

    eq_data.plot(x='velocity', y='power', kind='scatter',
                 title='Grid search result: Equation-controller on velocity and power', xlim=[0, None],
                 ylim=[0.0, 130000.0], figsize=(9, 6))
    plt.savefig(logdir + '/' + eq_fname + '_velocity_power_diagram.pdf')

    # fname = 'grid_search_results_2018-05-31_22:14:47__power_new7'
    # fname = 'grid_search_results_2018-06-02_22:36:23_new12'

    fname = 'grid_search_results_2018-05-29_17:11:29_new5'
    #ppo_data = pd.read_csv('/home/chris/openai_logdir/' + fname + '.csv')
    ppo_data = pd.read_csv(logdir + '/' + fname + '.csv')


    ppo_data['controller'] = 16

    ppo_data.plot(x='velocity', y='power', kind='scatter',
                  title='Grid search result: PPO-controller on velocity and power', xlim=[0, None],
                  ylim=[0.0, None], figsize=(9, 6))
    #plt.savefig('/home/chris/openai_logdir' + '/' + fname + '_velocity_power_diagram.pdf')
    plt.savefig(logdir + '/' + fname + '_velocity_power_diagram.pdf')

    df = pd.concat([eq_data, ppo_data])

    # df.plot(x='velocity', y='power', c='controller', kind='scatter', colors=['r', 'g', 'b'], xlim=[0.0, 1.0], ylim=[0.0, 130000.0])


    ax = eq_data.plot(kind='scatter', x='velocity', y='power', label='Equation-controller')

    ppo_data.plot(title='Comparison Equation- and PPO-controller', kind='scatter', x='velocity', y='power', color='red',
                  label='PPO-controller', ax=ax, xlim=[0.0, None], ylim=[0.0, None], figsize=(9, 6))

    #plt.savefig('/home/chris/openai_logdir' + '/' + fname + '_velocity_power_diagram_compare.pdf')
    plt.savefig(logdir + '/' + fname + '_velocity_power_diagram_compare.pdf')


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # evaluation takes time better do one at a time

    #todo uncomment
    #evaluate_target_tracking()
    #return

    # todo uncomment
    evaluate_locomotion_control()


    #args = parser.parse_args()
    #plots = Plots(None)
    #plots.plot_snake_gait(None)


if __name__ == '__main__':
    main()

