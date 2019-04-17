import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import math
import matplotlib.ticker as ticker

from gym_mujoco_planar_snake.benchmark.tracks_generator import TracksGenerator

matplotlib.style.use('ggplot')

def create_start_dot(ax):
    #ax.arrow(2, 2, 1, 0,head_width=1, head_length=3, color='red')
    ax.plot([0], [0], 'o', color="black")


def plot_line(ax):

    x = np.linspace(0, 50, 4096, endpoint=True)
    y = np.zeros(len(x))

    ax.plot(x, y, color="red", label="line")


    create_start_dot(ax)

    #ax.set_xlim((-28, 28))
    #ax.set_ylim((-20, 20))

    ax.set_title("Line (1)", size=11)
    #ax.set_title("Line (1)")
    #plt.savefig("track_line.pdf", dpi=300)


def plot_wave(ax):
    tg = TracksGenerator()
    x = np.linspace(-2, 48, 4096, endpoint=True)

    x =  [tg.gen_wave_step(x, 0, x, 0, 0.05)[0] for x in x]
    y = [tg.gen_wave_step(x, 0, x, 0, 0.05)[1] for x in x]

    ax.plot(x, y, color="red", label="wave")
    create_start_dot(ax)
    #plt.xlabel("x")
    #plt.ylabel("y")
    ax.set_title("Wave (3)", size=11)
    #x.set_title("Wave (3)")
    #plt.savefig("track_wave.pdf", dpi=300)


def plot_zigzag(ax):
    tg = TracksGenerator()
    x = np.linspace(-2, 48, 4096, endpoint=True)

    x = [tg.gen_zigzag_step(x, 0, x, 0, 0.05, True)[0] for x in x]
    y = [tg.gen_zigzag_step(x, 0, x, 0, 0.05, True)[1] for x in x]

    ax.plot(x, y, color="red", label="zigzag")

    create_start_dot(ax)
    #plt.xlabel("x")
    #plt.ylabel("y")
    ax.set_title("ZigZag (2)", size=11)
    #ax.set_title("ZigZag (5)")
    #plt.savefig("track_zigzag.pdf", dpi=300)


def plot_circle(ax):
    start_sin_at = 5 #8
    start_sin_at = 10  # 8
    head_target_dist = 2

    tg = TracksGenerator()
    x = np.linspace(0, start_sin_at, 4096, endpoint=True)
    y = np.zeros(len(x))

    radius = start_sin_at

    #fig2, ax2 = ax.subplots()

    c1 = plt.Circle((0, 0), start_sin_at, color='red', fill=False, linewidth=1.5)

    ax.plot(x, y, color="red", label="line")
    create_start_dot(ax)
    ax.add_artist(c1)
    #ax.add_artist(p2)

    ax.set_xlim((-28, 28))
    ax.set_ylim((-32, 32))

    #plt.xlabel("x")
    #plt.ylabel("y")
    ax.set_title("Circle (2)", size=11)
    #ax.set_title("Circle (2)")
    #plt.savefig("track_circle.pdf", dpi=300)


def plot_random(ax):
    segment_length = 3
    segments = 15
    seed = 21#11 #10 #7 #4 #6

    tg = TracksGenerator()

    x = [0]
    y = [0]
    for i in range(1000):
        x2, y2 = tg.gen_random_step(0, 0, x[-1], y[-1], 180/1000, seed=seed)
        x.append(x2)
        y.append(y2)


    ax.plot(x, y, label="random", color='red')

    ax.set_ylim(-10, 10)
    create_start_dot(ax)


    #plt.xlabel("x")
    #plt.ylabel("y")
    ax.set_title("Random (4)", size=11)
    #ax.set_title("Random (4)")
    #plt.gca().set_aspect('equal', adjustable='box')
    #plt.savefig("track_random.pdf", dpi=300)



def tracks_with_circle():
    fig, ax = plt.subplots(3, 2, sharex='col', sharey='col', squeeze=False)
    # fig, ax = plt.subplots(3, 2, squeeze=False)
    # fig.suptitle("Tracks", fontsize=16)

    plt.setp(ax, xticks=[0, 10, 20, 30, 40, 50], yticks=[-10, 0, 10])

    """
    fs = 9
    ax[0, 0].set_ylabel("y", fontsize=fs)
    ax[1, 0].set_ylabel("y", fontsize=fs)
    ax[2, 0].set_ylabel("y", fontsize=fs)

    ax[2, 0].set_xlabel("x", fontsize=fs)
    ax[1, 1].set_xlabel("x", fontsize=fs)
    """
    ax[0, 0].set_ylabel("y")
    ax[1, 0].set_ylabel("y")
    ax[2, 0].set_ylabel("y")

    ax[2, 0].set_xlabel("x")
    ax[1, 1].set_xlabel("x")

    ax[0, 0].axis('equal')
    ax[1, 0].axis('equal')
    ax[2, 0].axis('equal')
    ax[0, 1].axis('equal')
    ax[1, 1].axis('equal')

    start, end = ax[0, 0].get_xlim()

    plt.setp(ax[0, 1], xticks=[-20, -10, 0, 10, 20], yticks=[-10, 0, 10])

    # ax[0, 0].xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))


    # ax[1, 1].set_xticklabels(['a','f'])

    plot_line(ax[0, 0])
    plot_wave(ax[1, 0])
    plot_zigzag(ax[2, 0])

    plot_circle(ax[0, 1])
    plot_random(ax[1, 1])
    # plt.setp([ax[2,0].get_xticklabels() ], visible=True)
    # ax[1, 1].set_xticklabels([ax[2, 0].get_xticklabels()], visible=True)
    # ax[2,1].set_xticklabels(["-20", "-10", "0", "10", "20"])
    # ax[1, 1].get_xticklabels().set_visible(True)

    # plt.setp(ax[2, 1].get_xticklabels(), visible=True)
    # ax[1, 1] = ax[2, 1]
    # ax[2, 1].set_visible(False)

    # plt.setp(ax[1, 1], visible=True)

    fig.delaxes(ax[2, 1])

    plt.tight_layout()
    # fig.subplots_adjust(top=0.9, left=0.08, right=0.98, bottom=0.1)

    # fig.subplots_adjust(hspace=0.3)

    for ax in ax.flatten():
        for tk in ax.get_yticklabels():
            tk.set_visible(True)
        for tk in ax.get_xticklabels():
            tk.set_visible(True)

    fig.savefig("tracks.pdf", dpi=300, bbox_inches='tight')



def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    fig, ax = plt.subplots(2, 2, sharex='all', sharey='all', squeeze=False, figsize=(10,5))
    # fig, ax = plt.subplots(3, 2, squeeze=False)
    # fig.suptitle("Tracks", fontsize=16)

    #plt.setp(ax[1,1], xticks=[0, 10, 20, 30, 40, 50], yticks=[-10, 0, 10])

    #ax[0, 0].set_aspect(aspect=0.2)
    """
    fs = 9
    ax[0, 0].set_ylabel("y", fontsize=fs)
    ax[1, 0].set_ylabel("y", fontsize=fs)
    ax[2, 0].set_ylabel("y", fontsize=fs)

    ax[2, 0].set_xlabel("x", fontsize=fs)
    ax[1, 1].set_xlabel("x", fontsize=fs)
    """
    ax[0, 0].set_ylabel("y [m]")
    ax[1, 0].set_ylabel("y [m]")
    ax[1, 0].set_xlabel("x [m]")
    ax[1, 1].set_xlabel("x [m]")

    ax[0, 0].axis('equal')
    ax[1, 0].axis('equal')
    ax[0, 1].axis('equal')
    ax[1, 1].axis('equal')

    start, end = ax[0, 0].get_xlim()
    plt.setp(ax[0, 0], xticks=[-10, 0, 10, 20, 30, 40, 50], yticks=[-10, -5, 0, 5, 10])



    # ax[0, 0].xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
    # ax[1, 1].set_xticklabels(['a','f'])

    plot_line(ax[0, 0])
    plot_wave(ax[1, 0])
    plot_zigzag(ax[0, 1])
    plot_random(ax[1, 1])
    # plt.setp([ax[2,0].get_xticklabels() ], visible=True)
    # ax[1, 1].set_xticklabels([ax[2, 0].get_xticklabels()], visible=True)
    # ax[2,1].set_xticklabels(["-20", "-10", "0", "10", "20"])
    # ax[1, 1].get_xticklabels().set_visible(True)

    # plt.setp(ax[2, 1].get_xticklabels(), visible=True)
    # ax[1, 1] = ax[2, 1]
    # ax[2, 1].set_visible(False)

    # plt.setp(ax[1, 1], visible=True)


    plt.tight_layout()
    # fig.subplots_adjust(top=0.9, left=0.08, right=0.98, bottom=0.1)
    # fig.subplots_adjust(hspace=0.3)



    for ax in ax.flatten():
        for tk in ax.get_yticklabels():
            tk.set_visible(True)
        for tk in ax.get_xticklabels():
            tk.set_visible(True)

    fig.savefig(".tracks.pdf", dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()

