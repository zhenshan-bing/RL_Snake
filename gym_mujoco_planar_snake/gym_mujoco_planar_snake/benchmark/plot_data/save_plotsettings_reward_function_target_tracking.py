import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')


def reward(dist, target_distance, distance_radius):
    reward = 1- (np.abs(target_distance - dist))/ distance_radius
    return reward

def plot_td_reward():

    dist = np.linspace(2, 6, 4096, endpoint=True)

    #plt.plot(power, p_reward(power, 0.5, 0.8), color="red", label="b1=0.5")
    #plt.plot(power, p_reward(power, 0.35, 0.8), color="blue", label="b1=0.35")
    #plt.plot(power, p_reward(power, 0.2, 0.8), color="green", label="b1=0.2")
    plt.figure(figsize=(8, 5))

    plt.plot(dist, reward(dist, target_distance=4.0, distance_radius=2), color="red")

    #plt.gca().set_aspect(aspect=0.7)

    #plt.legend(loc='upper right')
    plt.xlabel("Distance [m]")
    plt.ylabel("Reward")
    #plt.title("Power reward function")

    #plt.ylim(-0.05, 1.05)
    plt.ylim(-0.04, 1.04)
    #plt.xlim(-0.05, 6.05)
    plt.xlim(1.9, 6.1)

    #plt.show()

    plt.savefig("reward_target_tracking.pdf", dpi=300, bbox_inches='tight')

    plt.clf()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    plot_td_reward()


if __name__ == '__main__':
    main()

