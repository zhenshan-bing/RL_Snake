import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')

def v_reward(velocity, a1, a2, target_v):
    rew_v = (1.0 - (np.abs(target_v - velocity) / a1)) ** (1.0 / a2)

    return rew_v

def plot_v_reward():

    target_v = 0.1
    velocity = np.linspace(0.0, 0.3, 4096, endpoint=True)

    #C, S = np.cos(X), np.sin(X)

    #plt.plot(velocity, v_reward(velocity, 0.5, 0.6, target_v), color="red", label="a1=0.5, a2=0.6")
    #plt.plot(velocity, v_reward(velocity, 0.5, 0.2, target_v), color="blue", label="a1=0.5, a2=0.2")
    #plt.plot(velocity, v_reward(velocity, 0.5, 0.04, target_v), color="green", label="a1=0.5, a2=0.04")
    plt.figure(figsize=(8, 5))

    plt.plot(velocity, v_reward(velocity, 0.2, 0.3, target_v), color="red", label="a1=0.2, a2=0.3")
    plt.plot(velocity, v_reward(velocity, 0.2, 0.2, target_v), color="blue", label="a1=0.2, a2=0.2")
    plt.plot(velocity, v_reward(velocity, 0.2, 0.1, target_v), color="green", label="a1=0.2, a2=0.1")


    plt.legend(loc='upper right')
    plt.xlabel("Velocity [m/s]")
    plt.ylabel("Reward")
    #plt.title("Velocity reward function")

    #plt.gca().set_aspect(aspect=0.7)

    #plt.ylim(-0.2, 1)

    #plt.show()

    plt.savefig("./power_velocity/reward_v_2.pdf", dpi=300, bbox_inches='tight')
    plt.clf()


def p_reward(power, b1, rew_v):
    rew_p = np.abs(1.0 - power) ** (b1 ** (-2.0))
    reward = rew_v * rew_p
    return reward

def plot_p_reward():

    power = np.linspace(0, 1, 4096, endpoint=True)

    #plt.plot(power, p_reward(power, 0.5, 0.8), color="red", label="b1=0.5")
    #plt.plot(power, p_reward(power, 0.35, 0.8), color="blue", label="b1=0.35")
    #plt.plot(power, p_reward(power, 0.2, 0.8), color="green", label="b1=0.2")
    plt.figure(figsize=(8, 5))

    plt.plot(power, p_reward(power, 0.8, 0.8), color="red", label="b1=0.8")
    plt.plot(power, p_reward(power, 0.6, 0.8), color="blue", label="b1=0.6")
    plt.plot(power, p_reward(power, 0.4, 0.8), color="green", label="b1=0.4")

    #plt.gca().set_aspect(aspect=0.7)

    plt.legend(loc='upper right')
    plt.xlabel("Normalized power")
    plt.ylabel("Reward")
    #plt.title("Power reward function")

    plt.ylim(-0.05, 1.05)
    #plt.xlim(-0.05, 1.05)

    #plt.show()

    plt.savefig("./power_velocity/reward_p_2.pdf", dpi=300, bbox_inches='tight')

    plt.clf()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    plot_v_reward()
    plot_p_reward()


if __name__ == '__main__':
    main()

