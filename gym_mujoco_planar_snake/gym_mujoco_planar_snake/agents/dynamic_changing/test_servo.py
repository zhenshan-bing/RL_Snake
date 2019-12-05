#!/usr/bin/env python3
"""
Shows how to toss a capsule to a container.
"""
from mujoco_py import load_model_from_path, MjSim, MjViewer, MjSimState
import os
from pprint import pprint
import time
import numpy as np
import matplotlib.pyplot as plt

import csv

time_list = []

with open('nowtime.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        time_list.append(row)

time_list_array = np.array(time_list)

measured_angle_list = []

with open('now.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        measured_angle_list.append(float(row[0]) - 90)
        
calculated_angle_list = []

with open('real.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        calculated_angle_list.append(float(row[0]) - 90)

import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--render', help='render', type=bool, default=False)
args = parser.parse_args()

model = load_model_from_path("test_servo.xml")
sim = MjSim(model)

joints = ['servo_1']
joints_idx = list(map(lambda x: sim.model.joint_name2id(x), joints))
qpos = sim.data.qpos
joints_pos = qpos[joints_idx]

viewer = MjViewer(sim)

sim_state = sim.get_state()

target_position_list = []
sim_position_list = []

i = 0

while (i < 10):

    target_position = 30 * np.sin((i+0) * 4 * np.pi) + 0
    target_position = target_position / 180 * np.pi 
    target_position_list.append(target_position * 180 / np.pi)

    # time.sleep(0.02)
    j = 0
    while (j < 10):
        vel_before_action = sim.data.qvel[joints_idx]
        pos_before_action = sim.data.qpos[joints_idx]

        error_position = target_position - pos_before_action
        print(error_position)
        sim.data.ctrl[:] = (target_position - error_position) * 5000
        sim.step()
        j+=1

    

    real_position = sim.data.qpos[joints_idx]
    sim_position_list.append(real_position*180/np.pi)

    if args.render:
        viewer.render()
        pass

    if os.getenv('TESTING') is not None:
        break

    i+=0.01

step_index = np.arange(0, len(sim_position_list))

fig, ax = plt.subplots(figsize=(9,6))
ax.plot(step_index*0.01, target_position_list, '-', alpha=0.2)
# ax.plot(step_index*1, calculated_angle_list, '--', alpha=0.2)
# ax.plot(step_index*1, measured_angle_list)
ax.plot(step_index*0.01, sim_position_list)
# plt.xlim(300, 400)
plt.grid()
ax.legend(["sim_target", "calculate", "measured", "observe (mujoco)"], loc="upper left")

fig.tight_layout()
plt.savefig('test_servo.pdf', bbox_inches='tight')
plt.show()
