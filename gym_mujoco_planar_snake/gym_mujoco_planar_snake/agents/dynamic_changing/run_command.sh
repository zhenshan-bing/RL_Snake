#!/bin/sh
for i in 1 2 3
do
  python run_mujoco_ppo.py --train=1 --env Mujoco-planar-snake-cars-angle-line-v1
  python run_mujoco_ppo.py --env Mujoco-planar-snake-cars-angle-line-v1 --evaluate_power_velocity True
done