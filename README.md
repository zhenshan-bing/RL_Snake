<img src="images/env_image.png" width=40% align="right" />


# Reinforcement Learning for Autonomous Locomotion Control of Snake-Like Robots


Development of an artificial intelligent robot controller that performs a power efficient locomotion at a specified velocity.



## Abstract
<img src="images/v25_01.gif" width=40% align="right" />

<p align="justify">
Similar to their counterparts in nature, the flexible bodies of snake-like robots enhance their movement capability and adaptability in diverse environments. 
However, this flexibility corresponds to a complex control task involving highly redundant degrees of freedom, where traditional model-based methods usually fail to propel the robots energy-efficiently.
In this work, we present a novel approach for designing an energy-efficient slithering gait for a snake-like robot using a model-free reinforcement learning (RL) algorithm.
Specifically, 
we present an RL-based controller for generating locomotion gaits at a wide range of velocities, which is trained using the proximal policy optimization (PPO) algorithm.
Meanwhile, a traditional parameterized gait controller is presented and the parameter sets are optimized using the grid search and Bayesian optimization algorithms for the purposes of reasonable comparisons.
Based on the analysis of the simulation results, we demonstrate that this RL-based controller exhibits very natural and adaptive movements, which are also substantially more energy-efficient than the gaits generated by the parameterized controller.
Videos can be found at https://videoviewsite.wixsite.com/rlsnake.
</p>


## Setup


### System:
- Ubuntu 16.04 Xenial 
- Python 3.5 (try to avoid Anaconda, I had problems with it while installing mujoco)

### Requirements:
- Tensorflow
  - If GPU version: Might need cuda version 8. Might need specific nvidia driver "nvidia-384".

- mujoco 1.50
- mujoco-py >=1.50.1
- gym[mujoco]>=0.9.6'
- glfw>=1.4.0
- Cython>=0.27.2
- Baselines
- Pandas



### Installation:
For MuJoCo and mujoco-py follow the install instructions at https://github.com/openai/mujoco-py. Try to run the 
Install Gym at https://github.com/openai/gym. Try to run a mujoco example.


```bash
git clone https://github.com/Superchicken1/MasterThesis.git
cd gym_mujoco_planar_snake
pip install -e .
```
or add it in the PYTHONPATH like below.

### System Variables:
I can recomend to use the ~/.bashrc file to set system variables.

Add the following line to the end of the file:
```bash
# Used for tensorboard logs, benchmarks and models
export OPENAI_LOGDIR=$HOME/openai_logdir/tensorboard/x_new
export OPENAI_LOG_FORMAT="stdout,log,csv,json,tensorboard"

# Maybe needed for mujoco
export MUJOCO_PY_MJPRO_PATH=$HOME/.mujoco/mjpro150
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:~/.mujoco/mjpro150/bin
#LD_LIBRARY_PATH=$HOME/.mujoco/mjpro150/bin pip install mujoco-py

# Alternative to pip
#export GYM_MUJOCO_PATH={path to project}/gym-mujoco-planar-snake
#export PYTHONPATH="$GYM_MUJOCO_PATH/gym-mujoco-planar-snake:$PYTHONPATH"

# Some fixes to get GLEW to work. Uncomment if needed
# GODLIKE FIX ... GLEW init error
#export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-384/libGL.so
# X Error of failed request:  BadAccess (attempt to access private resource denied)
#export QT_X11_NO_MITSHM=1

```

### Run it
Be sure you set up the openai_logdir directory (did't test jet if directorys have to be created).
For parameters info use --help or look into run_mujoco_ppo.py.
At the moment at every start, a empty GLEW windows opens up with fullscreen. Do not close it just minimize it.


#### 1. Experiment: Power efficient locomotion 
**PPO controller:**

Train:
```bash 
gym_mujoco_planar_snake/agents/run_mujoco_ppo.py --train=1 --env Mujoco-planar-snake-cars-angle-line-v1
```
The model is stored in OPENAI_LOGDIR/models/ENV/PPO.

Enjoy:
```bash 
gym_mujoco_planar_snake/agents/run_mujoco_ppo.py --env Mujoco-planar-snake-cars-angle-line-v1
```

Evaluate:
```bash 
gym_mujoco_planar_snake/agents/run_mujoco_ppo.py --env Mujoco-planar-snake-cars-angle-line-v1 --evaluate_power_velocity
```
The results are stored in OPENAI_LOGDIR/power_velocity.


**Equation controller:**

Enjoy:
```bash 
gym_mujoco_planar_snake/agents/run_mujoco_coded_control.py --env Mujoco-planar-snake-cars-angle-line-v1
```

Evaluate:
```bash 
gym_mujoco_planar_snake/agents/run_mujoco_coded_control.py --env Mujoco-planar-snake-cars-angle-line-v1 --evaluate_power_velocity
```
The results are stored in OPENAI_LOGDIR/power_velocity.




Create comparison plot:
```bash 
gym_mujoco_planar_snake/benchmark/plots.py
```
First change the corresponding filenames in the evaluate_locomotion_control() method.
Check which evaluation is active in main(). 
The results are stored in OPENAI_LOGDIR.





#### Plots and data
See the files in directory [gym_mujoco_planar_snake/benchmark/plot_data].


