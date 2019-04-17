# ~/.bashrc: executed by bash(1) for non-login shells.
# see /usr/share/doc/bash/examples/startup-files (in the package bash-doc)
# for examples

# If not running interactively, don't do anything
case $- in
    *i*) ;;
      *) return;;
esac

# don't put duplicate lines or lines starting with space in the history.
# See bash(1) for more options
HISTCONTROL=ignoreboth

# append to the history file, don't overwrite it
shopt -s histappend

# for setting history length see HISTSIZE and HISTFILESIZE in bash(1)
HISTSIZE=1000
HISTFILESIZE=2000

# check the window size after each command and, if necessary,
# update the values of LINES and COLUMNS.
shopt -s checkwinsize

# If set, the pattern "**" used in a pathname expansion context will
# match all files and zero or more directories and subdirectories.
#shopt -s globstar

# make less more friendly for non-text input files, see lesspipe(1)
[ -x /usr/bin/lesspipe ] && eval "$(SHELL=/bin/sh lesspipe)"

# set variable identifying the chroot you work in (used in the prompt below)
if [ -z "${debian_chroot:-}" ] && [ -r /etc/debian_chroot ]; then
    debian_chroot=$(cat /etc/debian_chroot)
fi

# set a fancy prompt (non-color, unless we know we "want" color)
case "$TERM" in
    xterm-color|*-256color) color_prompt=yes;;
esac

# uncomment for a colored prompt, if the terminal has the capability; turned
# off by default to not distract the user: the focus in a terminal window
# should be on the output of commands, not on the prompt
#force_color_prompt=yes

if [ -n "$force_color_prompt" ]; then
    if [ -x /usr/bin/tput ] && tput setaf 1 >&/dev/null; then
	# We have color support; assume it's compliant with Ecma-48
	# (ISO/IEC-6429). (Lack of such support is extremely rare, and such
	# a case would tend to support setf rather than setaf.)
	color_prompt=yes
    else
	color_prompt=
    fi
fi

if [ "$color_prompt" = yes ]; then
    PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '
else
    PS1='${debian_chroot:+($debian_chroot)}\u@\h:\w\$ '
fi
unset color_prompt force_color_prompt

# If this is an xterm set the title to user@host:dir
case "$TERM" in
xterm*|rxvt*)
    PS1="\[\e]0;${debian_chroot:+($debian_chroot)}\u@\h: \w\a\]$PS1"
    ;;
*)
    ;;
esac

# enable color support of ls and also add handy aliases
if [ -x /usr/bin/dircolors ]; then
    test -r ~/.dircolors && eval "$(dircolors -b ~/.dircolors)" || eval "$(dircolors -b)"
    alias ls='ls --color=auto'
    #alias dir='dir --color=auto'
    #alias vdir='vdir --color=auto'

    alias grep='grep --color=auto'
    alias fgrep='fgrep --color=auto'
    alias egrep='egrep --color=auto'
fi

# colored GCC warnings and errors
#export GCC_COLORS='error=01;31:warning=01;35:note=01;36:caret=01;32:locus=01:quote=01'

# some more ls aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'

# Add an "alert" alias for long running commands.  Use like so:
#   sleep 10; alert
alias alert='notify-send --urgency=low -i "$([ $? = 0 ] && echo terminal || echo error)" "$(history|tail -n1|sed -e '\''s/^\s*[0-9]\+\s*//;s/[;&|]\s*alert$//'\'')"'

# Alias definitions.
# You may want to put all your additions into a separate file like
# ~/.bash_aliases, instead of adding them here directly.
# See /usr/share/doc/bash-doc/examples in the bash-doc package.

if [ -f ~/.bash_aliases ]; then
    . ~/.bash_aliases
fi

# enable programmable completion features (you don't need to enable
# this, if it's already enabled in /etc/bash.bashrc and /etc/profile
# sources /etc/bash.bashrc).
if ! shopt -oq posix; then
  if [ -f /usr/share/bash-completion/bash_completion ]; then
    . /usr/share/bash-completion/bash_completion
  elif [ -f /etc/bash_completion ]; then
    . /etc/bash_completion
  fi
fi

# vrep
export VREP_ROOT="$HOME/V-REP_PRO_EDU_V3_4_0_Linux"

# added by Anaconda3 4.4.0 installer
# PYTHON
export PATH="/home/chris/anaconda3/bin:$PATH"
#export PATH="/home/chris/anaconda3/envs/charmconda/bin/:$PATH"


# cuda for gpu-tensorflow
#Nvidia
#export CUDA_HOME=/usr/local/cuda-8.0
#export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64
#export PATH=${PATH}:${CUDA_HOME}/bin

# gym
export GYM_PATH=~/gym/python3_ws:$GYM_PATH

# ros
#source /opt/ros/kinetic/setup.bash

# gazebo-ros
export GAZEBO_MODEL_PATH=/home/chris/gym-gazebo/gym_gazebo/envs/installation/../assets/models
export GYM_GAZEBO_WORLD_MAZE=/home/chris/gym-gazebo/gym_gazebo/envs/installation/../assets/worlds/maze.world
export GYM_GAZEBO_WORLD_CIRCUIT=/home/chris/gym-gazebo/gym_gazebo/envs/installation/../assets/worlds/circuit.world
export GYM_GAZEBO_WORLD_CIRCUIT2=/home/chris/gym-gazebo/gym_gazebo/envs/installation/../assets/worlds/circuit2.world
export GYM_GAZEBO_WORLD_CIRCUIT2C=/home/chris/gym-gazebo/gym_gazebo/envs/installation/../assets/worlds/circuit2c.world
export GYM_GAZEBO_WORLD_ROUND=/home/chris/gym-gazebo/gym_gazebo/envs/installation/../assets/worlds/round.world

# ros
#source /home/chris/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/devel/setup.bash
#export GAZEBO_MODEL_PATH=~/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/src:$GAZEBO_MODEL_PATH
#export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/home/chris/gym-gazebo/gym_gazebo/envs/assets/models

# roboschool
export ROBOSCHOOL_PATH=/home/chris/python3_ws/roboschool
export PYTHONPATH="$ROBOSCHOOL_PATH/roboschool:$PYTHONPATH"

export PKG_CONFIG_PATH="/home/chris/anaconda3/lib/pkgconfig:$PKG_CONFIG_PATH"

# gym-mujoco-planar-snake?
export GYM_MUJOCO_PATH=/home/chris/python3_ws/gym-mujoco-planar-snake
export PYTHONPATH="$GYM_MUJOCO_PATH/gym-mujoco-planar-snake:$PYTHONPATH"

# gym_gazebo_projects
#export GYM_GAZEBO_PROJECTS_PATH=/home/chris/python3_ws/gym_gazebo_projects
#export PYTHONPATH="$GYM_GAZEBO_PROJECTS_PATH/envs:$PYTHONPATH"

# roboschool_projects
# python3_ws
#export ROBOSCHOOL_PROJECTS_PATH=/home/chris/python3_ws/roboschool_projects

# baseline
export OPENAI_LOGDIR=/home/chris/openai_logdir/tensorboard/x_new
export OPENAI_LOG_FORMAT="stdout,log,csv,json,tensorboard"
export PYTHONPATH="/home/chris/python3_ws/baselines:$PYTHONPATH"

# mujoco-py 1.50.1
#export PYTHONPATH="/home/chris/python3_ws/mujoco-py:$PYTHONPATH"
#export PYTHONPATH="/home/chris/python3_ws/mujoco-py/mujoco_py:$PYTHONPATH"

# all?
#export PYTHONPATH="/home/chris/python3_ws:$PYTHONPATH"

# MUJOCO
#export MUJOCO_PY_MJKEY_PATH=/bin/mjkey.txt
#export MUJOCO_PY_MJPRO_PATH=/home/chris/python3_ws/mujoco-py-1.50.1.0

# TODO
export MUJOCO_PY_MJPRO_PATH=/home/chris/.mujoco/mjpro150
#export MUJOCO_PY_MJPRO_PATH=/home/chris/.mujoco/mjpro131

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/chris/.mujoco/mjpro150/bin
#LD_LIBRARY_PATH=~/.mujoco/mjpro150/bin pip install mujoco-py
#LD_LIBRARY_PATH=$HOME/.mujoco/mjpro150/bin pip install mujoco-py
#LD_LIBRARY_PATH=$home/chris/python3_ws/mujoco-py-1.50.1.0/bin pip install mujoco-py
#export MUJOCO_PY_FORCE_CPU="True"
#LD_LIBRARY_PATH=$HOME/.mujoco/mjpro150/bin pip install mujoco-py
#LD_LIBRARY_PATH=$HOME/.mujoco/mjpro150/bin pip3 install -U 'mujoco-py<1.50.2,>=1.50.1'

#rllab
export PYTHONPATH="/home/chris/python3_ws/rllab:$PYTHONPATH"

#dm-control
export PYTHONPATH="/home/chris/python3_ws/dm-control:$PYTHONPATH"

# GODLIKE FIX ... GLEW init error
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-384/libGL.so
# X Error of failed request:  BadAccess (attempt to access private resource denied)
export QT_X11_NO_MITSHM=1

