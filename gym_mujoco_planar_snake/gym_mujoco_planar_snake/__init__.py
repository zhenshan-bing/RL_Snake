from gym.envs.registration import register

# rl control
register(
    id='Mujoco-planar-snake-cars-angle-wave-v1',
    entry_point='gym_mujoco_planar_snake.envs.mujoco_15:MujocoPlanarSnakeCarsAngleWaveEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='Mujoco-planar-snake-cars-cam-wave-v1',
    entry_point='gym_mujoco_planar_snake.envs.mujoco_15:MujocoPlanarSnakeCarsCamWaveEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='Mujoco-planar-snake-cars-cam-dist-wave-v1',
    entry_point='gym_mujoco_planar_snake.envs.mujoco_15:MujocoPlanarSnakeCarsCamWaveDistanceEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)


#
register(
    id='Mujoco-planar-snake-cars-angle-line-v1',
    entry_point='gym_mujoco_planar_snake.envs.mujoco_15:MujocoPlanarSnakeCarsAngleLineEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

# Bing Create this for testing 
register(
    id='Mujoco-planar-snake-cars-angle-v1',
    entry_point='gym_mujoco_planar_snake.envs.mujoco_15:MujocoPlanarSnakeCarsAngleLineEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='Mujoco-planar-snake-cars-cam-line-v1',
    entry_point='gym_mujoco_planar_snake.envs.mujoco_15:MujocoPlanarSnakeCarsCamLineEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='Mujoco-planar-snake-cars-cam-dist-line-v1',
    entry_point='gym_mujoco_planar_snake.envs.mujoco_15:MujocoPlanarSnakeCarsCamLineDistanceEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)



#
register(
    id='Mujoco-planar-snake-cars-angle-zigzag-v1',
    entry_point='gym_mujoco_planar_snake.envs.mujoco_15:MujocoPlanarSnakeCarsAngleZigzagEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='Mujoco-planar-snake-cars-cam-zigzag-v1',
    entry_point='gym_mujoco_planar_snake.envs.mujoco_15:MujocoPlanarSnakeCarsCamZigzagEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='Mujoco-planar-snake-cars-cam-dist-zigzag-v1',
    entry_point='gym_mujoco_planar_snake.envs.mujoco_15:MujocoPlanarSnakeCarsCamZigzagDistanceEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)


#
register(
    id='Mujoco-planar-snake-cars-angle-circle-v1',
    entry_point='gym_mujoco_planar_snake.envs.mujoco_15:MujocoPlanarSnakeCarsAngleCircleEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='Mujoco-planar-snake-cars-cam-circle-v1',
    entry_point='gym_mujoco_planar_snake.envs.mujoco_15:MujocoPlanarSnakeCarsCamCircleEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='Mujoco-planar-snake-cars-cam-dist-circle-v1',
    entry_point='gym_mujoco_planar_snake.envs.mujoco_15:MujocoPlanarSnakeCarsCamCircleDistanceEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)


#
register(
    id='Mujoco-planar-snake-cars-angle-random-v1',
    entry_point='gym_mujoco_planar_snake.envs.mujoco_15:MujocoPlanarSnakeCarsAngleRandomEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='Mujoco-planar-snake-cars-cam-random-v1',
    entry_point='gym_mujoco_planar_snake.envs.mujoco_15:MujocoPlanarSnakeCarsCamRandomEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='Mujoco-planar-snake-cars-cam-dist-random-v1',
    entry_point='gym_mujoco_planar_snake.envs.mujoco_15:MujocoPlanarSnakeCarsCamRandomDistanceEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)


#
register(
    id='Mujoco-planar-snake-cars-angle-cursor-v1',
    entry_point='gym_mujoco_planar_snake.envs.mujoco_15:MujocoPlanarSnakeCarsAngleCursorEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='Mujoco-planar-snake-cars-cam-cursor-v1',
    entry_point='gym_mujoco_planar_snake.envs.mujoco_15:MujocoPlanarSnakeCarsCamCursorEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)
