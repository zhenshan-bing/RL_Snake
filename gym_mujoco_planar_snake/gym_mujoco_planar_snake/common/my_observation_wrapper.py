from gym.core import ObservationWrapper


class MyObservationWrapper(ObservationWrapper):

    def __init__(self, env, reset_listener):
        ObservationWrapper.__init__(self, env=env)

        self.reset_listener = reset_listener

        self.last_e_steps = 0
        self.episodes = 1
        self.observations_list = []

    def _observation(self, observation):
        self.observations_list.append(observation)

        return observation

    def _step(self, action):
        self.last_e_steps += 1
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    def _reset(self, **kwargs):
        #res = self._observation(observation)

        # TODO
        self.observations_list = self.reset_listener(self.observations_list)

        # first reset or empty list
        if len(self.observations_list) == 0:
            self.env.env.metadata['gen_init_state'] = True

            # Do reset and append to observations_list
            init =self.env.reset(**kwargs)

            # Called otherhow called
            #if self.last_e_steps == 1:
            #    return init

            self._observation(init)

            self.env.env.metadata['gen_init_state'] = False

        # get observation
        next_init_state = self.observations_list[-1]

        self.env.env.metadata['new_init0'], self.env.env.metadata['new_init1'] = next_init_state

        #
        print('--------reset---------')
        print('episode: ', self.episodes)
        print('last episode steps: ', self.last_e_steps)
        print('new observations_list_size: ', len(self.observations_list))
        print('next_init_state: ', next_init_state)

        # TODO handle max timesteps
        # max_timesteps = max_timesteps - len(observations_list)

        # TODO
        #self.env.render()


        self.last_e_steps = 0
        self.episodes += 1

        return next_init_state