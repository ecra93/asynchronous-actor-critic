from threading import Thread
import time
import gym

import processing

class Worker(Thread):

    def __init__(self, game, network, n_episodes):
        Thread.__init__(self)
        self.env = gym.make(game)
        self.network = network
        self.n_episodes = n_episodes
        self.is_main = False

    def play_episode(self):

        # reset environment for each iteration
        s0 = self.env.reset()

        # store episode transitions
        episode_s0 = []
        episode_a = []
        episode_s1 = []
        episode_r = []

        # run the game for an episode
        done = False
        while not done:

            # choose an action
            a = self.network.choose_action([s0])

            # execute action
            s1, r, done, info = self.env.step(a)

            # store transition
            episode_s0.append(s0)
            episode_a.append(a)
            episode_s1.append(s1)
            episode_r.append(r)

            # display action on screen
            if self.is_main:
                self.env.render()

            # update initial state
            s0 = s1

        # send transitions over to the network
        self.send_transitions_to_network(episode_s0, episode_a,
                episode_s1, episode_r)

    def run(self):
        # play episode n times
        for _ in range(self.n_episodes):
            self.play_episode()
        
        # gracefully shutdown environment
        self.env.close()

    def send_transitions_to_network(self, s0, a, s1, r):
        r_n_step = processing.process_n_step_rewards(r)
        s_n_step = processing.process_n_step_states(s0)
        self.network.store_transitions(s0, a, s_n_step, r_n_step)
