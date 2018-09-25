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
        x0 = self.env.reset()
        x0 = processing.process_raw_state(x0)
        s0 = [x0, x0, x0, x0]

        # store episode transitions
        episode_s0 = []
        episode_a = []
        episode_s1 = []
        episode_r = []

        # run the game for an episode
        done = False
        while not done:

            # choose an action
            a = self.network.choose_action(
                    processing.process_state_stacks([s0]))

            # execute action
            x1, r, done, info = self.env.step(a)
            x1 = processing.process_raw_state(x1)
            s1 = [x1, s0[0], s0[1], s0[2]]

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
        s0 = processing.process_state_stacks(s0)
        s1 = processing.process_state_stacks(s1)
        r_discounted = processing.process_discounted_rewards(r)
        self.network.store_transitions(s0, a, s1, r_discounted)
