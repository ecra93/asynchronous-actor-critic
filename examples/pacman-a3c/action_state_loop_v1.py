"""
Simple script, demonstrats state->action->state,reward loop for OpenAI's
MsPacman.
"""

import gym

env = gym.make("MsPacman-v0")

# restart the game at initial state
s0 = env.reset()

# run the game for an episode
done = False
while not done:

    # choose an action - the action is randomly chosen in this case, but
    # would be passed through some choose_action() function held by an
    # agent
    a = env.action_space.sample()

    # execute the action, causing a transition to the next state (s1) 
    # and receipt of  reward (r1)
    s1, r1, done, info = env.step(a)

    # render the state change in the emulator - this is actually optional;
    # you only have to do this if you want to see the game being played
    # on screen
    env.render()

# gracefully shutdown the emulator
env.close()
