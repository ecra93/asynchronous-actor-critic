"""
Simple script, demonstrats state->action->state,reward loop for OpenAI's
MsPacman.
"""

import gym

env = gym.make("MsPacman-v0")

# restart the game at initial state
x0 = env.reset()
s0 = (x0, x0, x0, x0)

# run the game for an episode
done = False
while not done:

    # choose an action - the action is randomly chosen in this case, but
    # would be passed through some choose_action() function held by an
    # agent
    a = env.action_space.sample()

    # execute the action, causing a transition to the next state (s1) 
    # and receipt of  reward (r1)
    s1 = []

    # we want to run an action for N timesteps to give the renderer a
    # sense of the trajectory of moving objects
    for i in range(4):
        x1, r1, done, info = env.step(a)
        s1.append(x1)
    s1 = tuple(s1)
    t = (s0, a, s1, r1, done)

    # render the state change in the emulator - this is actually optional;
    # you only have to do this if you want to see the game being played
    # on screen
    env.render()

# gracefully shutdown the emulator
env.close()
