import numpy as np
import cv2


def process_n_step_rewards(r, discount_factor=0.95, n_steps=8):
    """
    A3C
    Converts the sequence of raw rewards (r) into discounted n-step
    rewards.

        V(si) = ri + GAMMA*ri+1 * GAMMA^2*ri_2 ... + GAMMA^nV(si+n)

    """
    # to store n_step reward
    r_n_step = np.zeros_like(r)

    # compute n_step rewards
    for i in range(len(r)):
        accum = r[i]
        for j in reversed(range(1,n_steps+1)):
            t = i + j
            if t >= len(r):
                continue
            else:
                accum = accum * discount_factor + r[t]
        r_n_step[i] = accum

    return r_n_step


def process_n_step_states(s, n_steps=8):

    # to store the n_step states
    s_n_step = np.zeros_like(s)

    # compute the nth state forward
    for i in range(len(s)):
        t = i + n_steps

        # if there aren't n states left, then use preceeding state
        while t >= len(s):
            t = t - 1

        s_n_step[i] = s[t]

    return s_n_step



def process_discounted_rewards(r, discount_factor=0.95):
    """
    Policy Gradient
    Backwards sums and discounts episode rewards - assumes that the last
    reward in the array pertains to the final timestep.
    """
    # accumulate and discount r
    r_discounted = np.zeros_like(r)
    accum = 0.0
    for i in reversed(range(len(r))):
        accum = accum * discount_factor + r[i]
        r_discounted[i] = accum

    # standardise discounted r
    r_discounted -= np.mean(r_discounted)
    r_discounted /= np.std(r_discounted)

    return r_discounted
