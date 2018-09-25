import numpy as np
import cv2


def process_raw_state(state):
    """
    Processes a single MsPacman state screenshot.
    """

    # crop out bottom of the pacman bar
    state = state[1:171, :]

    # convert to grayscale, then to binary
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = cv2.threshold(state, 100, 255, cv2.THRESH_BINARY)[1]

    # resize to 84,84
    state = cv2.resize(state, (84,84))

    return state


def process_state_stack(s):
    return np.transpose(s, axes=[0,2,3,1])


def process_state_stacks(s):
    """
    Swaps an array of states, each of shape (4, 84, 84, 1), into an array
    of states, each of shape (84, 84, 4)
    """
    # pass s into np array
    s = np.array(s)

    # swap from (?,4,84,84,1) to (?,1,84,84,4)
    s = np.transpose(s, axes=[0,2,3,1])

    return s


def process_discounted_rewards(r, discount_factor=0.95):
    """
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
