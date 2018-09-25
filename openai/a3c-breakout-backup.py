"""
Implementation of A3C Algorithm in TensorFlow for OpenAI CartPole.
"""

import numpy as np
import tensorflow as tf

from threading import Thread, Lock
import time, gym, random, cv2

ENVIRONMENT = "Breakout-ram-v0"
SAVE_DIR = "saved-networks-" + ENVIRONMENT
SAVE_FILE = "network"
SAVE_FREQ = 1000

tmp = gym.make(ENVIRONMENT)
NUM_STATES = tmp.observation_space.shape[0]
print(NUM_STATES)
NUM_ACTIONS = tmp.action_space.n
NONE_STATE = np.zeros(NUM_STATES)

NUM_DRONES = 2
NUM_OPTIMIZERS = 1
THREAD_DELAY = 0.001

GAMMA = 0.99
N_STEP = 8
GAMMA_N = GAMMA ** N_STEP

EPSILON_INIT = 0.40
EPSILON_STOP = 0.15
EPSILON_STEP = 0.001

MIN_TRAINING_BATCH = 32
LEARNING_RATE = 5e-3

COEF_LOSS_V = 0.5
COEF_LOSS_ENT = 0.01


"""
TensorFlow helper functions.
"""
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.01, shape=shape))

def conv_2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1,stride,stride,1], padding="SAME")

def max_pool(x, stride):
    return tf.nn.max_pool(x, ksize=[1,stride,stride,1],
        strides=[1,stride,stride,1], padding="SAME")


"""
Preprocessing.
"""
def preprocess(state):
    """
    Preprocess the screenshot of the state into an 80x80x1 binary matrix.
    """
    state = cv2.resize(state, (84, 110))
    state = state[22:102, 2:82] # crop to 80 x 80
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    _, state = cv2.threshold(state, 1, 255, cv2.THRESH_BINARY)
    state = np.reshape(state, (80,80,1))
    #cv2.imshow("img",state)
    return state


"""
A3C Implementation.
"""
class Master(object):
    """
    Implementation of the network used to approximate the policy and value
    functions.
    """
    def __init__(self):
        self.training_queue = [ [],[],[],[],[] ]
        self.lock = Lock()

        self.sess = tf.Session()
        self.initialize_architecture()
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        self.load_network()

    def initialize_architecture(self):
        with tf.name_scope("input-layer"):
            self.s = tf.placeholder(tf.float32, shape=[None,NUM_STATES])
            self.a = tf.placeholder(tf.float32, shape=[None,NUM_ACTIONS])
            self.r = tf.placeholder(tf.float32, shape=[None,1])

        """
        with tf.name_scope("conv-layer-1"):
            self.conv1_W = weight_variable([8,8,1,32])
            self.conv1_b = bias_variable([32])
            self.conv1 = conv_2d(self.s, self.conv1_W, 4) + self.conv1_b
            self.conv1 = tf.nn.relu(self.conv1)
            self.conv1 = max_pool(self.conv1, 4)

        with tf.name_scope("conv-layer-2"):
            self.conv2_W = weight_variable([4,4,32,64])
            self.conv2_b = bias_variable([64])
            self.conv2 = conv_2d(self.conv1, self.conv2_W, 2) + self.conv2_b
            self.conv2 = tf.nn.relu(self.conv2)
            self.conv2 = max_pool(self.conv2, 2)
        """

        with tf.name_scope("flat-layer-1"):
            self.flat_W = weight_variable([NUM_STATES, 512])
            self.flat_b = bias_variable([512])
            self.flat = tf.matmul(self.s, self.flat_W) + self.flat_b
            self.flat = tf.nn.relu(self.flat)

        with tf.name_scope("flat-layer-2"):
            self.flat2_W = weight_variable([512, 512])
            self.flat2_b = bias_variable([512])
            self.flat2 = tf.matmul(self.flat, self.flat2_W) + self.flat2_b
            self.flat2 = tf.nn.relu(self.flat2)

        with tf.name_scope("policy"):
            self.policy_W = weight_variable([512,NUM_ACTIONS])
            self.policy_b = bias_variable([NUM_ACTIONS])
            self.policy = tf.matmul(self.flat2, self.policy_W) + self.policy_b
            self.policy = tf.nn.softmax(self.policy)

        with tf.name_scope("value"):
            self.value_W = weight_variable([512,1])
            self.value_b = weight_variable([1])
            self.value = tf.matmul(self.flat, self.value_W) + self.value_b

        with tf.name_scope("policy-loss"):
            self.log_ap = tf.log(tf.reduce_sum(self.policy*self.a, axis=1,
                                               keepdims=True) + 1e-10)
            self.advantage = self.r - self.value
            self.loss_p = - self.log_ap * tf.stop_gradient(self.advantage)

        with tf.name_scope("value-loss"):
            self.loss_v = COEF_LOSS_V * tf.square(self.advantage)

        with tf.name_scope("entropy-loss"):
            self.loss_e = COEF_LOSS_ENT * tf.reduce_sum(self.policy *
                tf.log(self.policy + 1e-10), axis=1, keepdims=True)

        with tf.name_scope("total-loss"):
            self.loss_t = tf.reduce_mean(self.loss_p + self.loss_v +
                self.loss_e)

        with tf.name_scope("global-step"):
            self.global_step = tf.Variable(0, name="step", trainable=False)

        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE,
                decay=0.99)
            self.minimize = self.optimizer.minimize(self.loss_t,
                global_step=self.global_step)

    def add_to_queue(self, s, a, r, s_):
        # add samples to the training queue
        with self.lock:
            self.training_queue[0].append(s)
            self.training_queue[1].append(a)
            self.training_queue[2].append(r)
            if s_ is None:
                self.training_queue[3].append(NONE_STATE)
                self.training_queue[4].append(0.0)
            else:
                self.training_queue[3].append(s_)
                self.training_queue[4].append(1.0)

    def train(self):
        if len(self.training_queue[0]) < MIN_TRAINING_BATCH:
            time.sleep(0) # yield
            return

        with self.lock:
            # extract s, a, r, s_, mask from training queue
            s, a, r, s_, mask = self.training_queue
            self.training_queue = [ [],[],[],[],[] ]

        s = np.vstack(s)
        a = np.vstack(a)
        r = np.vstack(r)
        s_ = np.vstack(s_)
        mask = np.vstack(mask)

        # generate value predictions, 
        v = self.sess.run(self.value, feed_dict={self.s : s_})
        r = r + GAMMA_N * v * mask

        # train
        self.sess.run(self.minimize,
            feed_dict={self.s : s, self.a : a, self.r : r})

        # save if global step large enough
        if self.sess.run(self.global_step) % SAVE_FREQ == 0:
            self.save_network()

    def load_network(self):
        checkpoint = tf.train.latest_checkpoint(
            checkpoint_dir=SAVE_DIR)
        if not (checkpoint is None):
            print("Existing network found at " + SAVE_DIR + ". Loading ...")
            self.saver.restore(self.sess, checkpoint)
            print("... loaded.")

    def save_network(self):
        print("Saving network ...")
        self.saver.save(self.sess, SAVE_DIR + "/" + SAVE_FILE,
            global_step=self.global_step)
        print("... saved.")

    def predict_policy(self, s):
        # predict the policy distribution of a given state
        policy = self.sess.run([self.policy], feed_dict={self.s : [s]})
        policy = np.squeeze(policy, axis=1)
        return policy

    def predict_value(self, s):
        # predict the value of a given state
        value = self.sess.run([self.value], feed_dict={self.s : [s]})
        value = np.squeeze(value, axis=1)
        return value


class Optimizer(Thread):
    """
    Single thread, runs the train method for the network on Master at
    regular intervals.
    """
    def __init__(self, master):
        super(Optimizer, self).__init__()
        self.master = master
    
    def run(self):
        while True:
            self.master.train()


class Drone(Thread):
    """
    Drone thread, encapsulates an environment / agent. Collects training
    samples from the environment, and passes them to Master.
    """
    def __init__(self, master, exemplar=False):
        super(Drone, self).__init__()
        self.master = master
        self.env = gym.make(ENVIRONMENT)
        self.epsilon = EPSILON_INIT
        self.memory = []
        self.R = 0.0
        self.sess = self.master.sess
        self.exemplar = exemplar

    def run(self):
        while True:
            self.run_episode()

    def run_episode(self):
        s = self.env.reset()
        R = 0
        while True:
            time.sleep(THREAD_DELAY) # yield
            if self.exemplar:
                self.env.render()

            # act and collect state transition
            a = self.act(s)
            s_, r, end, _ = self.env.step(a)
            if end:
                s_ = None
            a_onehot = np.zeros(NUM_ACTIONS)
            a_onehot[a] = 1

            transition = (s, a_onehot, r, s_)
            self.memory.append(transition)

            self.R = (self.R + r * GAMMA_N) / GAMMA

            if s_ is None:
                while len(self.memory) > 0:
                    n = len(self.memory)
                    s, a, r, s_ = self.sample_memory(n)
                    master.add_to_queue(s, a, r, s_)
                    self.R = (self.R - self.memory[0][2]) / GAMMA
                    self.memory.pop(0)
                self.R = 0

            if len(self.memory) >= N_STEP:
                s, a, r, s_ = self.sample_memory(N_STEP)
                master.add_to_queue(s, a, r, s_)
                self.R = self.R - self.memory[0][2]
                self.memory.pop(0)

            s = s_
            R += r

            if end:
                break
        print ("Total Discounted Reward = " +  str(R))

    def sample_memory(self, n):
        s, a, _, _ = self.memory[0]
        _, _, _, s_ = self.memory[n-1]
        return s, a, self.R, s_

    def act(self, s):
        if random.random() < self.epsilon:
            if self.epsilon > EPSILON_INIT:
                self.epsilon -= EPSILON_STEP
            return random.randint(0, NUM_ACTIONS-1)
        else:
            policy = self.master.predict_policy(s)
            return np.random.choice(NUM_ACTIONS, p=policy)


if __name__ == "__main__":

    master = Master()
    #Drone(master).run()

    workers = [Drone(master) for i in range(NUM_DRONES-1)]
    optimizers = [Optimizer(master) for i in range(NUM_OPTIMIZERS)]

    for worker in workers:
        worker.start()

    for optimizer in optimizers:
        optimizer.start()

    Drone(master, exemplar=True).run()
