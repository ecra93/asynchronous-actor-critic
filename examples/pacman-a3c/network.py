import tensorflow as tf
import numpy as np
import time

import threading
from threading import Thread

WEIGHT_VALUE_LOSS = 0.5
WEIGHT_ENTROPY_LOSS = 0.01


class Network:
    """
    Global network.
    """
    def __init__(self, n_actions, sess):
        self.n_actions = n_actions
        self.episodes = []
        self.lock = threading.Lock()

        self.sess = sess
        self.initialize_network()
        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.5)
        self.load_network()
        self.sess.run(tf.global_variables_initializer())


    def initialize_network(self):

        with tf.name_scope("state-input"):
            self.X = tf.placeholder(tf.float32, shape=[None,84,84,4])

        with tf.name_scope("conv_layer-1") as scope:
            k1 = tf.get_variable("k1", shape=[8,8,4,16])
            b1 = tf.get_variable("b1", shape=[16])
            c1 = tf.nn.conv2d(self.X, k1, strides=[1,1,1,1], padding="SAME")
            conv_with_bias1 = tf.add(c1, b1)
            conv1 = tf.nn.relu(conv_with_bias1)
            pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1],
                        strides=[1,2,2,1], padding="SAME")   

        with tf.name_scope("conv-layer-2") as scope:
            k2 = tf.get_variable("k2", shape=[4,4,16,32])
            b2 = tf.get_variable("b2", shape=[32])
            c2 = tf.nn.conv2d(pool1, k2, strides=[1,1,1,1], padding="SAME")
            conv_with_bias2 = tf.add(c2, b2)
            conv2 = tf.nn.relu(conv_with_bias2)
            pool2 = tf.nn.max_pool(conv2, ksize=[1,3,3,1],
                        strides=[1,2,2,1], padding="SAME")

        with tf.name_scope("fc-layer-1"):
            x_flat = tf.reshape(pool2, shape=[-1,21*21*32])
            w3 = tf.get_variable(name="w3", shape=[21*21*32, 100])
            b3 = tf.get_variable(name="b3", shape=[100])
            fc1 = tf.nn.relu(tf.matmul(x_flat, w3) + b3)

        with tf.name_scope("policy-output"):
            w4 = tf.get_variable(name="w4", shape=[100, self.n_actions])
            b4 = tf.get_variable(name="b4", shape=[self.n_actions])
            y_ = tf.matmul(fc1, w4) + b4
            self.policy = tf.nn.softmax(y_)

        with tf.name_scope("value"):
            Wv = tf.get_variable(name="Wv", shape=[100,1])
            bv = tf.get_variable(name="bv", shape=[1])
            value = tf.matmul(fc1, Wv) + bv

        with tf.name_scope("policy-loss"):
            self.a = tf.placeholder(tf.int32, shape=[None,])
            a_one_hot = tf.one_hot(self.a, depth=self.n_actions)
            self.r = tf.placeholder(tf.float32, shape=[None,])
            log_ap = tf.log(tf.reduce_sum(self.policy * a_one_hot,
                axis=1, keepdims=True) + 1e-10)
            advantage = self.r - value
            loss_p = -log_ap * tf.stop_gradient(advantage)

        with tf.name_scope("value-loss"):
            loss_v = WEIGHT_VALUE_LOSS * tf.square(advantage)

        with tf.name_scope("entropy-loss"):
            loss_e = WEIGHT_ENTROPY_LOSS * tf.reduce_sum(self.policy *\
                tf.log(self.policy + 1e-10), axis=1, keepdims=True)

        with tf.name_scope("total-loss"):
            self.loss_t = tf.reduce_mean(loss_p + loss_v + loss_e)

        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.RMSPropOptimizer(
                    learning_rate=0.01,decay=0.99).minimize(self.loss_t)


    def train_network(self):
        # if no training data, then skip
        if not self.episodes:
            return

        # grab the most recent episode
        self.lock.acquire()
        episode = self.episodes.pop()
        self.lock.release()

        s0 = episode[0]
        a = episode[1]
        s1 = episode[2]
        r = episode[3]

        # train network here
        loss, _ = self.sess.run([self.loss_t, self.optimizer], feed_dict={
            self.X: s0,
            self.a: a,
            self.r: r
        })

        # logging messages
        print("========================================================")
        print("Training Episode Complete")
        print("Episode Reward: " + str(r[0]))
        print("Episode Loss: " + str(loss))
        print("========================================================")

        # save network
        self.save_network()


    def choose_action(self, state):
        policy = self.sess.run(self.policy, feed_dict={self.X:state})[0]
        action = np.random.choice(len(policy), p=policy)
        return action


    def store_transitions(self, s0, a, s1, r_discounted):
        self.lock.acquire()
        self.episodes.append((s0, a, s1, r_discounted))
        self.lock.release()

    def load_network(self):
        ckpt = tf.train.latest_checkpoint(checkpoint_dir="./saved-networks")
        if not (ckpt is None):
            self.saver.restore(self.sess, ckpt)

    def save_network(self):
        self.saver.save(self.sess, "saved-networks/pacman")


class Optimizer(Thread):

    def __init__(self, network):
        super(Optimizer, self).__init__()
        self.network = network
        self.stop = False

    def run(self):
        while (not self.stop):
            time.sleep(0)
            self.network.train_network()
