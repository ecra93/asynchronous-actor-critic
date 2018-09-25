import tensorflow as tf
import time

from network import Network, Optimizer
from worker import Worker


if __name__ == "__main__":

    n_actions = 9
    n_episodes = 3
    n_workers = 6
    n_optimizers = 2
    game = "MsPacman-v0"

    with tf.Session() as sess:

        # create master
        print("Initializing network ...")
        network = Network(n_actions, sess)

        # create workers
        print("Initializing workers ...")
        workers = []
        for _ in range(n_workers):
            worker = Worker(game, network, n_episodes)
            workers.append(worker)

        # start each of the workers
        for i in range(1, n_workers):
            workers[i].start()

        # start an optimizer
        print ("Initializing optimizer ...")
        optimizers = []
        for i in range(n_optimizers):
            optimizer = Optimizer(network)
            optimizers.append(optimizer)
        
        # start each of the optimizers
        for i in range(n_optimizers):
            optimizers[i].start()

        # run worker[0] on main thread
        workers[0].is_main = True
        workers[0].run()

        # wait for the remaining workers to finish
        for i in range(1, n_workers):
            workers[i].join()
        print("... all workers closed.")

        # wait till all network episodes have been used for training
        while network.episodes:
            time.sleep(10)
        for i in range(n_optimizers):
            optimizers[i].stop = True

        # wait for the optimizer to wrap up
        for i in range(n_optimizers):
            optimizers[i].join()
        print("... all optimizers closed.")
