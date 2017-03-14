import matplotlib

matplotlib.use('TkAgg')

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib import animation as animation
from scipy.signal import convolve2d


def update_board(X):
    N = convolve2d(X, np.ones((3, 3)), mode='same', boundary='wrap') - X
    X = (N == 3) | (X & N == 2)
    return X


shape = (50, 50)
initial_board = tf.random_uniform(shape, minval=0, maxval=2, dtype=tf.int32)

board = tf.placeholder(tf.int32, shape=shape, name='board')
board_update = tf.py_func(update_board, [board], [tf.int32])

with tf.Session() as session:
    initial_board_values = session.run(initial_board)

    # plot = plt.imshow(X, cmap='Greys', interpolation='nearest')

    def game_of_life(*args):
        X = session.run(board_update, feed_dict={board: args[0]})[0]
        plot.set_array(X)
        return plot, X


    plot, X = game_of_life(initial_board_values)
    # X = session.run(board_update, feed_dict={board: initial_board_values})[0]


    fig = plt.figure()
    ani = animation.FuncAnimation(fig, game_of_life, interval=200, blit=True)
    plt.show()

# plt.show()
