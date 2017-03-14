import numpy as np
import os
import tensorflow as tf
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

filename = os.path.join('tfl02', 'MarshOrchid.jpg')
image = mpimg.imread(filename)
height, width, depth = image.shape

x = tf.Variable(image, name='x')

model = tf.initialize_all_variables()

with tf.Session() as session:
    # x = tf.transpose(x, perm=[1, 0, 2])
    x = tf.reverse_sequence(x, [width]*height, 1, batch_dim=0)
    session.run(model)
    result = session.run(x)


plt.imshow(result)
plt.show()
