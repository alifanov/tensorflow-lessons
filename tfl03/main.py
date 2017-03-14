import matplotlib

matplotlib.use('TkAgg')

import os
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

filename = os.path.join('tfl03', 'MarshOrchid.jpg')
raw_image_data = mpimg.imread(filename)

image = tf.placeholder("uint8", [None, None, 3])
slice = tf.slice(image, [1000, 0, 0], [3000, -1, -1])

with tf.Session() as session:
    result = session.run(slice, feed_dict={image: raw_image_data})

plt.imshow(result)
plt.show()
