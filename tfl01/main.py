import numpy as np
import tensorflow as tf

data = np.random.randint(1000, size=10000)

x = tf.constant(data, name='x')
y = tf.Variable(x+5, name='y')

with tf.Session() as session:
    model = tf.global_variables_initializer()
    session.run(model)
    file_writer = tf.summary.FileWriter('logs', session.graph)
    print(session.run(y))