import matplotlib

matplotlib.use('TkAgg')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = tf.placeholder('float')
y = tf.placeholder('float')

w = tf.Variable([1.0, 2.0], name='w')

y_model = tf.mul(x, w[0]) + w[1]

error = tf.square(y - y_model)

train_op = tf.train.GradientDescentOptimizer(0.01).minimize(error)

model = tf.global_variables_initializer()

errors = []
with tf.Session() as session:
    session.run(model)
    for i in range(1000):
        x_train = tf.random_normal((1,), mean=5, stddev=2.0)
        y_train = x_train * 2 + 6
        x_value, y_value = session.run([x_train, y_train])
        _, error_value = session.run([train_op, error], feed_dict={x: x_value, y: y_value})
        errors.append(error_value)
        # x_value = np.random.rand()
        # y_value = x_value * 2 + 6
        # session.run(train_op, feed_dict={x: x_value, y: y_value})

    w_value = session.run(w)
    print('Predicted model: {a:.3f}x + {b:.3f}'.format(a=w_value[0], b=w_value[1]))

plt.plot([np.mean(errors[i - 50:i]) for i in range(len(errors))])
plt.show()
