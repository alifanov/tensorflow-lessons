import tensorflow as tf
import numpy as np

cluster = tf.train.ClusterSpec({'local': ['localhost:2222', 'localhost:2223']})
# server = tf.train.Server(cluster, job_name='local', task_index=task_number)

# print('Serving server #{}'.format(task_number))
#
# server.start()
# server.join()

x = tf.placeholder(tf.float32, 100)

with tf.device('job:local/task:0'):
    first_batch = tf.slice(x, [0], [50])
    mean1 = tf.reduce_mean(first_batch)

with tf.device('job:local/task:1'):
    second_batch = tf.slice(x, [50], [-1])
    mean2 = tf.reduce_mean(second_batch)
    mean = (mean1 + mean2) / 2

with tf.Session('grpc://localhost:2222') as session:
    result = session.run(mean, feed_dict={x: np.random.random(100)})
    print(result)
