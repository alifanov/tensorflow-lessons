import numpy as np
import pandas as pd
import prettytensor as pt

import tensorflow as tf

VALIDATION_SIZE = 2000

data = pd.read_csv('train.csv')

images = data.iloc[:, 1:].values
images = images.astype(np.float)


def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


# For labels
labels_flat = data[[0]].values.ravel()
labels_count = np.unique(labels_flat).shape[0]
labels = dense_to_one_hot(labels_flat, labels_count)
labels = labels.astype(np.uint8)

# Normalize from [0:255] => [0.0:1.0]
images = np.multiply(images, 1.0 / 255.0)
image_size = images.shape[1]
image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)

validation_images = images[:VALIDATION_SIZE]
validation_labels = labels[:VALIDATION_SIZE]

train_images = images[VALIDATION_SIZE:]
train_labels = labels[VALIDATION_SIZE:]

img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_channels = 1
num_classes = 10
train_batch_size = 100


def random_batch(x_train, y_train):
    # Total number of images in the training-set.
    num_images = len(x_train)

    # Create a random index into the training-set.
    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)

    # Use the random index to select random images and labels.
    x_batch = x_train[idx, :]  # Images.
    y_batch = y_train[idx, :]  # Labels.

    # Return the batch.
    return x_batch, y_batch


x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

x_pretty = pt.wrap(x_image)

with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred, loss = x_pretty. \
        conv2d(kernel=5, depth=32, name='layer_conv1'). \
        max_pool(kernel=2, stride=2). \
        conv2d(kernel=5, depth=64, name='layer_conv2'). \
        max_pool(kernel=2, stride=2). \
        flatten(). \
        fully_connected(size=1024, name='layer_fc1'). \
        softmax_classifier(num_classes=num_classes, labels=y_true)

y_pred_cls = tf.argmax(y_pred, dimension=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred,
                                                        labels=y_true, name='cross_entropy')

cost = tf.reduce_mean(cross_entropy, name='cost')
tf.summary.scalar('cost', cost)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
tf.summary.scalar('accuracy', accuracy)


def optimize(X, Y, iterations=100):
    for i in range(iterations + 1):
        x_batch, y_batch = random_batch(X, Y)
        _, acc = session.run([optimizer, accuracy], feed_dict={x: x_batch, y_true: y_batch})
        if i % 50 == 0:
            print('Iteration: {0} | Validation accuracy: {1:.4%}'.format(i, acc))


BATCH_SIZE = 200

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    optimize(images, labels, iterations=200)

    test_images = pd.read_csv('test.csv').values
    test_images = test_images.astype(np.float)
    test_images = np.multiply(test_images, 1.0 / 255.0)

    predicted_labels = np.zeros(test_images.shape[0])
    for i in range(0, test_images.shape[0] // BATCH_SIZE):
        predicted_labels[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = y_pred_cls.eval(
            feed_dict={x: test_images[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]})
    # predicted_labels = session.run(y_pred_cls, feed_dict={x: test_images})

    np.savetxt('submission.csv',
               np.c_[range(1, len(test_images) + 1), predicted_labels],
               delimiter=',',
               header='ImageId,Label',
               comments='',
               fmt='%d')
