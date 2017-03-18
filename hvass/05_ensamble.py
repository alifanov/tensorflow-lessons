import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os

# number of CPU
NUM_THREADS = 4

# Convolutional Layer 1.
filter_size1 = 5  # Convolution filters are 5 x 5 pixels.
num_filters1 = 32  # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = 5  # Convolution filters are 5 x 5 pixels.
num_filters2 = 64  # There are 36 of these filters.

# Fully-connected layer.
fc_size = 128  # Number of neurons in fully-connected layer.

from tensorflow.examples.tutorials.mnist import input_data
import prettytensor as pt

data = input_data.read_data_sets("MNIST_data/", one_hot=True)

print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))

# We know that MNIST images are 28 pixels in each dimension.
img_size = 28

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1

# Number of classes, one class for each of 10 digits.
num_classes = 10

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

train_batch_size = 50

total_iterations = 0

test_batch_size = 100

merged = tf.summary.merge_all()


# saver = tf.train.Saver()
# save_dir = 'checkpoints'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# save_path = os.path.join(save_dir, 'best_validation')


def print_test_accuracy():
    # Number of images in the test-set.
    num_test = len(data.test.images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = data.test.images[i:j, :]

        # Get the associated labels.
        labels = data.test.labels[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y_true: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    # cls_true = data.test.cls
    cls_true = tf.argmax(data.test.labels, dimension=1)

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))


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


def optimize(session, num_iterations, writer=None, x_train=None, y_train=None):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(num_iterations):

        x_batch, y_true_batch = random_batch(x_train, y_train)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        # _, _, sm = session.run([cost, optimizer, merged], feed_dict=feed_dict_train)
        # tf.summary.scalar('cost', cost)
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations.
        if i % 100 == 0:
            # Calculate the accuracy on the training-set.
            acc = session.run(accuracy, feed_dict=feed_dict_train)
            # writer.add_summary(sm, i)
            # tf.summary.scalar('accuracy', acc)
            # saver.save(sess=session, save_path=save_path)

            print("Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}".format(i + 1, acc))


num_networks = 5
combined_images = np.concatenate([data.train.images, data.validation.images], axis=0)
combined_labels = np.concatenate([data.train.labels, data.validation.labels], axis=0)
combined_size = len(combined_images)
train_size = int(0.8 * combined_size)
validation_size = combined_size - train_size


def random_training_set():
    # Create a randomized index into the full / combined training-set.
    idx = np.random.permutation(combined_size)

    # Split the random index into training- and validation-sets.
    idx_train = idx[0:train_size]
    idx_validation = idx[train_size:]

    # Select the images and labels for the new training-set.
    x_train = combined_images[idx_train, :]
    y_train = combined_labels[idx_train, :]

    # Select the images and labels for the new validation-set.
    x_validation = combined_images[idx_validation, :]
    y_validation = combined_labels[idx_validation, :]

    # Return the new training- and validation-sets.
    return x_train, y_train, x_validation, y_validation

vaccuracy = []

with tf.Session(config=tf.ConfigProto(
        intra_op_parallelism_threads=NUM_THREADS)) as session:
    # if os.path.exists(save_path):
    #     saver.restore(sess=session, save_path=save_path)

    for i in range(num_networks):
        print("Neural network: {0}".format(i))

        # Create a random training-set. Ignore the validation-set.
        x_train, y_train, _, _ = random_training_set()

        # Initialize the variables of the TensorFlow graph.
        session.run(tf.global_variables_initializer())

        # Optimize the variables using this training-set.
        optimize(session, num_iterations=1000,
                 x_train=x_train,
                 y_train=y_train)

        # Save the optimized variables to disk.
        # saver.save(sess=session, save_path=get_save_path(i))

        # Print newline.
        validation_accuracy = session.run(accuracy,
                                          feed_dict={x: data.validation.images, y_true: data.validation.labels})
        vaccuracy.append(validation_accuracy)
        print('Validation accuracy: {0:.4}%'.format(validation_accuracy))
        print()

        # writer = tf.summary.FileWriter('./tf_logs', graph=session.graph)
        # session.run(tf.global_variables_initializer())
        # optimize(session, 1001, writer=writer)

        # validation_accuracy = session.run(accuracy, feed_dict={x: data.validation.images, y_true: data.validation.labels})
        # print('Validation accuracy: {0:.4}%'.format(validation_accuracy))

        # test_accuracy = session.run(accuracy, feed_dict={x: data.test.images, y_true: data.test.labels})
        # print('Test accuracy: {0:.4}%'.format(test_accuracy))

print('Sum accuracy: {0:.4}%'.format(np.mean(vaccuracy)))