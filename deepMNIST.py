import argparse

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None


def create_input_placeholders(number_of_features, number_of_labels):
    x = tf.placeholder(dtype=tf.float32, shape=(None, number_of_features))
    y_ = tf.placeholder(dtype=tf.float32, shape=(None, number_of_labels))
    return x, y_


def generate_weights(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)


def generate_biases(shape):
    initial = tf.constant(value=0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pooling(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def infer(x):
    W_conv1 = generate_weights([5, 5, 1, 32])
    b_conv1 = generate_biases([32])
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pooling(h_conv1)

    W_conv2 = generate_weights([5, 5, 32, 64])
    b_conv2 = generate_biases([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pooling(h_conv2)

    W_fc1 = generate_weights([7 * 7 * 64, 1024])
    b_fc1 = generate_biases([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # keep_prob = tf.placeholder(tf.float32)
    # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = generate_weights([1024, 10])
    b_fc2 = generate_biases([10])

    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
    return y_conv


def get_cost_function(logits, labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))


def get_train_op(cost_function, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(cost_function, global_step)
    return train_op


def fill_feed_dict(data_sets, input_placeholder, labels_placeholder, batch_size):
    input_feed, labels_feed = data_sets.train.next_batch(batch_size)

    feed_dict = {
        input_placeholder: input_feed,
        labels_placeholder: labels_feed
    }

    return feed_dict


def run_training():
    data_sets = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    with tf.Graph().as_default():
        x, y_ = create_input_placeholders(number_of_features=784, number_of_labels=10)
        y = infer(x)
        train_op = get_train_op(get_cost_function(y, y_), learning_rate=1e-4)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for i in range(20000):
                if i % 100 == 0:
                    train_accuracy = accuracy.eval(feed_dict=fill_feed_dict(data_sets, x, y_, 50))
                    print("step %d, training accuracy %g" % (i, train_accuracy))
                train_op.run(feed_dict=fill_feed_dict(data_sets, x, y_, 50))

            print("test accuracy %g" % accuracy.eval(feed_dict={
                x: data_sets.test.images, y_: data_sets.test.labels}))


def main(_):
    run_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/data',
                        help='Directory for storing data')
    FLAGS = parser.parse_args()
    tf.app.run()
