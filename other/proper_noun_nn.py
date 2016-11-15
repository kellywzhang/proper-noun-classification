import tensorflow as tf
import numpy as np

class ProperNounNN(object):
    """
    A NN for proper noun classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    # go through and number the weight vector matrices
    def __init__(self, num_features, num_classes, num_nodes, dropout_keep_prob, l2_reg_lambda=0.0): 

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, num_features], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Fully connected layers
        for i in range(len(num_nodes)):
            with tf.name_scope("fully-connected-layer-{}".format(i+1)):
                b_fc = tf.get_variable("b_fc-{}".format(i+1), shape=[num_nodes[i]], initializer=tf.constant_initializer(0.0))

                if i == 0:
                    W_fc = tf.get_variable(
                        name="W_fc-{}".format(i+1),
                        shape=[num_features, num_nodes[i]],
                        initializer=tf.contrib.layers.xavier_initializer())
                    h = tf.nn.relu(tf.matmul(self.input_x, W_fc) + b_fc)
                else:
                    W_fc = tf.get_variable(
                        name="W_fc-{}".format(i+1),
                        shape=[num_nodes[i-1], num_nodes[i]],
                        initializer=tf.contrib.layers.xavier_initializer())
                    h = tf.nn.relu(tf.matmul(h_dropped, W_fc) + b_fc)

                self.h_dropped = tf.nn.dropout(h, dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_nodes[-1], num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.0, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_dropped, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
            # calculates cross entropy for log-probabilities and true labels
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            # take mean of correct_predictions counts => accuracy
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
