import tensorflow as tf
import numpy as np
import data_helpers
from tensorflow.contrib import learn
from proper_noun_nn import ProperNounNN
from sklearn.feature_extraction.text import CountVectorizer
import time
import datetime
import os

# Train: 23121 examples

# Parameters
# ==================================================

train_filename = "pnp-train-utf8.txt"
test_filename = "pnp-test-utf8.txt"
validate_filename = "pnp-validate-utf8.txt"

# Model Hyperparameters
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Prob that node is kept and not masked/dropped (default: 0.5)")
tf.flags.DEFINE_string("num_nodes", "20", "Number of nodes in fully connected layer")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 5, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 200, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 200, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("min_steps", 1500, "Minimum number of steps before beginning early stopping")
tf.flags.DEFINE_integer("patience", 8, "Number of decreasing validation errors (measured in `evaluate_every` steps) to warrant early stopping")

# Misc Parameters
#True means TensorFlow will automatically choose an existing and supported device to run the operations in case the specified one doesn't exist
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
# To find out which devices your operations and tensors are assigned to, create the session with log_device_placement configuration option set to True.
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Data Preparation
# ==================================================

# Load data
print("Loading data...")
x_text, y_data = data_helpers.load_data_and_labels(train_filename)
x_dev, y_dev = data_helpers.load_data_and_labels(validate_filename)

print(len(x_text))
print(len(y_data))

cv2 = CountVectorizer(lowercase=False, ngram_range=(1,2), analyzer="char_wb")
r2 = cv2.fit_transform(x_text)
print(np.transpose(r2).shape)
print(type(r2))

def generate_features(x_text):
    """
    Takes list of proper noun strings as input and outputs np array of feature vectors.
    """
    # initialize feature dictionary; key=index of noun in x_text, 
    feature_dict = {}
    for i in range(len(x_text)):
        feature_dict[i] = []

    # word length
    for i in range(len(x_text)):
        feature_list[i].append(len(x_text[i]))

    # unigram
    seen_chars = []
    for i in range(len(x_text)):
        for c in x_text[i]:
            if c not in seen_chars:
                seen_chars.append(c)

x_train = np.transpose(np.transpose(r2).todense())
y_train = y_data

print(x_train)
print(x_train.shape)
print(y_train.shape)

# Build vocabulary
max_noun_length = max([len(x) for x in x_text])
# VocabularyProcessor: Maps documents to sequences of word ids.
vocab_processor = learn.preprocessing.VocabularyProcessor(max_noun_length)

# Randomly shuffle data
"""np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y_data)))
x_train = x_data[shuffle_indices]
y_train = y_data[shuffle_indices]"""

# Training
# ==================================================
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        nn = ProperNounNN(
            num_features=x_train.shape[1], # CHECK THIS
            num_classes=5,
            #vocab_size=len(vocab_processor.vocabulary_), # CHECK THIS
            num_nodes= list(map(int, FLAGS.num_nodes.split(","))),
            dropout_keep_prob=FLAGS.dropout_keep_prob,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3) #SGD (arg is step size)
        grads_and_vars = optimizer.compute_gradients(nn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", nn.loss)
        acc_summary = tf.scalar_summary("accuracy", nn.accuracy)

        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())

        # Save parameters
        with open(os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp, "Parameters.txt")), "w") as text_file:
            for attr, value in sorted(FLAGS.__flags.items()):
                text_file.write("{}={}\n".format(attr.upper(), value))

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        def train_step(x_batch, y_batch, print_bool=False):
            """
            A single training step
            """
            feed_dict = {
                nn.input_x: x_batch,
                nn.input_y: y_batch,
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, nn.loss, nn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            if (print_bool):
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

            return (time_str, step, loss, accuracy)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                nn.input_x: x_batch,
                nn.input_y: y_batch,
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, nn.loss, nn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

            return (time_str, step, loss, accuracy)

        # Generate batches
        print("zip")
        print(x_train.shape)
        print(y_train.shape)
        a = np.squeeze(x_train[0])
        print(a.shape)
        print(y_train[0].shape)
        a = list(zip(x_train, y_train))
        print(np.squeeze(np.transpose(a[0][0])).shape)
        print(a[0][1].shape)
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

        max_accuracy = 0
        decreasing_accuracy_count = 0
        current_step = 0

        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_time_str, step, train_loss, train_accuracy = train_step(x_batch, y_batch, current_step % FLAGS.evaluate_every == 0)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                time_str, step, loss, accuracy = dev_step(x_dev, y_dev, writer=dev_summary_writer)
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    decreasing_accuracy_count = 0
                else:
                    decreasing_accuracy_count += 1
                if decreasing_accuracy_count >= FLAGS.patience and FLAGS.min_steps < current_step:
                    break
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))


        print("\nFinal Evaluation:")
        time_str, step, loss, accuracy = dev_step(x_dev, y_dev, writer=dev_summary_writer)
        print("Maximum validation accuracy: {}".format(max_accuracy))
        print("")

        with open(os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp, "Parameters.txt")), "a") as text_file:
            text_file.write("\n\nTime Step = {}".format(step))
            text_file.write("\nTrain Loss = {}".format(train_loss))
            text_file.write("\nTrain Accuracy = {}".format(train_accuracy))
            text_file.write("\nValidation Loss = {}".format(loss))
            text_file.write("\nValidation Accuracy = {}".format(accuracy))

            for attr, value in sorted(FLAGS.__flags.items()):
                text_file.write("{}={}\n".format(attr.upper(), value))

        