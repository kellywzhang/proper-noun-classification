import data_utils
import numpy as np
import tensorflow as tf

import tf_helpers
import data_utils
from RNNClassification import RNNClassifier

train_filename = "pnp-train.txt"
validate_filename = "pnp-validate.txt"
test_filename = "pnp-test.txt"

embedding_dim = 64
batch_size = 32
hidden_size = 100
num_classes = 5
num_epochs = 5
learning_rate = 0.001

x_train, y_train, seq_lens_train, vocab_dict \
    = data_utils.load_data_and_labels(train_filename)
train_data = list(zip(x_train, y_train, seq_lens_train))

x_dev, y_dev, seq_lens_dev, vocab_dict \
    = data_utils.load_data_and_labels(validate_filename, train=False)
dev_data = list(zip(x_dev, y_dev, seq_lens_dev))

batches = data_utils.batch_iter(train_data, batch_size=batch_size, num_epochs=num_epochs, shuffle=True)
batches_dev = data_utils.batch_iter(dev_data, batch_size=batch_size, num_epochs=num_epochs, shuffle=True)

for batch in batches_dev:
    x_batch = batch[:,0]
    y_batch = batch[:,1]
    seq_lens_batch = batch[:,2]

    nn = RNNClassifier(
            num_classes=num_classes,
            vocab_size=len(vocab_dict),
            hidden_size=hidden_size,
            embedding_dim=embedding_dim,
            batch_size=batch_size,
            bidirectional=False
        )

    feed_dict = {
        nn.input_x: x_batch,
        #nn.input_y: y_batch,
        #nn.seq_lens: seq_lens_batch
    }

    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    nn.W_embeddings.eval()
    sess.close()
