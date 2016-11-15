import tensorflow as tf
import numpy as np
import os

import data_utils

#1479175091
checkpoint_dir = os.path.join(os.path.abspath(os.path.curdir), "runs", "1479176649", "checkpoints")

# Shared Parameters
# ==================================================
embedding_dim = 64
batch_size = 54
hidden_size = 100
num_classes = 5
num_epochs = 10
learning_rate = 0.001

# Load Data
# ==================================================
test_filename = "pnp-test.txt"

x_test, y_test, seq_lens_test, vocab_dict \
    = data_utils.load_data_and_labels(test_filename, train=False)
test_data = list(zip(x_test, y_test, seq_lens_test))

print(vocab_dict)

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()

    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        input_y = graph.get_operation_by_name("input_y").outputs[0]
        seq_lens = graph.get_operation_by_name("seq_lens").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("prediction/predictions").outputs[0]
        prediction_probs = graph.get_operation_by_name("prediction/prediction_probs").outputs[0]
        correct_vector = graph.get_operation_by_name("prediction/correct_vector").outputs[0]

        # Collect the predictions here
        all_predictions = []
        all_labels = []

        # Generate batches for one epoch
        batches = data_utils.batch_iter(test_data, batch_size=batch_size, num_epochs=1, shuffle=False)

        probs = []

        for batch in batches:
            x_batch = batch[:,0]
            y_batch = batch[:,1]
            seq_lens_batch = batch[:,2]

            x_batch = data_utils.pad(x_batch, seq_lens_batch)

            batch_predictions, correct, batch_probs = sess.run([predictions, correct_vector, prediction_probs], \
            #batch_predictions = sess.run([predictions], \
                {input_x: x_batch, input_y: y_batch, seq_lens: seq_lens_batch})
            #print(batch_probs)
            correct += np.sum(correct)
            #print(y_batch)
            all_labels = np.concatenate([all_labels, y_batch]) #np.concatenate([all_labels, [np.argmax(x) for x in y_batch]])
            all_predictions = np.concatenate([all_predictions, batch_predictions])

            for i in range(len(y_batch)):
                probs.append(batch_probs[i][y_batch[i]])

# Print accuracy if y_test is defined; [np.argmax(x) for x in all_labels]
correct_predictions = np.sum(all_predictions == np.array(all_labels))
print(correct_predictions)
print("Total number of test examples: {}".format(len(seq_lens_test)))
print("Accuracy: {:g}".format(correct_predictions/float(len(seq_lens_test))))

labels = ["drug", "person", "place", "movie", "company"]

# Write predictions
test_filename = "pnp-test.txt"

file_path = os.path.join(os.path.abspath(os.path.curdir), "data", test_filename)
f = open(file_path, 'r', encoding = "ISO-8859-1")
data = list(f.readlines())
f.close()
data = [s.strip().split() for s in data]

proper_nouns_strings = [" ".join(d[1:]) for d in data]

f = open("output.txt", 'w', encoding = "ISO-8859-1")
for i in range(len(proper_nouns_strings)):
    f.write("Example: "+proper_nouns_strings[i]+" guess= "+str(labels[int(all_predictions[i])])+" gold= confidence="+str(probs[i])+"\n")
f.close()
