import data_utils
import numpy as np

train_filename = "pnp-train.txt"
validate_filename = "pnp-validate.txt"
test_filename = "pnp-test.txt"

# Load Data
vectorized_train, train_labels, seq_lens_train, vocab_dict \
    = data_utils.load_data_and_labels(train_filename)
train_data = list(zip(vectorized_train, train_labels, seq_lens_train))

x_dev, y_dev, seq_lens_dev, vocab_dict \
    = data_utils.load_data_and_labels(validate_filename)

padded = data_utils.pad(vectorized_train, seq_lens_train)
print(padded[:100])

"""for x in data_utils.batch_iter(train_data, batch_size=32, num_epochs=1, shuffle=True):
    print(type(x[:,0]))
    print(x[:,1])
    print(x[:,2])
    print("\n")"""

# Parameters
vocab_size = len(vocab_dict)
embedding_dim = 64
batch_size = 32
hidden_size = 100
num_classes = 5
num_epochs = 5

# Create NN Layers
