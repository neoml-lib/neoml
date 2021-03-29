"""This sample partially reproduces experiment from the article:
    https://arxiv.org/pdf/1504.00941.pdf
It demonstrates that IRNN is especially good when working with long sequences
It transposes MNIST images (28 x 28) into sequences of length 784
The article claims that IRNN can achieve 0.9+ accuracy in these conditions
"""

__copyright__ = """

Copyright © 2017-2021 ABBYY Production LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

__license__ = 'Apache 2.0'

import numpy as np
import neoml
import time

# Get data
from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

# Normalize
X = (255 - X) * 2 / 255 - 1

# Fix data types
X = X.astype(np.float32)
y = y.astype(np.int32)

# Split into train/test
train_size = 60000
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
del X, y


def irnn_data_iterator(X, y, batch_size, math_engine):
    """Slices numpy arrays into batches and wraps them in blobs"""
    def make_blob(data, math_engine):
        """Wraps numpy data into neoml blob"""
        shape = data.shape
        if len(shape) == 2:  # data
            # Wrap 2-D array into blob of (BatchWidth, Channels) shape
            return neoml.Blob.asblob(math_engine, data,
                                     (1, shape[0], 1, 1, 1, 1, shape[1]))
        elif len(shape) == 1:  # dense labels
            # Wrap 1-D array into blob of (BatchWidth,) shape
            return neoml.Blob.asblob(math_engine, data,
                                     (1, shape[0], 1, 1, 1, 1, 1))
        else:
            assert(False)

    start = 0
    data_size = y.shape[0]
    while start < data_size:
        yield (make_blob(X[start : start+batch_size], math_engine),
               make_blob(y[start : start+batch_size], math_engine))
        start += batch_size


def run_net(X, y, batch_size, dnn, data_layer_name, label_layer_name,
            loss_layer, accuracy_layer, accuracy_sink, is_train):
    """Runs dnn on given data"""
    start = time.time()
    total_loss = 0.
    run_iter = dnn.learn if is_train else dnn.run
    math_engine = dnn.math_engine
    accuracy_layer.reset = True  # Reset previous statistics

    for X_batch, y_batch in irnn_data_iterator(X, y, batch_size, math_engine):
        run_iter({data_layer_name: X_batch, label_layer_name: y_batch})
        total_loss += loss.last_loss * y_batch.batch_width
        accuracy_layer.reset = False  # Don't reset statistics within one epoch

    avg_loss = total_loss / y.shape[0]
    acc = accuracy_sink.get_blob().asarray()[0]
    run_time = time.time() - start
    return avg_loss, acc, run_time


# Network params
batch_size = 40
hidden_size = 100
lr = 1e-6
n_classes = 10
n_epoch = 200

data_layer_name = 'data'
label_layer_name = 'labels'

# Create net
math_engine = neoml.MathEngine.CpuMathEngine(1)
dnn = neoml.Dnn.Dnn(math_engine)

# Create layers
data = neoml.Dnn.Source(dnn, data_layer_name)  # Source for data
labels = neoml.Dnn.Source(dnn, label_layer_name)  # Source for labels
# (BatchWidth, Channels) -> (BatchLength, BatchWidth) for recurrent layer
transpose = neoml.Dnn.Transpose(data, first_dim='batch_length',
                                      second_dim='channels')
irnn = neoml.Dnn.Irnn(transpose, hidden_size, identity_scale=1.,
                       input_weight_std=1e-3, name='irnn')
# IRNN returns whole sequence, need to take only last element
subseq = neoml.Dnn.SubSequence(irnn, start_pos=-1,
                                       length=1, name='subseq')
# Forming distribution (unsoftmaxed!)
fc = neoml.Dnn.FullyConnected(subseq, n_classes, name='fc')
# Softmax is applied inside cross-entropy
loss = neoml.Dnn.CrossEntropyLoss((fc, labels), name='loss')
# Auxilary layers in order to get statistics
accuracy = neoml.Dnn.Accuracy((fc, labels), name='accuracy')
accuracy_sink = neoml.Dnn.Sink(accuracy, name='accuracy_sink')

# Create solver
dnn.solver = neoml.Dnn.AdaptiveGradient(math_engine, learning_rate=lr,
                                           l1=0., l2=0.,  # No regularization
                                           max_gradient_norm=1.,  # clip grad
                                           moment_decay_rate=0.9,
                                           second_moment_decay_rate=0.999)

for epoch in range(n_epoch):
    # Train
    avg_loss, acc, run_time = run_net(X_train, y_train, batch_size, dnn,
                                      data_layer_name, label_layer_name, loss,
                                      accuracy, accuracy_sink, is_train=True)
    print(f'Train #{epoch}\tLoss: {avg_loss:.4f}\t'
          f'Accuracy: {acc:.4f}\tTime: {run_time:.2f} sec')
    # Test
    avg_loss, acc, run_time = run_net(X_test, y_test, batch_size, dnn,
                                      data_layer_name, label_layer_name, loss,
                                      accuracy, accuracy_sink, is_train=False)
    print(f'Test  #{epoch}\tLoss: {avg_loss:.4f}\t'
          f'Accuracy: {acc:.4f}\tTime: {run_time:.2f} sec')
