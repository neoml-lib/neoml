"""Sample which downloads, prepares CIFAR10 dataset
and then trains a simple network on it.
Afterwards it performs some model optimizations.
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

import neoml
import numpy as np
import os
import time
import tarfile

np.random.seed(666)


def calc_md5(file_name):
    """Calculates md5 hash of an existing file"""
    import hashlib
    curr_hash = hashlib.md5()
    with open(file_name, 'rb') as file_in:
        chunk = file_in.read(8192)
        while chunk:
            curr_hash.update(chunk)
            chunk = file_in.read(8192)
    return curr_hash.hexdigest()


# Download data
url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
file_name = url[url.rfind('/')+1:]
ARCHIVE_SIZE = 170498071
ARCHIVE_MD5 = 'c58f30108f718f92721af3b95e74349a'

# Download when archive is missing or broken
if (not os.path.isfile(file_name)) \
        or os.path.getsize(file_name) != ARCHIVE_SIZE \
        or calc_md5(file_name) != ARCHIVE_MD5:
    import requests
    with requests.get(url, stream=True) as url_stream:
        url_stream.raise_for_status()
        with open(file_name, 'wb') as file_out:
            for chunk in url_stream.iter_content(chunk_size=8192):
                file_out.write(chunk)

# Unpack data
tar = tarfile.open(file_name, 'r:gz')
tar.extractall()
tar.close()


def load_batch(file_name):
    """Loads data from one of the batch files"""
    import pickle
    with open(file_name, 'rb') as file_in:
        result = pickle.load(file_in, encoding='bytes')
    return result


def transform_data(X):
    """Normalizes and transposes data for NeoML"""
    X = X.astype(np.float32)
    X = (X - 127.5) / 255.
    X = X.reshape((X.shape[0], 3, 32, 32))
    X = X.transpose((0, 2, 3, 1))  # NeoML uses channel-last pack
    return X


# Preparing data
batch_name = 'cifar-10-batches-py/data_batch_{0}'
train_data = [load_batch(batch_name.format(i)) for i in range(1, 6)]
X_train = np.concatenate(list(x[b'data'] for x in train_data), axis=0)
X_train = transform_data(X_train)
y_train = np.concatenate(list(x[b'labels'] for x in train_data), axis=0)
y_train = y_train.astype(np.int32)

test_data = load_batch('cifar-10-batches-py/test_batch')
X_test = test_data[b'data']
X_test = transform_data(X_test)
y_test = np.array(test_data[b'labels'], dtype=np.int32)


def make_blob(data, math_engine):
    """Wraps numpy data into neoml blob"""
    shape = data.shape
    if len(shape) == 4:  # data
        # Wrap 4-D array into (BatchWidth, Height, Width, Channels) blob
        blob_shape = (1, shape[0], 1, shape[1], shape[2], 1, shape[3])
        return neoml.Blob.asblob(math_engine, data, blob_shape)
    elif len(shape) == 1:  # dense labels
        # Wrap 1-D array into blob of (BatchWidth,) shape
        return neoml.Blob.asblob(math_engine, data,
                                 (1, shape[0], 1, 1, 1, 1, 1))
    else:
        assert(False)


def cifar10_array_iter(X, y, batch_size):
    """Slices numpy arrays into batches"""
    start = 0
    data_size = y.shape[0]
    while start < data_size:
        yield X[start : start+batch_size], y[start : start+batch_size]
        start += batch_size


def cifar10_blob_iter(X, y, batch_size, math_engine):
    """Slices numpy arrays into batches and wraps them in blobs"""
    for X_b, y_b in cifar10_array_iter(X, y, batch_size):
        yield make_blob(X_b, math_engine), make_blob(y_b, math_engine)


def run_net(X, y, batch_size, dnn, is_train):
    """Runs dnn on given data"""
    start = time.time()
    total_loss = 0.
    run_iter = dnn.learn if is_train else dnn.run
    math_engine = dnn.math_engine
    layers = dnn.layers
    loss = layers['loss']
    accuracy = layers['accuracy']
    sink = layers['accuracy_sink']

    accuracy.reset = True  # Reset previous statistics
    for X_batch, y_batch in cifar10_blob_iter(X, y, batch_size, math_engine):
        run_iter({'data': X_batch, 'labels': y_batch})
        total_loss += loss.last_loss * y_batch.batch_width
        accuracy.reset = False  # Don't reset statistics within one epoch

    avg_loss = total_loss / y.shape[0]
    avg_acc = sink.get_blob().asarray()[0]
    run_time = time.time() - start
    return avg_loss, avg_acc, run_time


class ConvBlock:
    """Block of dropout->conv->batch_norm->relu6"""
    def __init__(self, inputs, filter_count, name):
        self.dropout = neoml.Dnn.Dropout(inputs, rate=0.1, spatial=True,
                                         batchwise=True, name=name+'_dropout')
        self.conv = neoml.Dnn.Conv(self.dropout, filter_count=filter_count,
                                   filter_size=(3, 3), stride_size=(2, 2),
                                   padding_size=(1, 1), name=name+'_conv')
        self.bn = neoml.Dnn.BatchNormalization(self.conv, channel_based=True,
                                               name=name+'_bn')
        self.output = neoml.Dnn.ReLU(self.bn, threshold=6., name=name+'_relu6')


# Create math engine
math_engine = neoml.MathEngine.GpuMathEngine(0)
# If GPU can't be found it will return CPU math engine
print('Device: ', math_engine.info)

# Create net
dnn = neoml.Dnn.Dnn(math_engine)

# Network params
batch_size = 50
lr = 1e-3
n_classes = 10

# Create layers
data = neoml.Dnn.Source(dnn, 'data')  # Source for data
labels = neoml.Dnn.Source(dnn, 'labels')  # Source for labels
# Add a few convolutional blocks
block1 = ConvBlock(data, filter_count=16, name='block1')  # -> (16,  16)
block2 = ConvBlock(block1.output, filter_count=32, name='block2')  # -> (8, 8)
block3 = ConvBlock(block2.output, filter_count=64, name='block3')  # -> (4, 4)
# Fully connected flattens its input automatically
fc = neoml.Dnn.FullyConnected(block3.output, n_classes, name='fc')
# Softmax is applied within cross-entropy
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

n_epoch = 10
for epoch in range(n_epoch):
    # Train
    avg_loss, acc, run_time = run_net(X_train, y_train, batch_size,
                                      dnn, is_train=True)
    print(f'Train #{epoch}\tLoss: {avg_loss:.4f}\t'
          f'Accuracy: {acc:.4f}\tTime: {run_time:.2f} sec')
    # Test
    avg_loss, acc, run_time = run_net(X_test, y_test, batch_size,
                                      dnn, is_train=False)
    print(f'Test  #{epoch}\tLoss: {avg_loss:.4f}\t'
          f'Accuracy: {acc:.4f}\tTime: {run_time:.2f} sec')
    if epoch == 1:
        # If you want to save training progress you can do it via checkpoints
        # It stores dnn weights and other training data (solver stats, etc.)
        print('Creating checkpoint...')
        dnn.store_checkpoint('cifar10_sample.checkpoint')
    if epoch == 5:
        # If you want you can resume training from the checkpoint
        print('Loading checkpoint... (this will roll dnn back to epoch #1)')
        dnn.load_checkpoint('cifar10_sample.checkpoint')
        # Be careful! dnn now points to the new net
        # But other layer/solver variables still pointing to the old net!

# Prepare network for inference

# Remove training-only layers
dnn.delete_layer('labels')
dnn.delete_layer('loss')
dnn.delete_layer('accuracy')
dnn.delete_layer('accuracy_sink')

# Add sink for dnn output
sink = neoml.Dnn.Sink(dnn.layers['fc'], name='sink')


def fuse_batch_norm(dnn, block_name):
    """Fuses batch_norm into convolution
    As a result reduces inference time
    Should be used after training
    """
    bn_name = block_name + '_bn'
    if not dnn.has_layer(bn_name):
        # Batch norm has already been fused
        return
    bn_layer = dnn.layers[bn_name]
    conv_name = block_name + '_conv'
    conv_layer = dnn.layers[conv_name]
    # Fuse batch normalization
    conv_layer.apply_batch_normalization(bn_layer)
    # Delete layer from net (conv already 'contains' it)
    dnn.delete_layer(bn_name)
    # Connect layer after batchnorm to convolution
    # because batchnorm was removed from the dnn
    output_name = block_name + '_relu6'
    dnn.layers[output_name].connect(conv_layer)


# Fuse batchnorms into convolutions
fuse_batch_norm(dnn, 'block1')
fuse_batch_norm(dnn, 'block2')
fuse_batch_norm(dnn, 'block2')

# Store trained net
# In that case it's better to use method load/store
# Unlike checkpoints those aren't working with training-related data
# As a result they use less disk space
dnn.store('cifar10_sample.dnn')

# Load trained net
# It's done for sample purpose only
# You may comment the next line and everything will be just fine
dnn.load('cifar10_sample.dnn')

# Be careful! Layer variables must be updated
# because they're pointing to the layers of the old dnn
sink = dnn.layers['sink']

# Evaluate inference
inference_acc = 0.
for X_b, y_b in cifar10_array_iter(X_test, y_test, batch_size):
    dnn.run({'data': make_blob(X_b, math_engine)})
    # Extract data from sink
    # unnormalized probs of shape (batch_size, n_classes)
    logits = sink.get_blob().asarray()
    # Calculate accuracy
    inference_acc += (np.argmax(logits, axis=1) == y_b).sum()
inference_acc /= len(X_test)

# This number must be equal to the test accuracy of the last epoch
print(f'Inference net test accuracy: {inference_acc:.4f}')
