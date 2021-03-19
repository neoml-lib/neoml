""" Copyright (c) 2017-2020 ABBYY Production LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
--------------------------------------------------------------------------------------------------------------*/
"""

import neoml.PythonWrapper as PythonWrapper
import neoml.Random


# -------------------------------------------------------------------------------------------------------------


class Dnn(PythonWrapper.Dnn):
    """Neural network implementation.
    A neural network is a directed graph consisting of layers that perform
    calculations on data blobs. It starts with source layers and ends with 
    sink layers.
    
    Parameters
    ---------
    math_engine : object
        The math engine that will perform calculations.
    random : object, default=None
        The random numbers generator to be used for training and initialization.
    """
    def __init__(self, math_engine, random=None):
        if not isinstance(math_engine, neoml.MathEngine.MathEngine):
            raise ValueError('The `math_engine` must be a neoml.MathEngine.MathEngine.')

        if random is None:
            random = neoml.Random.Random()

        if not isinstance(random, neoml.Random.Random):
            raise ValueError('The `random_generator` must be a neoml.Random.Random.')

        super().__init__(random, math_engine._internal)

    def store(self, path):
        """Serializes the network.
        
        Parameters
        ---------
        path : str
            The full path to the location where the network should be stored.
        """
        self._store(str(path))

    def load(self, path):
        """Loads the network from file.
        
        Parameters
        ---------
        path : str
            The full path to the location from where the network should be loaded.
        """
        self._load(str(path))

    def store_checkpoint(self, path):
        """Serializes the network with the data required to resume training.
        
        Parameters
        ---------
        path : str
            The full path to the location where the network should be stored.
        """
        self._store_checkpoint(str(path))

    def load_checkpoint(self, path):
        """Loads the checkpoint from file.
        A new solver will be created, because the old pointers will point
        to an object no longer used by this network.
        
        Parameters
        ---------
        path : str
            The full path to the location from where the network should be loaded.
        """
        self._load_checkpoint(str(path))

    @property
    def math_engine(self):
        """Gets the math engine used by the network.
        """
        return self.get_math_engine()

    @property
    def solver(self):
        """Gets the optimizer for the layer's trainable parameters.
        """
        return self.get_solver()

    @solver.setter
    def solver(self, new_solver):
        """Sets the optimizer for the layer's trainable parameters.
        """
        self.set_solver(new_solver._internal)

    @property
    def initializer(self):
        """Gets the initializer that will fill in the weight values 
        before training starts. Xavier initialization is the default.
        """
        return self.get_initializer()

    @initializer.setter
    def initializer(self, new_initializer):
        """Sets the initializer that will fill in the weight values 
        before training starts.
        """
        self.set_initializer(new_initializer._internal)

    @property
    def input_layers(self):
        """Gets the source layers of the network.
        """
        return self.get_inputs()

    @property
    def output_layers(self):
        """Gets the sink layers of the network.
        """
        return self.get_outputs()

    @property
    def layers(self):
        """Gets all layers of the network.
        """
        return self.get_layers()

    def add_layer(self, layer):
        """Adds layer to the network

        :param layer: The layer to be added
        :type layer: neoml.Dnn.Layer
        """
        if not isinstance(layer, Layer):
            raise ValueError('The `layer` is expected to be neoml.Dnn.Layer`')
        self._add_layer(layer.internal)
    
    def delete_layer(self, layer):
        """Deletes layer from the network

        :param layer: The layer (or its name) to be deleted
        :type layer: neoml.Dnn.Layer or str
        """
        if type(layer) is str:
            self._delete_layer(layer)
        elif isinstance(layer, Layer):
            self._delete_layer(layer.name)
        else:
            raise ValueError('The `layer` is expected to be `str` or `neoml.Dnn.Layer`')

    def run(self, inputs):
        """Runs the network.
        
        Parameters
        ----------
        inputs : The dictionary of input blobs.
            The dictionary keys (str) are the source layer names.
            The dictionary values (neoml.Blob) are the blobs passed 
            to these source layers.

        Returns
        -------
            The list of output blobs.
        """
        dnn_inputs = self.get_inputs()

        if len(inputs) < len(dnn_inputs):
            raise ValueError('The `inputs` contains less layers than the dnn.')

        if len(inputs) > len(dnn_inputs):
            raise ValueError('The `inputs` contains more layers than the dnn.')

        input_list = []
        for layer_name in dnn_inputs:
            input_list.append(inputs[layer_name]._internal)

        dnn_outputs = self._run(input_list)

        outputs = {}
        for layer_name in dnn_outputs:
            outputs[layer_name] = neoml.Blob.Blob(dnn_outputs[layer_name])

        return outputs

    def run_and_backward(self, inputs):
        """Runs the network and performs a backward pass with the input data.
        
        Parameters
        ---------
        inputs : The dictionary of input blobs.
            The dictionary keys (str) are the source layer names.
            The dictionary values (neoml.Blob) are the blobs passed 
            to these source layers.

        Returns
        -------
            The list of output blobs.
        """
        dnn_inputs = self.get_inputs()

        if len(inputs) < len(dnn_inputs):
            raise ValueError('The `inputs` contains less layers than the dnn.')

        if len(inputs) > len(dnn_inputs):
            raise ValueError('The `inputs` contains more layers than the dnn.')

        input_list = []
        for layer_name in dnn_inputs:
            input_list.append(inputs[layer_name]._internal)

        return self._run_and_backward(input_list)

    def learn(self, inputs):
        """Runs the network, performs a backward pass 
        and updates the trainable weights.
        
        Parameters
        ---------
        inputs : The dictionary of input blobs.
            The dictionary keys (str) are the source layer names.
            The dictionary values (neoml.Blob) are the blobs passed 
            to these source layers.

        Returns
        -------
            The list of output blobs.
        """
        dnn_inputs = self.get_inputs()

        if len(inputs) < len(dnn_inputs):
            raise ValueError('The `inputs` contains less layers than the dnn.')

        if len(inputs) > len(dnn_inputs):
            raise ValueError('The `inputs` contains more layers than the dnn.')

        input_list = []
        for layer_name in dnn_inputs:
            input_list.append(inputs[layer_name]._internal)

        self._learn(input_list)

# -------------------------------------------------------------------------------------------------------------


class Layer:
    """The base class for a network layer.
    """
    def __init__(self, internal):
        if not isinstance(internal, PythonWrapper.Layer):
            raise ValueError('The `internal` must be PythonWrapper.Layer')

        self._internal = internal

    @property
    def name(self):
        """Gets the layer name.
        """
        return self._internal.get_name()
    
    def connect(self, layer, output_index=0, input_index=0):
        if not isinstance(layer, Layer):
            raise ValueError('The `layer` is expected to be neoml.Dnn.Layer')
        if output_index < 0:
            raise ValueError('The `output_index` must be >= 0')
        if input_index < 0:
            raise ValueError('The `input_index` must be >= 0')
        self._internal.connect(layer._internal, output_index, input_index)
