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

import neoml.Dnn as Dnn


def check_input_layers(input_layers, layer_count):
    min_count = 0;
    max_count = 0;
    if isinstance(layer_count, int):
        min_count = int(layer_count)
        max_count = int(layer_count)
    else:
        min_count = int(layer_count[0])
        max_count = int(layer_count[1])

    layers = []
    outputs = []

    if isinstance(input_layers, Dnn.Layer):
        if 1 < min_count or 1 > max_count:
            raise ValueError('The layer has (' + str(min_count) + ', ' + str(max_count) + ') inputs.')
        layers.append(input_layers._internal)
        outputs.append(0)
        return layers, outputs

    if len(input_layers) == 2 and isinstance(input_layers[0], Dnn.Layer) and isinstance(input_layers[1], int):
        if 2 < min_count or 2 > max_count:
            raise ValueError('The layer has (' + str(min_count) + ', ' + str(max_count) + ') inputs.')

        layers.append(input_layers[0]._internal)
        outputs.append(int(input_layers[1]))
        return layers, outputs

    if len(input_layers) == 0:
        raise ValueError('The `input_layers` must contain at least one layer.')

    if max_count != 0 and (len(input_layers) < min_count or len(input_layers) > max_count):
        raise ValueError('The layer has (' + str(min_count) + ', ' + str(max_count) + ') inputs.')

    for i in input_layers:
        if isinstance(i, Dnn.Layer):
            layers.append(i._internal)
            outputs.append(0)
        elif isinstance(i, (list, tuple)) and len(i) == 2 and isinstance(i[0], Dnn.Layer) and isinstance(i[1], int):
            if int(i[1]) < 0 or int(i[1]) >= i[0].output_count():
                raise ValueError('Invalid value `input_layers`.'
                                 ' It must be a list of layers or a list of (layer, output).')
            layers.append(i[0]._internal)
            outputs.append(int(i[1]))
        else:
            raise ValueError('Invalid value `input_layers`. It must be a list of layers or a list of (layer, output).')

    return layers, outputs
