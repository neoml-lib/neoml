/* Copyright © 2017-2020 ABBYY Production LLC

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

#pragma once

#include <NeoOnnx/NeoOnnxDefs.h>
#include <NeoML/NeoML.h>

namespace NeoOnnx {

// Loads network "dnn" from onnx file "fileName"
//
// For every uninitialized onnx graph input there will be CSourceLayer with the same name
// For every CSourceLayer will be allocated input blob of the given size
// Inputs with initializers will be ignored (used for parameters calculation)
//
// For every onnx graph output there will be CSinkLayer with the same name
// Graph inputs' and outputs' names will be added to the corresponding CArray's
// Names' pointers are attached to the corresponding layers' names
//
// Throws std::logic_error if failed to load network
NEOONNX_API void LoadFromOnnx( const char* fileName, NeoML::CDnn& dnn, CArray<const char*>& inputs, CArray<const char*>& outputs );

// Loads network "dnn" from buffer with onnx data
//
// For every uninitialized onnx graph input there will be CSourceLayer with the same name
// For every CSourceLayer will be allocated input blob of the given size
// Inputs with initializers will be ignored (used for parameters calculation)
//
// For every onnx graph output there will be CSinkLayer with the same name
// Graph inputs' and outputs' names will be added to the corresponding CArray's
// Names' pointers are attached to the corresponding layers' names
//
// Throws std::logic_error if failed to load network
NEOONNX_API void LoadFromOnnx( const void* buffer, int bufferSize, NeoML::CDnn& dnn, CArray<const char*>& inputs, CArray<const char*>& outputs );

} // namespace NeoOnnx
