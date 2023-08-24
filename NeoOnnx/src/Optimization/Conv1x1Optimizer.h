/* Copyright © 2023 ABBYY

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

// Forward declaration(s)
namespace NeoML {
namespace optimization {
class CGraph;
} // namespace optimization
} // namespace NeoML

namespace NeoOnnx {

namespace optimization {

// In order to train faster on PyTorch sometimes 1x1 Convolutions are replaced with MatMul + Add
// This optimizer replaces these operations with convolution (it's better for NeoML::OptimizeDnn)
int OptimizeConv1x1( NeoML::optimization::CGraph& graph );

} // namespace optimization

} // namespace NeoOnnx

