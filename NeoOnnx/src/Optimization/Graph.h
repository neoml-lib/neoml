/* Copyright © 2017-2023 ABBYY Production LLC

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

#include <NeoML/Dnn/Optimization/Graph.h>

namespace NeoOnnx {

namespace optimization {

// NeoOnnx::optimization aliases for NeoML::optimization
using CGraph = NeoML::optimization::CGraph;
template<typename TLayer = CBaseLayer>
using CLayerInput = NeoML::optimization::CLayerInput<TLayer>;
template<typename TLayer = CBaseLayer>
using CLayerOutput = NeoML::optimization::CLayerOutput<TLayer>;

} // namespace optimization

} // namespace NeoOnnx
