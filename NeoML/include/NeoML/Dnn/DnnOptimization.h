/* Copyright © 2017-2022 ABBYY Production LLC

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

#include <NeoML/NeoMLDefs.h>

namespace NeoML {

class CDnn;

// Struct which contains the details of optimization result
struct NEOML_API CDnnOptimizationReport {
	// Number of optimized MobileNetV2 blocks without residual connection
	int MobileNetV2NonResidualBlocks;
	// Number of optimized MobileNetV2 blocks with residual connection
	int MobileNetV2ResidualBlocks;
	// Number of optimized MobileNetV3 blocks without residual connection
	int MobileNetV3NonResidualBlocks;
	// Number of optimized MobileNetV3 blocks with residual connection
	int MobileNetV3ResidualBlocks;
	// Number of merged (channelwise->activation->1x1) constructions without residual connection
	int ChannelwiseWith1x1NonResidual;
	// Number of merged (channelwise->activation->1x1) constructions with residual connection
	int ChannelwiseWith1x1Residual;
};

// Optimizes inference of given CDnn at the cost of trainability
//
// List of supported optimizations:
//
//
//     1. Channelwise with 1x1 optimizations.
//        Replaces the non-residual blocks of layers
//
//            channelwise3x3 -> activation -> conv1x1
//
//        and residual blocks of layers
//
//            -+--> non-residual block ----> sum ->
//             |                              |
//             +------------------------------+
//
//        with optimized CChannelwiseWith1x1Layer
//        ReLU and HSwish activations are supported (or trivial Linear{mul=1, ft=0}).
//
//
//     2. MobileNetV2 block optimizations.
//        Replaces the non-residual blocks of layers
//
//            conv1x1 (expand) -> expandActivation -> channelwise3x3 -> channelwiseActivation -> conv1x1 (down)
//
//        and residual blocks of layers
//
//            -+--> non-residual block ----> sum ->
//             |                              |
//             +------------------------------+
//
//        with optimized CMobileNetV2BlockLayer
//        ReLU and HSwish activations are supported (or trivial Linear{mul=1, ft=0}).
//
//
//     3. MobileNetV3 block optimizations.
//        Replaces the non-residual blocks of layers
//
//            conv1x1 (expand) -> expandActivation -> channelwise(3x3 or 5x5) -> channelwiseActivation ->
//                -> Squeeze-and-Excite -> conv1x1 (down)
//        or
//            conv1x1 (expand) -> expandActivation -> channelwise(3x3 or 5x5) -> Squeeze-and-Excite ->
//                -> channelwiseActivation -> conv1x1 (down)
//
//
//        and residual blocks of layers
//
//            -+--> non-residual block ----> sum ->
//             |                              |
//             +------------------------------+
//
//        with optimized CMobileNetV3BlockLayer
//        ReLU and HSwish activations are supported (or trivial Linear{mul=1, ft=0}).
//
CDnnOptimizationReport NEOML_API OptimizeDnn( CDnn& dnn );

} // namespace NeoML
