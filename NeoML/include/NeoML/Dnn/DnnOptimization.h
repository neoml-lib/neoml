/* Copyright © 2017-2024 ABBYY

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
struct NEOML_API CDnnOptimizationReport final {
	// Number of HeadAdapters layers which internal dnn is optimized
	int OptimizedHeadAdapters = 0;
	// Number of composite layers which where unpacked
	// (unpack == content of the layer moved to the root CDnn, composite itself is removed)
	int UnpackedCompositeLayers = 0;
	// Number of trivial layers removed from the CDnn (dropout, linear(1,0) etc)
	int RemovedTrivialLayers = 0;
	// Number of batch normalizations fused into other layers
	int FusedBatchNormalizations = 0;
	// Number of merged (channelwise->activation->1x1) constructions without residual connection
	int ChannelwiseWith1x1NonResidual = 0;
	// Number of merged (channelwise->activation->1x1) constructions with residual connection
	int ChannelwiseWith1x1Residual = 0;
	// Number of optimized MobileNetV2 blocks without residual connection
	int MobileNetV2NonResidualBlocks = 0;
	// Number of optimized MobileNetV2 blocks with residual connection
	int MobileNetV2ResidualBlocks = 0;
	// Number of optimized MobileNetV3 blocks without residual connection
	int MobileNetV3NonResidualBlocks = 0;
	// Number of optimized MobileNetV3 blocks with residual connection
	int MobileNetV3ResidualBlocks = 0;
	// Number of chains of rowwise operations
	int RowwiseChainCount = 0;

	bool IsOptimized() const;
};

// Check for is any optimization succeed
inline bool CDnnOptimizationReport::IsOptimized() const
{
	return OptimizedHeadAdapters > 0
		|| UnpackedCompositeLayers > 0
		|| RemovedTrivialLayers > 0
		|| FusedBatchNormalizations > 0
		|| ChannelwiseWith1x1NonResidual > 0
		|| ChannelwiseWith1x1Residual > 0
		|| MobileNetV2NonResidualBlocks > 0
		|| MobileNetV2ResidualBlocks > 0
		|| MobileNetV3NonResidualBlocks > 0
		|| MobileNetV3ResidualBlocks > 0
		|| RowwiseChainCount > 0;
}

// Settings for optional optimizations
struct NEOML_API CDnnOptimizationSettings final {
	// Enable additional optimizations which are useful for inference on CPU
	// Recommended for convolutional nets with at least 20MB RAM usage
	// (You can measure RAM usage by running the dnn and dnn.GetMathEngine().GetPeakMemoryUsage())
	// Turned ON by default because it's the most common scenario
	// (After these optimizations dnn still can be launched via CUDA
	// but they may lead to increased VRAM consumption)
	bool AllowCpuOnlyOptimizations = true;
};

// Optimizes inference of given CDnn at the cost of trainability
//
// List of supported optimizations:
//
//     1. Fuses batch normalization into convolutions and fully-connected layers when possible
//
//     2. Channelwise with 1x1 optimizations.
//        Replaces the non-residual blocks of layers
//            channelwise3x3 -> activation -> conv1x1
//        and residual blocks of layers
//            -+--> non-residual block ----> sum ->
//             |                              |
//             +------------------------------+
//        with optimized CChannelwiseWith1x1Layer
//        ReLU and HSwish activations are supported (or trivial Linear{mul=1, ft=0}).
//
//     3. MobileNetV2 block optimizations.
//        Replaces the non-residual blocks of layers
//            conv1x1 (expand) -> expandActivation -> channelwise3x3 -> channelwiseActivation -> conv1x1 (down)
//        and residual blocks of layers
//            -+--> non-residual block ----> sum ->
//             |                              |
//             +------------------------------+
//        with optimized CMobileNetV2BlockLayer
//        ReLU and HSwish activations are supported (or trivial Linear{mul=1, ft=0}).
//
//     4. MobileNetV3 block optimizations.
//        Replaces the non-residual blocks of layers
//            conv1x1 (expand) -> expandActivation -> channelwise(3x3 or 5x5) -> channelwiseActivation ->
//                -> Squeeze-and-Excite -> conv1x1 (down)
//        or
//            conv1x1 (expand) -> expandActivation -> channelwise(3x3 or 5x5) -> Squeeze-and-Excite ->
//                -> channelwiseActivation -> conv1x1 (down)
//        and residual blocks of layers
//            -+--> non-residual block ----> sum ->
//             |                              |
//             +------------------------------+
//        with optimized CMobileNetV3BlockLayer
//        ReLU and HSwish activations are supported (or trivial Linear{mul=1, ft=0}).
CDnnOptimizationReport NEOML_API OptimizeDnn( CDnn& dnn,
	const CDnnOptimizationSettings& settings = CDnnOptimizationSettings() );

} // namespace NeoML
