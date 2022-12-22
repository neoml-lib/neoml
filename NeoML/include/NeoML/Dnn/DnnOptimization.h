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

namespace NeoML {

// Struct which is returned by OptimizeDnn function
struct NEOML_API CDnnOptimizationReport {
	// Number of optimized MobileNetV2 blocks without residual connection
	int MobileNetV2NonResidualBlocks;
	// Number of optimized MobileNetV2 blocks with residual connection
	int MobileNetV2ResidualBlocks;
};

// Optimizes inference of given CDnn at the cost of trainability
//
// List of supported optimizations:
//
//     1. MobileNetV2 block optimizations.
//        Replaces the non-residual blocks of layers
//
//            conv1x1 (expand) -> relu (expandReLU) -> channelwiseConv3x3 -> relu (channelwiseReLU) -> conv1x1 (down)
//
//        and residual blocks of layers
//
//            -+--> non-residual block ----> sum ->
//             |                              |
//             +------------------------------+
//
//        with optimized CMobileNetV2BlockLayer
//
CDnnOptimizationReport NEOML_API OptimizeDnn( CDnn& dnn );

} // namespace NeoML
