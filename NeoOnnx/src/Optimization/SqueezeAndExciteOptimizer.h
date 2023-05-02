/* Copyright © 2017-2023 ABBYY

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

#include "Graph.h"

namespace NeoML {

// Forward declaration(s)
class CBaseLayer;
class CConvLayer;
struct CDnnOptimizationReport;
class COnnxSourceHelper;
class COnnxTransformHelper;
class COnnxTransposeHelper;
class CGlobalMeanPoolingLayer;

} // namespace NeoML

namespace NeoOnnx {

namespace optimization {

// Removes unnenecessary trasnforms, transposes and reshapes from Squeeze-and-Excite part of MobileNetV3
class CSqueezeAndExciteOptimizer final {
public:
	explicit CSqueezeAndExciteOptimizer( CGraph& graph ) :
		graph( graph )
	{}

	void Apply();

private:
	struct CSEBlockInfo final {
		CLayerOutput<> InputData{};
		CGlobalMeanPoolingLayer* SEPooling = nullptr;
		CBaseLayer* SEFirstFc = nullptr;
		CBaseLayer* SESecondActivation = nullptr;
		CLayerInput<> SEMulVectorInput{};
	};

	CGraph& graph;

	int optimizeSEBlocks();

	bool detectSqueezAndExcite( CBaseLayer& mulLayer, CSEBlockInfo& detectedBlock );
	bool isValidMul( CBaseLayer& layer ) const;
	bool isValidSEActivation( CBaseLayer& layer ) const;
	bool isValid1x1Conv( CBaseLayer* layer ) const;
	bool isValidOnnxTransform( COnnxTransformHelper& transform, std::initializer_list<TBlobDim> expectedRules ) const;
	bool isValidOnnxTranspose( COnnxTransposeHelper& transpose, TBlobDim firstDim, TBlobDim secondDim ) const;
	bool isValidOnnxSource( COnnxSourceHelper& source, std::initializer_list<int> expectedData ) const;
};

} // namespace optimization

} // namespace NeoOnnx
