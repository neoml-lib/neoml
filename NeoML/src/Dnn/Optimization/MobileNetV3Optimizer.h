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
#include <NeoML/Dnn/Layers/ActivationLayers.h>

namespace NeoML {

// Forward declaration(s)
class CBaseLayer;
class CChannelwiseConvLayer;
class CConvLayer;
struct CDnnOptimizationReport;
class COnnxSourceHelper;
class COnnxTransformHelper;
class COnnxTransposeHelper;
class CGlobalMeanPoolingLayer;

namespace optimization {

class CMobileNetV3Optimizer {
public:
	explicit CMobileNetV3Optimizer( CGraph& graph ) :
		graph( graph )
	{
	}

	// Optimizes the graph and writes the result to the report
	void Apply( CDnnOptimizationReport& report );

private:
	struct CMNv3BlockInfo {
		CLayerOutput<> InputData{};
		CConvLayer* ExpandConv = nullptr;
		CActivationDesc ExpandActivation{ AF_ReLU };
		CChannelwiseConvLayer* Channelwise = nullptr;
		CGlobalMeanPoolingLayer* SEPooling = nullptr;
		CBaseLayer* SEFirstFc = nullptr;
		CBaseLayer* SESecondActivation = nullptr;
		CLayerInput<> SEMulVectorInput{};
		CActivationDesc ChannelwiseActivation{ AF_ReLU };
		CConvLayer* DownConv = nullptr;
		CBaseLayer* Residual = nullptr;
	};

	CGraph& graph;

	int optimizeNonResidualBlocks();
	int optimizeResidualBlocks();

	bool detectMNv3Residual( CBaseLayer& residual, CMNv3BlockInfo& detectedBlock );
	bool detectMNv3NonResidual( CConvLayer& downConv, CMNv3BlockInfo& detectedBlock );
	bool detectMNv3PostSE( CConvLayer& downConv, CMNv3BlockInfo& detectedBlock );
	bool detectMNv3SE( CMNv3BlockInfo& detectedBlock );
	bool detectMNv3PreSE( CMNv3BlockInfo& detectedBlock );

	bool isValidSEMul( CBaseLayer& layer ) const;
	bool isValid1x1Conv( CBaseLayer* conv ) const;
	bool isValidBlockActivation( CBaseLayer& layer ) const;
	bool isValidSEActivation( CBaseLayer& layer ) const;
	bool isValidChannelwise( CChannelwiseConvLayer& channelwise ) const;

	void optimizeDetectedBlock( const CMNv3BlockInfo& detectedBlock );
};

} // namespace optimization

} // namespace NeoML
