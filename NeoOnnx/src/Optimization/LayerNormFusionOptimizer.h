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

#include <type_traits>
#include "Optimization/Graph.h"
#include <NeoML/Dnn/Layers/Onnx/OnnxTransformHelper.h>

namespace NeoOnnx {

namespace optimization {

//  Layer Normalization will fuse ObjectLayerNormalization into one layer:
// 
//  (x - mean(x, axis)) / sqrt(var(x, axis)) * scale + bias  , where 'x' is the input and var(x) = mean((x-mean)^2).
// 
//  +---------------------+
//  |                     |
//  |                     v
//  X --> ReduceMean --> Sub --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Mul --> Add
//                        |                                               ^
//                        |                                               |
//                        +-----------------------------------------------+
//  It also handles cases of duplicated sub layers exported from older version of PyTorch :
//  +---------------------+
//  |                     v
//  |          +-------> Sub ---------------------------------------------+
//  |          |                                                          |
//  |          |                                                          v
//  X --> ReduceMean --> Sub --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Mul --> Add
//  |                     ^
//  |                     |
//  +---------------------+
// 
//  In recent pytorch, Cast layers may be inserted before Pow to ensure that both inputs 'base' and 'power' are the same type
//  due to restriction in older opsets. Therefore, Layer Normalization will also handle the case below :
//  +---------------------+
//  |                     |
//  |                     v
//  X --> ReduceMean --> Sub --> Cast --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Mul --> Add
//                        |                                                        ^
//                        |                                                        |
//                        +--------------------------------------------------------+
//  +---------------------+       Cast
//  |                     |        |
//  |                     v        v
//  X --> ReduceMean --> Sub -->  Pow --> ReduceMean --> Add --> Sqrt --> Div --> Mul --> Add
//                        |                                                ^
//                        |                                                |
//                        +------------------------------------------------+
// 
//  When using Apex O2, a Cast layer may be inserted between Div and Mul, Layer Normalization will also handle the case below:
//  +---------------------+
//  |                     |
//  |                     v
//  X --> ReduceMean --> Sub --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Cast --> Mul --> Add
//                        |                                               ^
//                        |                                               |
//                        +-----------------------------------------------+
//  OR
//           +---------------------+
//           |                     |
//           |                     v
//  X --> Cast --> ReduceMean --> Sub --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Cast --> Mul --> Add
//                                 |                                               ^
//                                 |                                               |
//                                 +-----------------------------------------------+
//  Logically since LayerNormalization supports input and scale/bias in different data types, and during the kernel execution,
//  data are casted to float/double to calculate for precision, so if there is any Cast Ops in the sub-graph, we can remove it.
//  Such Cast Op can be the input of the sub-graph, or an Cast Op between the Div and Mul layers.
class CLayerNormFusionOptimizer final {
public:
	explicit CLayerNormFusionOptimizer( CGraph& graph ) : graph(graph) {}
	CLayerNormFusionOptimizer( const CLayerNormFusionOptimizer& ) = delete;
	CLayerNormFusionOptimizer( CLayerNormFusionOptimizer&& ) = delete;

	void Apply();

private:
	CGraph& graph;
	static constexpr const char* const FusionNamePrefix{ "NormFusion_" };

	// Checks if DataLayer is valid for CLayerNormFusionOptimizer conversion
	bool isValidDataLayer( const CDataLayer& dataLayer, TBlobType blobType, int blobSize = NotFound ) const;
	// Checks if TransformHelperLayer is valid for CLayerNormFusionOptimizer conversion
	bool isValidTransformLayer( const COnnxTransformHelper& transformHelperLayer ) const;

	// Get typed pointer to current 'layer' in the 'graph' (only if its typ is 'TLayer'), else returns nullptr.
	// If 'addToSelectedLayers' is true, also it adds this layer to 'graph.selection'.
	template<typename TLayer,
		typename std::enable_if<std::is_base_of<CBaseLayer, TLayer>::value, int>::type = 0>
	TLayer* getExactLayer( CBaseLayer* layer, bool addToSelectedLayers = true );

	// Select 1 exact input-layer of the 'currentLayer' in the 'graph' (only if its type is 'TInputLayer'), else returns nullptr.
	// NOTE: If returned layer is not nullptr, it has been added to 'graph.selection'.
	//       'COnnxTransformHelper' may be selected and skipped recursively, its input layers types would be checked and returned.
	template<typename TInputLayer,
		typename std::enable_if<std::is_base_of<CBaseLayer, TInputLayer>::value, int>::type = 0>
	TInputLayer* selectOneExactInputLayerRecursive( const CBaseLayer& currentLayer );

	// Select 2 exact input-layers of the 'currentLayer' in the 'graph' (only if their types are 'TFirstInputLayer' and 'TSecondInputLayer'),
	// then returns true ('layerBase' and 'layerData' != nullptr), else false ('layerBase' or 'layerData' == nullptr).
	// NOTE: If 'layerBase' or 'layerData' is not nullptr, if its layer type != 'CBaseLayer', it would be added to 'graph.selection'
	//       'COnnxTransformHelper' may be selected and skipped recursively, its input layers types would be checked and returned.
	template<typename TFirstInputLayer, typename TSecondInputLayer,
		typename std::enable_if<
			std::is_base_of<CBaseLayer, TFirstInputLayer>::value &&
			std::is_base_of<CBaseLayer, TSecondInputLayer>::value,
			int>::type = 0>
	bool selectTwoExactInputLayersRecursive( const CBaseLayer& currentLayer, TFirstInputLayer** layerBase, TSecondInputLayer** layerData );
};

//--------------------------------------------------------------------------------------------------------------
template<typename TLayer,
	typename std::enable_if<std::is_base_of<CBaseLayer, TLayer>::value, int>::type>
TLayer* CLayerNormFusionOptimizer::getExactLayer( CBaseLayer* layer, bool addToSelection )
{
	auto* typedLayer = dynamic_cast<TLayer*>( layer );
	if( addToSelection ) {
		if( typedLayer != nullptr ) {
			if( !graph.IsLayerSelected( *layer ) ) {
				graph.SelectLayer( *layer );
			}
			return typedLayer;
		}
		return nullptr;
	}
	return typedLayer;
}

//--------------------------------------------------------------------------------------------------------------
template<typename TInputLayer,
	typename std::enable_if<std::is_base_of<CBaseLayer, TInputLayer>::value, int>::type>
TInputLayer* CLayerNormFusionOptimizer::selectOneExactInputLayerRecursive( const CBaseLayer& currentLayer )
{
	static_assert(!std::is_same<TInputLayer, CBaseLayer>::value,
		"Wrong 'TInputLayer' CLayerNormFusionOptimizer::selectOneExactInputLayerRecursive");

	if( currentLayer.GetInputCount() != 1 ) {
		return nullptr;
	}
	auto* layer = graph.GetConnectedOutput( currentLayer, /*inputIndex*/0 ).Layer;
	auto* inputLayer = getExactLayer<TInputLayer>( layer, /*addToSelection*/true );
	if( inputLayer != nullptr ) {
		return inputLayer;
	}
	auto* transformLayer = getExactLayer<COnnxTransformHelper>( layer, /*addToSelection*/true );
	if( transformLayer != nullptr && isValidTransformLayer( *transformLayer ) ) {
		return selectOneExactInputLayerRecursive<TInputLayer>( *transformLayer );
	}
	return nullptr;
}

//--------------------------------------------------------------------------------------------------------------
template<typename TFirstInputLayer, typename TSecondInputLayer,
	typename std::enable_if<
		std::is_base_of<CBaseLayer, TFirstInputLayer>::value &&
		std::is_base_of<CBaseLayer, TSecondInputLayer>::value,
		int>::type>
bool CLayerNormFusionOptimizer::selectTwoExactInputLayersRecursive( const CBaseLayer& currentLayer, TFirstInputLayer** layerBase, TSecondInputLayer** layerData )
{
	static_assert( !std::is_same<TFirstInputLayer, CBaseLayer>::value,
		"Wrong 'TFirstInputLayer' CLayerNormFusionOptimizer::selectTwoExactInputLayersRecursive" );

	if( currentLayer.GetInputCount() > 2 ) { // also should fit for COnnxTransformHelper, it has 1 input
		return false;
	}
	for( int inputIndex = 0; inputIndex < currentLayer.GetInputCount(); ++inputIndex ) {
		auto* layer = graph.GetConnectedOutput( currentLayer, inputIndex ).Layer;

		if( *layerBase == nullptr && ( *layerBase = getExactLayer<TFirstInputLayer>( layer, /*addToSelection*/true ) ) != nullptr ) {
			continue;
		}
		if( *layerData == nullptr && ( *layerData = getExactLayer<TSecondInputLayer>( layer,
			/*addToSelection*/!std::is_same<TSecondInputLayer, CBaseLayer>::value /*CBaseLayer will not be selected*/) ) != nullptr ) {
			continue;
		}

		if( *layerBase == nullptr || *layerData == nullptr ) {
			auto* transformLayer = getExactLayer<COnnxTransformHelper>( layer, /*addToSelection*/true );
			if( transformLayer != nullptr && isValidTransformLayer( *transformLayer ) ) {
				return selectTwoExactInputLayersRecursive<TFirstInputLayer, TSecondInputLayer>( *transformLayer, layerBase, layerData );
			}
		}
	}
	return ( *layerBase != nullptr && *layerData != nullptr );
}

} // namespace optimization

} // namespace NeoOnnx

