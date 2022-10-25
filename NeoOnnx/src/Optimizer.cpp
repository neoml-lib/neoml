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

#include "Optimizer.h"

namespace NeoOnnx {

CPtr<CBaseLayer> IOptimizer::GetAnyInputLayer( const CPtr<CBaseLayer>& currentLayer, int i, const char* const layerSkipClass )
{
	for( int j = i; j < currentLayer->GetInputCount(); ++j ) {
		CPtr<CBaseLayer> inputLayer = graph.GetLayer( currentLayer->GetInputName( j ) );
		if( IsExactLayer( inputLayer, classesOfSkipLayers[0] ) ) {
			return graph.GetLayer( inputLayer->GetInputName( j ) );
		} else if( layerSkipClass != CString( "" ) && IsExactLayer( inputLayer, layerSkipClass ) ) {
			return graph.GetLayer( inputLayer->GetInputName( /*inputNumber*/0 ) );
		}
		return inputLayer;
	}
	return nullptr;
}

//--------------------------------------------------------------------------------------------------------------
bool IOptimizer::IsExactLayer( const CPtr<CBaseLayer>& layer, const char* layerClass, bool justCheck )
{
	if( layer != nullptr && layer->GetClassType() == layerClass ) {
		if( !justCheck ) {
			layersToRemove.Add( layer );
		}
		return true;
	}
	return false;
}

//--------------------------------------------------------------------------------------------------------------
bool IOptimizer::GetExactInputLayers( const CPtr<CBaseLayer>& currentLayer,
	CPtr<CBaseLayer>& layerBase, const char* const layerBaseClass,
	CPtr<CBaseLayer>& layerData, const char* const layerDataClass, const char* const layerSkipClass )
{
	for( int j = 0; j < currentLayer->GetInputCount(); ++j ) {
		CPtr<CBaseLayer> inputLayer = graph.GetLayer( currentLayer->GetInputName( j ) );
		if( IsExactLayer( inputLayer, layerBaseClass ) ) {
			layerBase = inputLayer;
		} else if( layerBase == nullptr && IsExactLayer( inputLayer, layerSkipClass ) ) {
			GetExactInputLayers( inputLayer, layerBase, layerBaseClass, layerData, layerDataClass, /*layerSkipClass*/"" );
		} else if( ( layerBase == nullptr || layerData == nullptr )
			&& ( IsExactLayer( inputLayer, classesOfSkipLayers[0] )
			|| IsExactLayer( inputLayer, classesOfSkipLayers[1] )
			|| IsExactLayer( inputLayer, classesOfSkipLayers[2] ) ) )
		{
			GetExactInputLayers( inputLayer, layerBase, layerBaseClass, layerData, layerDataClass, layerSkipClass );
		} else if( IsExactLayer( inputLayer, layerDataClass ) ) {
			layerData = inputLayer;
		}
	}
	return ( layerBase != nullptr
		&& ( layerDataClass == CString( "" ) || layerData != nullptr ) ); // skip check initializer type or found
}

} // namespace NeoOnnx

