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

#include "common.h"
#pragma hdrstop

#include <cmath>

#include "HardSigmoidOptimizer.h"
#include <NeoML/Dnn/Optimization/Graph.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxEltwiseLayer.h>

namespace NeoOnnx {

namespace optimization {

using CGraph = NeoML::optimization::CGraph;
template<typename TLayer = CBaseLayer>
using CLayerInput = NeoML::optimization::CLayerInput<TLayer>;
template<typename TLayer = CBaseLayer>
using CLayerOutput = NeoML::optimization::CLayerOutput<TLayer>;

void CHardSigmoidOptimizer::Apply()
{
	CArray<CBaseLayer*> layers;
	graph.GetLayers( layers );

	for( CBaseLayer* layer : layers ) {
		graph.ClearSelection();

		if( !graph.HasLayer( layer ) ) {
			// There is a risk that layer has already been deleted
			// and 'layer' points to an invalid object
			continue;
		}

		// Find the last mul/div layer
		COnnxEltwiseLayer* slopeLayer = dynamic_cast<COnnxEltwiseLayer*>( layer );
		if( slopeLayer == nullptr || graph.GetInputCount( *slopeLayer ) != 2 ||
			( slopeLayer->GetOperation() != COnnxEltwiseLayer::TOperation::Mul
				&& slopeLayer->GetOperation() != COnnxEltwiseLayer::TOperation::Div ) )
		{
			continue;
		}
		graph.SelectLayer( *slopeLayer );

		// Check slopeLayer's inputs
		CLayerOutput<CDataLayer> slopeDataOutput;
		CLayerOutput<CReLULayer> clipOutput;
		if( !graph.SelectBothConnectedOutputs( *slopeLayer, slopeDataOutput, clipOutput, true ) ) {
			continue;
		}

		// Extract slope coefficient
		float slopeValue = 0;
		if( !isValidDataLayer( *slopeDataOutput.Layer, slopeValue ) ) {
			continue;
		}
		if( slopeLayer->GetOperation() == COnnxEltwiseLayer::TOperation::Div ) {
			slopeValue = 1.f / slopeValue;
		}
		if( std::abs( clipOutput.Layer->GetUpperThreshold() * slopeValue - 1.f ) > 1e-4f ) {
			// Hard sigmoid can only return values in [0;1]
			continue;
		}

		// Find bias layer
		COnnxEltwiseLayer* bias = graph.SelectConnectedOutput<COnnxEltwiseLayer>( *clipOutput.Layer, 0, true ).Layer;
		if( bias == nullptr || graph.GetInputCount( *bias ) != 2
			|| ( bias->GetOperation() != COnnxEltwiseLayer::TOperation::Add
				&& bias->GetOperation() != COnnxEltwiseLayer::TOperation::Sub ) )
		{
			continue;
		}

		// Check bias layer inputs
		CLayerOutput<CDataLayer> biasDataOutput;
		CLayerOutput<> hardSigmoidInputData;
		float biasValue = 0.f;
		for( int i = 0; i < 2; ++i ) {
			biasDataOutput = graph.GetConnectedOutput<CDataLayer>( *bias, i );
			if( biasDataOutput.Layer != nullptr && isValidDataLayer( *biasDataOutput.Layer, biasValue ) ) {
				hardSigmoidInputData = graph.GetConnectedOutput<>( *bias, 1 - i );
				graph.SelectLayer( *biasDataOutput.Layer );
				break;
			} else {
				biasDataOutput.Layer = nullptr;
			}
		}

		if( biasDataOutput.Layer == nullptr ) {
			continue;
		}

		// Hard sigmoid firstly applies slope, then bias
		if( bias->GetOperation() == COnnxEltwiseLayer::TOperation::Sub ) {
			biasValue = -biasValue;
		}
		biasValue *= slopeValue;

		CPtr<CHardSigmoidLayer> hardSigmoidLayer = new CHardSigmoidLayer( graph.MathEngine() );
		hardSigmoidLayer->SetName( graph.GetUniqueName( "HardSigmoid" ) );
		hardSigmoidLayer->SetSlope( slopeValue );
		hardSigmoidLayer->SetBias( biasValue );
		graph.AddLayer( *hardSigmoidLayer );
		::printf( "[HARDSIGMOID] replace '%s' with '%s'\n", slopeLayer->GetName(), hardSigmoidLayer->GetName() );

		graph.Connect( *hardSigmoidLayer, 0, *hardSigmoidInputData.Layer, hardSigmoidInputData.Index );
		graph.SwitchOutputs( *slopeLayer, 0, *hardSigmoidLayer, 0 );

		graph.DeleteSelectedLayers();
	}

	graph.ClearSelection();
}

// Checks if data layer is valid for CHardSigmoid conversion
bool CHardSigmoidOptimizer::isValidDataLayer( CDataLayer& dataLayer, float& value ) const
{
	NeoAssert( graph.GetInputCount( dataLayer ) == 0 );
	NeoAssert( graph.GetOutputCount( dataLayer ) == 1 );

	if( graph.GetConnectedInputsCount( dataLayer, 0 ) != 1 ) {
		return false;
	}

	CPtr<CDnnBlob> valueBlob = dataLayer.GetBlob();
	if( valueBlob->GetDataType() != CT_Float || valueBlob->GetDataSize() != 1 ) {
		return false;
	}

	value = valueBlob->GetData().GetValue();
	return true;
}

} // namespace optimization

} // namespace NeoOnnx
