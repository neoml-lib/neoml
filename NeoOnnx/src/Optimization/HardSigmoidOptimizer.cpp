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

namespace NeoOnnx {

void CHardSigmoidOptimizer::Apply()
{
	CArray<CBaseLayer*> layers;
	graph.GetLayers( layers );

	for( CBaseLayer* layer : layers ) {
		if( !graph.HasLayer( layer ) ) {
			// There is a risk that layer has already been deleted
			// and 'layer' points to an invalid object
			continue;
		}

		// Find the last mul/div layer
		COnnxEltwiseLayer* slopeLayer = dynamic_cast<COnnxEltwiseLayer*>( layer );
		if( slopeLayer == nullptr ) {
			continue;
		}

		// Check slopeLayer's inputs
		CDataLayer* slopeDataLayer = nullptr;
		CReLULayer* clipLayer = nullptr;
		float slopeValue = 0;
		if( !isValidSlopeLayer( *slopeLayer, slopeDataLayer, clipLayer, slopeValue ) ) {
			continue;
		}

		// Check clip layer and find bias layer
		COnnxEltwiseLayer* biasLayer = nullptr;
		float clipThreshold = 0;
		if( !isValidClipLayer( *clipLayer, biasLayer, clipThreshold ) ) {
			continue;
		}
		if( std::fabsf( clipThreshold * slopeValue - 1.f ) > 1e-4 ) {
			// Hard sigmoid can only return values in [0;1]
			continue;
		}

		// Check bias layer inputs
		CDataLayer* biasDataLayer = nullptr;
		CDnnGraphLink hardSigmoidInput;
		float biasValue = 0;
		if( !isValidBiasLayer( *biasLayer, biasDataLayer, hardSigmoidInput, biasValue ) ) {
			continue;
		}
		// Hard sigmoid firstly applies slope, then bias
		biasValue *= slopeValue;

		CPtr<CHardSigmoidLayer> hardSigmoid = new CHardSigmoidLayer( graph.MathEngine() );
		hardSigmoid->SetName( graph.GetUniqueName( "HardSigmoid" ) );
		hardSigmoid->SetSlope( slopeValue );
		hardSigmoid->SetBias( biasValue );
		graph.AddLayer( *hardSigmoid );

		graph.Connect( { hardSigmoid, 0 }, hardSigmoidInput );
		graph.SwitchOutputs( { slopeLayer, 0 }, { hardSigmoid, 0 } );

		graph.DeleteLayer( *slopeLayer );
		graph.DeleteLayer( *slopeDataLayer );
		graph.DeleteLayer( *clipLayer );
		graph.DeleteLayer( *biasLayer );
		graph.DeleteLayer( *biasDataLayer );
	}
}

// Checks if slope layer is valid for CHardSigmoid conversion
bool CHardSigmoidOptimizer::isValidSlopeLayer( const COnnxEltwiseLayer& slopeLayer,
	CDataLayer*& slopeDataLayer, CReLULayer*& clipLayer, float& slopeValue ) const
{
	// Eltwise layer always has 1 output
	NeoAssert( graph.GetOutputCount( slopeLayer ) == 1 );

	if( slopeLayer.GetOperation() != COnnxEltwiseLayer::TOperation::Mul
		&& slopeLayer.GetOperation() != COnnxEltwiseLayer::TOperation::Div )
	{
		return false;
	}

	if( graph.GetInputCount( slopeLayer ) != 2 ) {
		return false;
	}

	slopeDataLayer = nullptr;
	clipLayer = nullptr;
	for( int i = 0; i < 2; ++i ) {
		const CDnnGraphLink& input = graph.GetInputLink( slopeLayer, i );
		if( input.Index != 0 ) {
			// slope layer must be connected to the only output of its inputs
			return false;
		}

		CDataLayer* dataLayerCandidate = dynamic_cast<CDataLayer*>( input.Layer );
		if( dataLayerCandidate != nullptr && isValidDataLayer( *dataLayerCandidate, slopeValue ) ) {
			slopeDataLayer = dataLayerCandidate;
		} else if( dynamic_cast<CReLULayer*>( input.Layer ) != nullptr ) {
			clipLayer = dynamic_cast<CReLULayer*>( input.Layer );
		} else {
			return false;
		}
	}

	if( slopeDataLayer == nullptr || clipLayer == nullptr ) {
		return false;
	}

	if( slopeLayer.GetOperation() == COnnxEltwiseLayer::TOperation::Div ) {
		slopeValue = 1.f / slopeValue;
	}
	
	return true;
}

// Checks if clip layer is valid for CHardSigmoid conversion
bool CHardSigmoidOptimizer::isValidClipLayer( const CReLULayer& clipLayer, COnnxEltwiseLayer*& biasLayer,
	float& clipThreshold ) const
{
	// ReLU layer always has 1 input and 1 output
	NeoAssert( graph.GetInputCount( clipLayer ) == 1 );
	NeoAssert( graph.GetOutputCount( clipLayer ) == 1 );

	// If ReLU is used by some other layer then we can't replace it with CHardSigmoid
	if( graph.GetOutputLinkCount( clipLayer, 0 ) != 1 ) {
		return false;
	}

	const CDnnGraphLink& biasOutput = graph.GetInputLink( clipLayer, 0 );
	if( biasOutput.Index != 0 ) {
		return false;
	}

	biasLayer = dynamic_cast<COnnxEltwiseLayer*>( biasOutput.Layer );
	if( biasLayer == nullptr ) {
		return false;
	}

	clipThreshold = clipLayer.GetUpperThreshold();
	// CHardSigmoid must have upper threshold
	return clipThreshold > 0;
}

// Checks if bias layer is valid for CHardSigmoid conversion
bool CHardSigmoidOptimizer::isValidBiasLayer( const COnnxEltwiseLayer& biasLayer, CDataLayer*& biasDataLayer,
	CDnnGraphLink& hardSigmoidInput, float& biasValue ) const
{
	// Eltwise layer always has 1 output
	NeoAssert( graph.GetOutputCount( biasLayer ) == 1 );

	if( biasLayer.GetOperation() != COnnxEltwiseLayer::TOperation::Add
		&& biasLayer.GetOperation() != COnnxEltwiseLayer::TOperation::Sub )
	{
		return false;
	}

	if( graph.GetInputCount( biasLayer ) != 2 ) {
		return false;
	}

	if( graph.GetOutputLinkCount( biasLayer, 0 ) != 1 ) {
		// Its output is used by some other layer
		return false;
	}

	biasDataLayer = nullptr;
	for( int i = 0; i < 2; ++i ) {
		const CDnnGraphLink& currInput = graph.GetInputLink( biasLayer, i );
		CDataLayer* dataLayerCandidate = dynamic_cast<CDataLayer*>( currInput.Layer );
		if( dataLayerCandidate != nullptr && isValidDataLayer( *dataLayerCandidate, biasValue ) ) {
			biasDataLayer = dataLayerCandidate;
			hardSigmoidInput = graph.GetInputLink( biasLayer, 1 - i );
			break;
		}
	}

	if( biasDataLayer == nullptr ) {
		return false;
	}

	if( biasLayer.GetOperation() == COnnxEltwiseLayer::TOperation::Sub ) {
		biasValue = -biasValue;
	}

	return true;
}

// Checks if data layer is valid for CHardSigmoid conversion
bool CHardSigmoidOptimizer::isValidDataLayer( const CDataLayer& dataLayer, float& value ) const
{
	NeoAssert( graph.GetInputCount( dataLayer ) == 0 );
	NeoAssert( graph.GetOutputCount( dataLayer ) == 1 );
	if( graph.GetOutputLinkCount( dataLayer, 0 ) != 1 ) {
		// This data layer is used by other layers, can't replace
		return false;
	}

	CPtr<CDnnBlob> valueBlob = dataLayer.GetBlob();
	if( valueBlob->GetDataType() != CT_Float || valueBlob->GetDataSize() != 1 ) {
		return false;
	}

	value = valueBlob->GetData().GetValue();
	return true;
}

} // namespace NeoOnnx
