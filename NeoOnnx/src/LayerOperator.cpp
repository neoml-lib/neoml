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

#include "common.h"
#pragma hdrstop

#include "LayerOperator.h"

using namespace NeoML;

namespace NeoOnnx {

// Returns true if some of the inputs are CUserTensor
static bool hasUserOrShapeInputs( const CTensorArray& inputs )
{
	static_assert( static_cast<int>( TTensorType::Count ) == 3, "TTensorType::Count != 3" );
	for( int inputIndex = 0; inputIndex < inputs.Size(); ++inputIndex ) {
		if( inputs[inputIndex] != nullptr && inputs[inputIndex]->Type() != TTensorType::Data ) {
			return true;
		}
	}

	return false;
}

// Returns true if tensor has elements
static bool tensorHasElements( const CTensorBase& tensor )
{
	// The only scenario when tensor has no elements is CShapeTensor with one of dimensions equal to 0
	if( tensor.Type() != TTensorType::Shape ) {
		return true;
	}

	const CShapeTensor& shapeTensor = dynamic_cast<const CShapeTensor&>( tensor );
	for( int i = 0; i < shapeTensor.DimCount(); ++i ) {
		if( shapeTensor.Shape()[i] == 0 ) {
			return false;
		}
	}

	return true;
}

// Builds an array of sinks (corresponding to the operator outputs)
// Also adds those layers to the dnn
static void addInternalDnnSinks( const CTensorArray& internalOutputs,
	CArray<CSinkLayer*>& sinks, CDnn& internalDnn )
{
	IMathEngine& mathEngine = internalDnn.GetMathEngine();

	for( int outputIndex = 0; outputIndex < internalOutputs.Size(); ++outputIndex ) {
		if( internalOutputs[outputIndex] == nullptr || internalOutputs[outputIndex]->Type() == TTensorType::Data ) {
			sinks.Add( nullptr );
		} else {
			// internalOutputs[outputIndex] is a CShapeTensor or CUserTensor
			CPtr<const CUserTensor> userOutput = AsUserTensor( *internalOutputs[outputIndex],
				Str( internalDnn.GetLayerCount() ), internalDnn );
			CPtr<CSinkLayer> sink = new CSinkLayer( mathEngine );
			sink->SetName( Str( internalDnn.GetLayerCount() ) );
			internalDnn.AddLayer( *sink );
			sink->Connect( 0, *userOutput->Layer(), userOutput->OutputIndex() );
			if( tensorHasElements( *internalOutputs[outputIndex] ) ) {
				sinks.Add( sink.Ptr() );
			} else {
				// Let this sink be in order to avoid hanging layers
				// But don't register it (the resulting tensor will be nullptr)
				sinks.Add( nullptr );
			}
		}
	}
}

// Builds array of the operator outputs based on outputs of the internal dnn
static void extractOutputs( const CTensorArray& internalOutputs, const CArray<CSinkLayer*>& sinks,
	CTensorArray& outputs )
{
	for( int outputIndex = 0; outputIndex < internalOutputs.Size(); ++outputIndex ) {
		if( internalOutputs[outputIndex] != nullptr && internalOutputs[outputIndex]->Type() == TTensorType::Data ) {
			// This data was calculated prior to the net
			outputs.Add( internalOutputs[outputIndex] );
		} else if( sinks[outputIndex] != nullptr ) {
			// Add network result as data tensor
			// Shape and layout remain unchanged
			outputs.Add( new CDataTensor( internalOutputs[outputIndex]->Layout(),
				*( sinks[outputIndex]->GetBlob() ) ) );
		} else {
			// otherwise leave internalOutputs[outputIndex] as nullptr
			outputs.Add( nullptr );
		}
	}
}

// --------------------------------------------------------------------------------------------------------------------

void CLayerOperator::ProcessTensors( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	if( hasUserOrShapeInputs( inputs ) ) {
		AddLayers( inputs, dnn, outputs );
		return;
	}

	CRandom random( 0x1231 );
	CDnn internalDnn( random, dnn.GetMathEngine() );

	// Add operator layers
	CTensorArray internalOutputs;
	internalOutputs.SetBufferSize( OutputCount() );
	AddLayers( inputs, internalDnn, internalOutputs );
	NeoAssert( internalOutputs.Size() == OutputCount() );

	// Add sink layers for the operator
	CArray<CSinkLayer*> sinks;
	addInternalDnnSinks( internalOutputs, sinks, internalDnn );

	// Launch the dnn in order to calculate values
	internalDnn.RunOnce();

	// Extract values from the net
	extractOutputs( internalOutputs, sinks, outputs );
}

} // namespace NeoOnnx

