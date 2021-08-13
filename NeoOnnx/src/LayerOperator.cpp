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

#include "common.h"
#pragma hdrstop

#include "LayerOperator.h"

namespace NeoOnnx {

// Returns true if some of the inputs are depending on user data
static bool hasUserInputs( const CTensorArray& inputs )
{
	for( int inputIndex = 0; inputIndex < inputs.Size(); ++inputIndex ) {
		if( inputs[inputIndex] != nullptr && !inputs[inputIndex]->IsCalculated() ) {
			return true;
		}
	}

	return false;
}

// Builds an array of sinks (corresponding to the operator outputs)
// Also adds those layers to the dnn
static void addInternalDnnSinks( const CTensorArray& internalOutputs,
	CArray<CSinkLayer*>& sinks, CDnn& internalDnn )
{
	IMathEngine& mathEngine = internalDnn.GetMathEngine();

	for( int outputIndex = 0; outputIndex < internalOutputs.Size(); ++outputIndex ) {
		if( internalOutputs[outputIndex] == nullptr || internalOutputs[outputIndex]->IsCalculated() ) {
			sinks.Add( nullptr );
		} else {
			CPtr<CSinkLayer> sink = new CSinkLayer( mathEngine );
			sink->SetName( Str( internalDnn.GetLayerCount() ) );
			internalDnn.AddLayer( *sink );
			const CLayerOutput& connectedOutput = dynamic_cast<const CUserTensor*>( internalOutputs[outputIndex].Ptr() )->LayerOutput();
			sink->Connect( 0, *connectedOutput.Layer, connectedOutput.OutputIndex );
			sinks.Add( sink.Ptr() );
		}
	}
}

// Builds array of the operator outputs based on outputs of the internal dnn
static void extractOutputs( const CTensorArray& internalOutputs, const CArray<CSinkLayer*>& sinks,
	CTensorArray& outputs )
{
	for( int outputIndex = 0; outputIndex < internalOutputs.Size(); ++outputIndex ) {
		if( internalOutputs[outputIndex] != nullptr && internalOutputs[outputIndex]->IsCalculated() ) {
			// This data was calculated prior to the net
			outputs.Add( internalOutputs[outputIndex] );
		} else if( sinks[outputIndex] != nullptr ) {
			// Add network result as data tensor
			// Shape and layout remain unchanged
			outputs.Add( new CDataTensor( internalOutputs[outputIndex]->Shape(),
				internalOutputs[outputIndex]->Layout(), *( sinks[outputIndex]->GetBlob() ) ) );
		} else {
			// otherwise leave internalOutputs[outputIndex] as nullptr
			outputs.Add( nullptr );
		}
	}
}

// --------------------------------------------------------------------------------------------------------------------

void CLayerOperator::ProcessTensors( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	if( hasUserInputs( inputs ) ) {
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

