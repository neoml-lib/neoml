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

void CLayerOperator::GetOutputTensors( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	if( !canCalculateOutput( inputs ) ) {
		AddLayers( inputs, dnn, outputs );
		return;
	}

	CRandom random( 0x1231 );
	CDnn internalDnn( random, dnn.GetMathEngine() );

	// Add source layers for the operator
	CTensorArray internalInputs;
	addInternalDnnSources( inputs, internalInputs, internalDnn );

	// Add operator layers
	CTensorArray internalOutputs;
	internalOutputs.Add( nullptr, OutputCount() );
	AddLayers( internalInputs, internalDnn, internalOutputs );

	// Add sink layers for the operator
	CArray<CSinkLayer*> sinks;
	addInternalDnnSinks( internalOutputs, sinks, internalDnn );

	// Launch the dnn in order to calculate values
	internalDnn.RunOnce();

	// Extract values from the net
	extractOutputs( internalOutputs, sinks, outputs );
}

// Returns true if output tensors' data can be calculated during import
bool CLayerOperator::canCalculateOutput( const CTensorArray& inputs ) const
{
	for( int inputIndex = 0; inputIndex < inputs.Size(); ++inputIndex ) {
		if( inputs[inputIndex] != nullptr && !inputs[inputIndex]->IsCalculated() ) {
			return false;
		}
	}

	return true;
}

// Builds array of tensors related to the internal dnn
// Also adds required source layers to the internal dnn (with corresponding blobs)
void CLayerOperator::addInternalDnnSources( const CTensorArray& inputs,
	CTensorArray& internalInputs, CDnn& internalDnn ) const
{
	IMathEngine& mathEngine = internalDnn.GetMathEngine();

	CUserInputMask isUserInput;
	UserInputMask( isUserInput );

	for( int inputIndex = 0; inputIndex < InputCount(); ++inputIndex ) {
		if( inputs[inputIndex] == nullptr || !inputs[inputIndex]->IsCalculated() ) {
			internalInputs.Add( nullptr );
		} else if( isUserInput[inputIndex] ) {
			NeoAssert( inputs[inputIndex]->IsCalculated() );
			CPtr<CSourceLayer> source = new CSourceLayer( mathEngine );
			source->SetName( InputName( inputIndex ) );
			internalDnn.AddLayer( *source );
			source->SetBlob( dynamic_cast<const CDataTensor*>( inputs[inputIndex].Ptr() )->Data()->GetCopy() );
			internalInputs.Add( new CUserTensor( inputs[inputIndex]->Shape(), inputs[inputIndex]->Layout(),
				CLayerOutput( source, 0 ) ) );
		} else {
			internalInputs.Add( inputs[inputIndex] );
		}
	}
}

// Builds array of sinks (corresponding to the operator outputs)
// Also adds those layers to the dnn
void CLayerOperator::addInternalDnnSinks( const CTensorArray& internalOutputs,
	CArray<CSinkLayer*>& sinks, CDnn& internalDnn ) const
{
	IMathEngine& mathEngine = internalDnn.GetMathEngine();

	for( int outputIndex = 0; outputIndex < OutputCount(); ++outputIndex ) {
		if( internalOutputs[outputIndex] == nullptr || internalOutputs[outputIndex]->IsCalculated() ) {
			sinks.Add( nullptr );
		} else {
			CPtr<CSinkLayer> sink = new CSinkLayer( mathEngine );
			sink->SetName( OutputName( outputIndex ) + "_Sink" );
			internalDnn.AddLayer( *sink );
			const CLayerOutput& connectedOutput = dynamic_cast<const CUserTensor*>( internalOutputs[outputIndex].Ptr() )->LayerOutput();
			sink->Connect( 0, *connectedOutput.Layer, connectedOutput.OutputIndex );
			sinks.Add( sink.Ptr() );
		}
	}
}

// Builds array of the operator outputs based on outputs of the internal dnn
void CLayerOperator::extractOutputs( const CTensorArray& internalOutputs, const CArray<CSinkLayer*>& sinks,
	CTensorArray& outputs ) const
{
	for( int outputIndex = 0; outputIndex < OutputCount(); ++outputIndex ) {
		if( internalOutputs[outputIndex]->IsCalculated() ) {
			// This data was calculated prior to the net
			outputs[outputIndex] = internalOutputs[outputIndex];
		} else if( sinks[outputIndex] != nullptr ) {
			// Add network result as data tensor
			// Shape and layout remain unchanged
			outputs[outputIndex] = new CDataTensor( internalOutputs[outputIndex]->Shape(),
				internalOutputs[outputIndex]->Layout(), *( sinks[outputIndex]->GetBlob() ) );
		}
		// otherwise leave internalOutputs[outputIndex] as nullptr
	}
}

} // namespace NeoOnnx
