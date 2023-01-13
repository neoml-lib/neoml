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

#include <common.h>
#pragma hdrstop

#include <NeoML/Dnn/Layers/Onnx/OnnxConcatLayer.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxSliceLayer.h>

namespace NeoML {

static const int OnnxConcatLayerVersion = 0;

void COnnxConcatLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( OnnxConcatLayerVersion );
	CBaseReshaper::Serialize( archive );
	archive.SerializeEnum( concatDim );
}

void COnnxConcatLayer::CalculateShapes()
{
	CheckArchitecture( GetInputCount() >= 1, GetPath(), "Layer must have inputs" );
	CheckArchitecture( GetOutputCount() == 1, GetPath(), "Layer must have 1 output" );

	if( inputShapeBlobs[0] == nullptr ) {
		CBlobDesc outputDesc = inputDescs[0];
		outputDesc.SetDimSize( concatDim, 0 );
		for( int i = 0; i < GetInputCount(); ++i ) {
			if( inputHasElements( i ) ) {
				outputDesc.SetDimSize( concatDim,
					outputDesc.DimSize( concatDim ) + inputDescs[i].DimSize( concatDim ) );
			}
		}
		outputDescs[0] = outputDesc;
		return;
	}

	CBlobDesc outputDesc = inputShapeBlobs[0]->GetDesc();
	outputDesc.SetDimSize( concatDim, 0 );
	for( int i = 0; i < GetInputCount(); ++i ) {
		if( inputHasElements( i ) ) {
			outputDesc.SetDimSize( concatDim,
				outputDesc.DimSize( concatDim ) + inputShapeBlobs[i]->DimSize( concatDim ) );
		}
	}
	outputShapeBlobs[0] = CDnnBlob::CreateBlob( inputShapeBlobs[0]->GetMathEngine(),
		outputDesc.GetDataType(), outputDesc );
	calcOutput( inputShapeBlobs, outputShapeBlobs[0] );
}

void COnnxConcatLayer::RunOnce()
{
	if( inputShapeBlobs[0] != nullptr ) {
		return;
	}

	calcOutput( inputBlobs, outputBlobs[0] );
}

bool COnnxConcatLayer::inputHasElements( int index ) const
{
	NeoPresume( index >= 0 );
	NeoPresume( index < GetInputCount() );

	const COnnxSliceLayer* slice = dynamic_cast<const COnnxSliceLayer*>( GetInputLayer( index ) );
	if( slice == nullptr ) {
		return true;
	}

	return slice->DoesOutputHaveElements();
}

void COnnxConcatLayer::calcOutput( const CObjectArray<CDnnBlob>& inputs, const CPtr<CDnnBlob>& output )
{
	CObjectArray<CDnnBlob> filteredInputs;
	for( int i = 0; i < inputs.Size(); ++i ) {
		if( inputHasElements( i ) ) {
			filteredInputs.Add( inputs[i] );
		}
	}

	CDnnBlob::MergeByDim( output->GetMathEngine(), concatDim, filteredInputs, output );
}

} // namespace NeoML
