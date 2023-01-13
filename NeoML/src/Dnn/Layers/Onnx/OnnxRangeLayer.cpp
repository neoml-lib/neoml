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

#include <cmath>

#include <NeoML/Dnn/Layers/Onnx/OnnxRangeLayer.h>

namespace NeoML {

template<class T>
static int calcOnnxRangeOutputSize( const CDnnBlob& startBlob, const CDnnBlob& limitBlob, const CDnnBlob& deltaBlob )
{
	const T start = startBlob.GetData<T>().GetValue();
	const T limit = limitBlob.GetData<T>().GetValue();
	const T delta = deltaBlob.GetData<T>().GetValue();

	return static_cast<int>( std::max( T( std::ceil( ( limit - start ) / delta ) ), T( 0 ) ) );
}

template<class T>
static void calcOnnxRangeOutput( const CDnnBlob& startBlob, const CDnnBlob& deltaBlob, CDnnBlob& outputBlob )
{
	T currValue = startBlob.GetData<T>().GetValue();
	const T delta = deltaBlob.GetData<T>().GetValue();

	CDnnBlobBuffer<T> output( outputBlob, TDnnBlobBufferAccess::Read );
	for( int i = 0; i < output.Size(); ++i ) {
		output[i] = currValue;
		currValue += delta;
	}
}


//---------------------------------------------------------------------------------------------------------------------

static const int OnnxRangeLayerVersion = 0;

void COnnxRangeLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( OnnxRangeLayerVersion );
	CBaseReshaper::Serialize( archive );
}

void COnnxRangeLayer::CalculateShapes()
{
	CheckArchitecture( GetInputCount() == 3, GetPath(), "Layer must have 3 inputs" );
	CheckArchitecture( GetOutputCount() == 1, GetPath(), "Layer must have 1 output" );
	CheckArchitecture( inputShapeBlobs[0] != nullptr, GetPath(), "'start' shape tensor missing" );
	CheckArchitecture( inputShapeBlobs[1] != nullptr, GetPath(), "'limit' shape tensor missing" );
	CheckArchitecture( inputShapeBlobs[2] != nullptr, GetPath(), "'delta' shape tensor missing" );

	outputDescs[0] = CBlobDesc( inputShapeBlobs[0]->GetDataType() );
	if( outputDescs[0].GetDataType() == CT_Float ) {
		outputDescs[0].SetDimSize( BD_BatchLength,
			calcOnnxRangeOutputSize<float>( *inputShapeBlobs[0], *inputShapeBlobs[1], *inputShapeBlobs[2] ) );
	} else {
		outputDescs[0].SetDimSize( BD_BatchLength,
			calcOnnxRangeOutputSize<int>( *inputShapeBlobs[0], *inputShapeBlobs[1], *inputShapeBlobs[2] ) );
	}
}

void COnnxRangeLayer::RunOnce()
{
	if( outputBlobs[0]->GetDataType() == CT_Float ) {
		calcOnnxRangeOutput<float>( *inputShapeBlobs[0], *inputShapeBlobs[2], *outputBlobs[0] );
	} else {
		calcOnnxRangeOutput<int>( *inputShapeBlobs[0], *inputShapeBlobs[2], *outputBlobs[0] );
	}
}

} // namespace NeoML
