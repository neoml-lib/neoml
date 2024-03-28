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

#include <common.h>
#pragma hdrstop

#include <NeoML/Dnn/Layers/MatrixMultiplicationLayer.h>

namespace NeoML {

CMatrixMultiplicationLayer::CMatrixMultiplicationLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CMatrixMultiplicationLayer", /*isLearnable*/false )
{}

static const int MatrixMultiplicationLayerVersion = 0;

void CMatrixMultiplicationLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( MatrixMultiplicationLayerVersion );
	CBaseLayer::Serialize( archive );
}


void CMatrixMultiplicationLayer::Reshape()
{
	CheckInputs();
	CheckLayerArchitecture( inputDescs.Size() == 2, "layer must have 2 inputs" );

	CheckLayerArchitecture( inputDescs[0].Channels() == inputDescs[1].GeometricalSize(),
		"input[0].Channels must be equal to input[1].GeometricalSize" );
	if( IsBackwardPerformed() ) {
		CheckLayerArchitecture( inputDescs[0].ObjectCount() == inputDescs[1].ObjectCount(),
			"object count mismatch between inputs" );
	} else {
		CheckLayerArchitecture( inputDescs[0].ObjectCount() == inputDescs[1].ObjectCount()
			|| inputDescs[0].ObjectCount() == 1 || inputDescs[1].ObjectCount() == 1,
			"object count mismatch between inputs" );
	}

	outputDescs.SetSize( 1 );
	CBlobDesc outputDesc = inputDescs[0];
	outputDesc.SetDimSize( BD_Channels, inputDescs[1].Channels() );
	if( inputDescs[1].ObjectCount() > inputDescs[0].ObjectCount() ) {
		outputDesc.SetDimSize( BD_BatchLength, inputDescs[1].BatchLength() );
		outputDesc.SetDimSize( BD_BatchWidth, inputDescs[1].BatchWidth() );
		outputDesc.SetDimSize( BD_ListSize, inputDescs[1].ListSize() );
	}

	outputDescs[0] = outputDesc;
}

void CMatrixMultiplicationLayer::RunOnce()
{
	const int firstHeight = inputBlobs[0]->GetGeometricalSize();
	const int firstWidth  = inputBlobs[0]->GetChannelsCount();
	const int secondWidth = inputBlobs[1]->GetChannelsCount();
	const int resultBufferSize = outputBlobs[0]->GetObjectSize();

	if( inputBlobs[0]->GetObjectCount() == inputBlobs[1]->GetObjectCount() ) {
		MathEngine().MultiplyMatrixByMatrix( /*batchSize*/inputBlobs[0]->GetObjectCount(),
			/*first*/inputBlobs[0]->GetData(), firstHeight, firstWidth,
			/*second*/inputBlobs[1]->GetData(), secondWidth,
			/*result*/outputBlobs[0]->GetData(),
			/*resultBufferSize*/outputBlobs[0]->GetDataSize() );
	} else if( inputBlobs[1]->GetObjectCount() == 1 ) {
		for( int i = 0; i < inputBlobs[0]->GetObjectCount(); ++i ) {
			MathEngine().MultiplyMatrixByMatrix( /*batchSize*/1,
				/*first*/inputBlobs[0]->GetObjectData( i ), firstHeight, firstWidth,
				/*second*/inputBlobs[1]->GetData(), secondWidth,
				/*result*/outputBlobs[0]->GetObjectData( i ), resultBufferSize );
		}
	} else if( inputBlobs[0]->GetObjectCount() == 1 ) {
		for( int i = 0; i < inputBlobs[1]->GetObjectCount(); ++i ) {
			MathEngine().MultiplyMatrixByMatrix( /*batchSize*/1,
				/*first*/inputBlobs[0]->GetData(), firstHeight, firstWidth,
				/*second*/inputBlobs[1]->GetObjectData( i ), secondWidth,
				/*result*/outputBlobs[0]->GetObjectData( i ), resultBufferSize );
		}
	} else {
		NeoAssert( false );
	}
}

void CMatrixMultiplicationLayer::BackwardOnce()
{
	const int batchSize = inputBlobs[0]->GetObjectCount();
	NeoAssert( batchSize == inputBlobs[1]->GetObjectCount() );

	{
		const int firstHeight = outputDiffBlobs[0]->GetGeometricalSize();
		const int firstWidth = outputDiffBlobs[0]->GetChannelsCount();
		const int secondHeight = inputBlobs[1]->GetGeometricalSize();
		const int resultBufferSize = inputDiffBlobs[0]->GetDataSize();

		NeoAssert( firstWidth == inputBlobs[1]->GetChannelsCount() );
		NeoAssert( firstHeight == inputBlobs[0]->GetGeometricalSize() );

		MathEngine().MultiplyMatrixByTransposedMatrix( batchSize,
			/*first*/outputDiffBlobs[0]->GetData(), firstHeight, firstWidth,
			/*second*/inputBlobs[1]->GetData(), secondHeight,
			/*result*/inputDiffBlobs[0]->GetData(), resultBufferSize );
	}

	{
		const int firstHeight = inputBlobs[0]->GetGeometricalSize();
		const int firstWidth = inputBlobs[0]->GetChannelsCount();
		const int secondWidth = outputDiffBlobs[0]->GetChannelsCount();
		const int resultBufferSize = inputDiffBlobs[1]->GetDataSize();

		MathEngine().MultiplyTransposedMatrixByMatrix( batchSize,
			/*first*/inputBlobs[0]->GetData(), firstHeight, firstWidth,
			/*second*/outputDiffBlobs[0]->GetData(), secondWidth,
			/*result*/inputDiffBlobs[1]->GetData(), resultBufferSize );
	}
}

CLayerWrapper<CMatrixMultiplicationLayer> MatrixMultiplication()
{
	return CLayerWrapper<CMatrixMultiplicationLayer>( "MatrixMultiplication" );
}

} // namespace NeoML
