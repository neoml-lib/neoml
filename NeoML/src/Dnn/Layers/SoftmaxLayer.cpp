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

#include <common.h>
#pragma hdrstop

#include <NeoML/Dnn/Layers/SoftmaxLayer.h>
#include <NeoMathEngine/NeoMathEngine.h>
#include <float.h>

namespace NeoML {

CSoftmaxLayer::CSoftmaxLayer( IMathEngine& mathEngine ) :
	CBaseInPlaceLayer( mathEngine, "CCnnSoftmaxLayer" ),
	area( NA_ObjectSize )
{
}

void CSoftmaxLayer::RunOnce()
{
	CheckInput1();

	static_assert( NA_Count == 4, "NA_Count != 4" );
	switch( area ) {
		case NA_ObjectSize:
			// softmax over objects
			MathEngine().MatrixSoftmaxByRows( inputBlobs[0]->GetData(), inputBlobs[0]->GetObjectCount(),
				inputBlobs[0]->GetObjectSize(), outputBlobs[0]->GetData() );
			break;
		case NA_BatchLength:
			// softmax over batchLength
			MathEngine().MatrixSoftmaxByColumns( inputBlobs[0]->GetData(), inputBlobs[0]->GetBatchLength(),
				inputBlobs[0]->GetDataSize() / inputBlobs[0]->GetBatchLength(), outputBlobs[0]->GetData() );
			break;
		case NA_ListSize:
			NeoAssert( inputBlobs[0]->GetObjectSize() == 1 );
			// softmax over listSize
			MathEngine().MatrixSoftmaxByRows( inputBlobs[0]->GetData(),
				inputBlobs[0]->GetObjectCount() / inputBlobs[0]->GetListSize(), inputBlobs[0]->GetListSize(),
				outputBlobs[0]->GetData() );
			break;
		case NA_Channel:
			// softmax over channel
			MathEngine().MatrixSoftmaxByRows( inputBlobs[0]->GetData(), inputBlobs[0]->GetObjectCount() * inputBlobs[0]->GetGeometricalSize(),
				inputBlobs[0]->GetChannelsCount(), outputBlobs[0]->GetData() );
			break;
		default:
			NeoAssert( false );
	}
}

void CSoftmaxLayer::BackwardOnce()
{
	static_assert( NA_Count == 4, "NA_Count != 4" );
	switch( area ) {
		case NA_ObjectSize:
			// softmax over object
			MathEngine().MatrixSoftmaxDiffOpByRows( outputBlobs[0]->GetData(), outputDiffBlobs[0]->GetData(),
				outputBlobs[0]->GetObjectCount(), outputBlobs[0]->GetObjectSize(), inputDiffBlobs[0]->GetData() );
			break;
		case NA_BatchLength:
			// softmax over batchLength
			MathEngine().MatrixSoftmaxDiffOpByColumns( outputBlobs[0]->GetData(), outputDiffBlobs[0]->GetData(),
				outputBlobs[0]->GetBatchLength(), outputBlobs[0]->GetDataSize() / outputBlobs[0]->GetBatchLength(),
				inputDiffBlobs[0]->GetData() );
			break;
		case NA_ListSize:
			NeoAssert( inputBlobs[0]->GetObjectSize() == 1 );
			// softmax over listSize
			MathEngine().MatrixSoftmaxDiffOpByRows( outputBlobs[0]->GetData(), outputDiffBlobs[0]->GetData(),
				inputBlobs[0]->GetObjectCount() / inputBlobs[0]->GetListSize(), outputBlobs[0]->GetListSize(),
				inputDiffBlobs[0]->GetData() );
			break;
		case NA_Channel:
			// softmax over channel
			MathEngine().MatrixSoftmaxDiffOpByRows( outputBlobs[0]->GetData(), outputDiffBlobs[0]->GetData(),
				outputBlobs[0]->GetObjectCount() * outputBlobs[0]->GetGeometricalSize(), outputBlobs[0]->GetChannelsCount(),
				inputDiffBlobs[0]->GetData() );
			break;
		default:
			NeoAssert( false );
	}
}

static const int SoftmaxLayerVersion = 2000;

void CSoftmaxLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( SoftmaxLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseInPlaceLayer::Serialize( archive );

	if( archive.IsStoring() ) {
		archive << static_cast<int>( area );
	} else if( archive.IsLoading() ) {
		int areaInt;
		archive >> areaInt;
		area = static_cast<TNormalizationArea>( areaInt );
	} else {
		NeoAssert( false );
	}
}

} // namespace NeoML
