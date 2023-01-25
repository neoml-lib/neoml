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

#include <memory>

#include <NeoML/Dnn/Layers/Onnx/OnnxSliceLayer.h>

namespace NeoML {

static const int OnnxSliceLayerVersion = 0;

void COnnxSliceLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( OnnxSliceLayerVersion );
	COnnxLayerBase::Serialize( archive );
	tensorLayout.Serialize( archive );
}

void COnnxSliceLayer::CalculateShapes()
{
	if( inputShapeBlobs[0] == nullptr ) {
		CBlobDesc outputDesc = sliceDesc( inputDescs[0] );
		if( outputDesc.BlobSize() == 0 ) {
			outputHasElements = false;
		} else {
			outputDescs[0] = outputDesc;
			outputHasElements = true;
		}
		return;
	}

	CBlobDesc outputDesc = sliceDesc( inputShapeBlobs[0]->GetDesc() );
	if( outputDesc.BlobSize() == 0 ) {
		outputHasElements = false;
		outputShapeBlobs[0] = CDnnBlob::CreateVector( GetSingleThreadCpuMathEngine(), outputDesc.GetDataType(), 1 );
	} else {
		outputHasElements = true;
		outputShapeBlobs[0] = CDnnBlob::CreateBlob( inputShapeBlobs[0]->GetMathEngine(),
			inputShapeBlobs[0]->GetDataType(), outputDesc );
		sliceBlob( *inputShapeBlobs[0], *outputShapeBlobs[0] );
	}
}

void COnnxSliceLayer::RunOnce()
{
	if( inputShapeBlobs[0] == nullptr && outputHasElements ) {
		sliceBlob( *inputBlobs[0], *outputBlobs[0] );
	}
}

// Number of slices performed by this layer
int COnnxSliceLayer::getSliceCount() const
{
	return inputShapeBlobs[1]->GetDataSize();
}

// Blob dimension splitted during index'th slice
TBlobDim COnnxSliceLayer::getAxis( int index ) const
{
	if( inputShapeBlobs.Size() <= 3 || inputShapeBlobs[3] == nullptr ) {
		return tensorLayout[index];
	}

	int axis = inputShapeBlobs[3]->GetData<int>().GetValueAt( index );
	if( axis < 0 ) {
		axis += tensorLayout.Size();
	}

	return tensorLayout[axis];
}

// Start coordinate of index'th slice
int COnnxSliceLayer::getStart( int index, int dimSize ) const
{
	int start = inputShapeBlobs[1]->GetData<int>().GetValueAt( index );
	if( start < 0 ) {
		start += dimSize;	
	}
	if( start > dimSize ) {
		start = dimSize;
	}
	return start;
}

// End coordinate of index'th slice
int COnnxSliceLayer::getEnd( int index, int dimSize ) const
{
	int end = inputShapeBlobs[2]->GetData<int>().GetValueAt( index );
	if( end < 0 ) {
		end += dimSize;	
	}
	if( end > dimSize ) {
		end = dimSize;
	}
	return end;
}

// Step of index'th slice
int COnnxSliceLayer::getStep( int index ) const
{
	if( inputShapeBlobs.Size() <= 4 || inputShapeBlobs[4] == nullptr ) {
		return 1;
	}

	return inputShapeBlobs[4]->GetData<int>().GetValueAt( index );
}

// Calculates CBlobDesc after all of the slices
CBlobDesc COnnxSliceLayer::sliceDesc( const CBlobDesc& inputDesc ) const
{
	CBlobDesc resultDesc = inputDesc;
	for( int sliceIndex = 0; sliceIndex < getSliceCount(); ++sliceIndex ) {
		const TBlobDim blobDim = getAxis( sliceIndex );
		const int start = getStart( sliceIndex, inputDesc.DimSize( blobDim ) );
		const int end = getEnd( sliceIndex, inputDesc.DimSize( blobDim ) );
		const int step = getStep( sliceIndex );
		CheckArchitecture( step == 1, GetPath(), "step != 1" );
		resultDesc.SetDimSize( blobDim, end - start );
	}
	return resultDesc;
}

// Performs slices over the given blob
void COnnxSliceLayer::sliceBlob( const CDnnBlob& inputBlob, CDnnBlob& output ) const
{
	TBlobType dataType = inputBlob.GetDataType();
	IMathEngine& mathEngine = inputBlob.GetMathEngine();
	CPtr<CDnnBlob> resultBlob = inputBlob.GetCopy();
	for( int sliceIndex = 0; sliceIndex < getSliceCount(); ++sliceIndex ) {
		const TBlobDim blobDim = getAxis( sliceIndex );
		const int dimSize = resultBlob->DimSize( blobDim );
		const int start = getStart( sliceIndex, dimSize );
		const int end = getEnd( sliceIndex, dimSize );
		NeoPresume( start < end );

		if( start == 0 && end == dimSize ) {
			if( sliceIndex == getSliceCount() - 1 ) {
				output.CopyFrom( resultBlob );
			}
			continue;
		}

		const int middlePartIndex = start == 0 ? 0 : 1;
		CObjectArray<CDnnBlob> parts;
		if( start != 0 ) {
			CBlobDesc frontDesc = resultBlob->GetDesc();
			frontDesc.SetDimSize( blobDim, start );
			parts.Add( CDnnBlob::CreateBlob( mathEngine, dataType, frontDesc ) );
		}
		
		CBlobDesc middleDesc = resultBlob->GetDesc();
		middleDesc.SetDimSize( blobDim, end - start );
		parts.Add( sliceIndex == getSliceCount() - 1 ? &output
			: CDnnBlob::CreateBlob( mathEngine, dataType, middleDesc ) );

		if( end < dimSize ) {
			CBlobDesc backDesc = resultBlob->GetDesc();
			backDesc.SetDimSize( blobDim, dimSize - end );
			parts.Add( CDnnBlob::CreateBlob( mathEngine, dataType, backDesc ) );
		}

		CDnnBlob::SplitByDim( mathEngine, blobDim, resultBlob, parts );
		resultBlob = parts[middlePartIndex];
	}
}

} // namespace NeoML
