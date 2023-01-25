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

#include <NeoML/Dnn/Layers/Onnx/OnnxGatherLayer.h>

namespace NeoML {

// Calculates the CBlobDesc of output
static CBlobDesc getOutputDesc( const CBlobDesc& dataDesc, const CBlobDesc& indicesDesc, TBlobDim gatherDim )
{
	CBlobDesc resultDesc = dataDesc;
	for( TBlobDim i = BD_BatchLength; i <= gatherDim; ++i ) {
		resultDesc.SetDimSize( i, indicesDesc.DimSize( i ) );
	}
	return resultDesc;
}

// Shifts indices from [-gatherDimSize;gatherDimSize-1] to [0;gatherDimSize-1]
static void shiftIndices( int gatherDimSize, const CDnnBlob& indices, CDnnBlob& result )
{
	IMathEngine& mathEngine = indices.GetMathEngine();

	mathEngine.VectorFill( result.GetData<int>(), 0, result.GetDataSize() );
	// Add imageSize value to negative indices
	mathEngine.VectorEltwiseLess( indices.GetData<int>(), result.GetData<int>(), result.GetData<int>(), result.GetDataSize() );
	CIntHandleStackVar imageSizeVar( mathEngine );
	imageSizeVar.SetValue( gatherDimSize );
	mathEngine.VectorMultiply( result.GetData<int>(), result.GetData<int>(), result.GetDataSize(), imageSizeVar );
	mathEngine.VectorAdd( result.GetData<int>(), indices.GetData<int>(), result.GetData<int>(), result.GetDataSize() );
}

// Runs Gather operation over given blobs where data type is T
template<class T>
static void runGather( const CDnnBlob& data, const CDnnBlob& indices, CDnnBlob& result, TBlobDim gatherDim )
{
	CPtr<CDnnBlob> shiftedIndices = indices.GetClone();
	shiftIndices( data.DimSize( gatherDim ), indices, *shiftedIndices);

	CLookupDimension lookupDim( 1, 1 );
	lookupDim.VectorCount = data.DimSize( gatherDim );
	lookupDim.VectorSize = data.GetDataSize() / data.DimSize( gatherDim );

	CTypedMemoryHandle<const T> lookupTable = data.GetData<T>();

	IMathEngine& mathEngine = data.GetMathEngine();
	result.Fill<T>( 0 );
	mathEngine.VectorMultichannelLookupAndCopy( shiftedIndices->GetDataSize(), 1, shiftedIndices->GetData<int>(),
		&lookupTable, &lookupDim, 1, result.GetData<T>(), lookupDim.VectorSize );
}

//---------------------------------------------------------------------------------------------------------------------

static const int OnnxGatherLayerVersion = 0;

void COnnxGatherLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( OnnxGatherLayerVersion );
	COnnxLayerBase::Serialize( archive );
	archive.SerializeEnum( gatherDim );
}

void COnnxGatherLayer::CalculateShapes()
{
	CheckArchitecture( GetInputCount() == 2, GetPath(), "Layer must have 2 inputs" );
	CheckArchitecture( GetOutputCount() == 1, GetPath(), "Layer must have 1 output" );

	if( inputShapeBlobs[0] == nullptr ) {
		// No shape-blobs
		// Gather will be executed over inputBlobs/outputBlobs during RunOnce
		CheckArchitecture( inputShapeBlobs[1] == nullptr, GetPath(), "Mixed shape-blobs and blobs" );
		outputDescs[0] = getOutputDesc( inputDescs[0], inputDescs[1], gatherDim );
		return;
	}

	CBlobDesc desc = getOutputDesc( inputShapeBlobs[0]->GetDesc(), inputShapeBlobs[1]->GetDesc(), gatherDim );
	outputShapeBlobs[0] = CDnnBlob::CreateBlob( inputShapeBlobs[0]->GetMathEngine(), desc.GetDataType(), desc );
	if( outputShapeBlobs[0]->GetDataType() == CT_Float ) {
		runGather<float>( *inputShapeBlobs[0], *inputShapeBlobs[1], *outputShapeBlobs[0], gatherDim );
	} else {
		runGather<int>( *inputShapeBlobs[0], *inputShapeBlobs[1], *outputShapeBlobs[0], gatherDim );
	}
}

void COnnxGatherLayer::RunOnce()
{
	if( inputShapeBlobs[0] == nullptr ) {
		if( outputBlobs[0]->GetDataType() == CT_Float ) {
			runGather<float>( *inputBlobs[0], *inputBlobs[1], *outputBlobs[0], gatherDim );
		} else {
			runGather<int>( *inputBlobs[0], *inputBlobs[1], *outputBlobs[0], gatherDim );
		}
	}
}

} // namespace NeoML
