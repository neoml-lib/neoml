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

#include <NeoMathEngine/NeoMathEngineDefs.h>

#ifdef NEOML_USE_METAL

#include <MetalMathEngine.h>
#include <MathEngineCommon.h>
#include <MathEngineDnnActivation.h>
#include <MetalKernel.h>

@import Foundation;
@import MetalKit;

namespace NeoML {

static const int FloatDescArrayMaxBlobs = 32;

struct CFloatDescArray {
	int Count;
	CBlobDesc Descs[FloatDescArrayMaxBlobs];
	CFloatHandle Data[FloatDescArrayMaxBlobs];
	int Widths[FloatDescArrayMaxBlobs];
};

void CMetalMathEngine::blobMergeByDim(int dimNum, const CBlobDesc* from, const CFloatHandle* fromData, int fromCount,
	const CBlobDesc& to, const CFloatHandle& toData)
{
	ASSERT_EXPR( toData.GetMathEngine() == this );
	ASSERT_EXPR( fromCount <= MaxBlobDescs );
	ASSERT_EXPR( 0 <= dimNum && dimNum < CBlobDesc::MaxDimensions );

	int s[CBlobDesc::MaxDimensions];
	CFloatDescArray fromArr;
	fromArr.Count = fromCount;
	for(int i = 0; i < fromCount; ++i) {
		fromArr.Descs[i] = from[i];
        ASSERT_EXPR( fromData[i].GetMathEngine() == this );
		fromArr.Data[i] = fromData[i];
		from[i].GetDimSizes(s);
 		fromArr.Widths[i] = 1;
		for(int z = dimNum; z < CBlobDesc::MaxDimensions; z++) {
			fromArr.Widths[i] *= s[z];
		}
	}
	to.GetDimSizes(s);
	int height = 1;
	for(int z  = 0; z < dimNum; z++) {
		height *= s[z];
	}
 	int width = to.BlobSize() / height;
    
    const int heightNorm = ( height + 15 ) / 16;
    int wStart = 0;
    for( int i = 0; i < fromArr.Count; i++ ) {
        C2DKernel kernel( *queue, "matrixKernelBlobMergeByDim", 1, 1, heightNorm, fromArr.Widths[i] );
        kernel.SetParam( height, 0 );
        kernel.SetParam( width, 1 );
        kernel.SetParam( fromArr.Descs[i], 2 );
        kernel.SetParam( fromArr.Data[i], 3 );
        kernel.SetParam( to, 4 );
        kernel.SetParam( toData, 5 );
        kernel.SetParam( heightNorm, 6 );
        kernel.SetParam( wStart, 7 );
        kernel.SetParam( fromArr.Widths[i], 8 );
        ASSERT_EXPR( kernel.Run() );
        
        wStart += fromArr.Widths[i];
    }
}

void CMetalMathEngine::blobSplitByDim(int dimNum, const CBlobDesc& from, const CFloatHandle& fromData,
	const CBlobDesc* to, const CFloatHandle* toData, int toCount)
{
	ASSERT_EXPR( fromData.GetMathEngine() == this );
	ASSERT_EXPR( toCount <= MaxBlobDescs );
	ASSERT_EXPR( 0 <= dimNum && dimNum < CBlobDesc::MaxDimensions );

	CFloatDescArray toArr;
	toArr.Count = toCount;
	int s[CBlobDesc::MaxDimensions];
	for(int i = 0; i < toCount; ++i) {
		toArr.Descs[i] = to[i];
        ASSERT_EXPR( toData[i].GetMathEngine() == this );
		toArr.Data[i] = toData[i];

		to[i].GetDimSizes(s);
 		toArr.Widths[i] = 1;
		for(int z = dimNum; z < CBlobDesc::MaxDimensions; z++) {
			toArr.Widths[i] *= s[z];
		}
	}
	from.GetDimSizes(s);
	int height = 1;
	for(int z  = 0; z < dimNum; z++) {
		height *= s[z];
	}
 	int width = from.BlobSize() / height;
    
    const int heightNorm = ( height + 15 ) / 16;
    int wStart = 0;
    for( int i = 0; i < toArr.Count; i++ ) {
        C2DKernel kernel( *queue, "matrixKernelBlobSplitByDim", 1, 1, heightNorm, toArr.Widths[i] );
        kernel.SetParam( height, 0 );
        kernel.SetParam( width, 1 );
        kernel.SetParam( from, 2 );
        kernel.SetParam( fromData, 3 );
        kernel.SetParam( toArr.Descs[i], 4 );
        kernel.SetParam( toArr.Data[i], 5 );
        kernel.SetParam( kernel.GetGridHeight(), 6 );
        kernel.SetParam( wStart, 7 );
        kernel.SetParam( toArr.Widths[i], 8 );
        ASSERT_EXPR( kernel.Run() );
        
        wStart += toArr.Widths[i];
    }
}

void CMetalMathEngine::BlobMergeByDim(TBlobDim dim, const CBlobDesc* from, const CFloatHandle* fromData, int fromCount, const CBlobDesc& to, const CFloatHandle& toData)
{
	ASSERT_EXPR(dim < BD_Count && fromCount <= MaxBlobDescs);
	blobMergeByDim(dim, from, fromData, fromCount, to, toData);
}

void CMetalMathEngine::BlobMergeByDim(TBlobDim /*dim*/, const CBlobDesc* /*from*/, const CIntHandle* /*fromData*/, int /*fromCount*/, const CBlobDesc& /*to*/, const CIntHandle& /*toData*/)
{
	ASSERT_EXPR(false);
}

void CMetalMathEngine::BlobSplitByDim(TBlobDim dim, const CBlobDesc& from, const CFloatHandle& fromData, const CBlobDesc* to, const CFloatHandle* toData, int toCount)
{
	ASSERT_EXPR(dim < BD_Count && toCount <= MaxBlobDescs);
	blobSplitByDim(dim, from, fromData, to, toData, toCount);
}

void CMetalMathEngine::BlobSplitByDim(TBlobDim /*dim*/, const CBlobDesc& /*from*/, const CIntHandle& /*fromData*/, const CBlobDesc* /*to*/, const CIntHandle* /*toData*/, int /*toCount*/)
{
	ASSERT_EXPR(false);
}

static const int BlobResizeImageCombine = 16;

void CMetalMathEngine::BlobResizeImage( const CBlobDesc& from, const CFloatHandle& fromData, int deltaLeft, int deltaRight,
	int deltaTop, int deltaBottom, float defaultValue, const CBlobDesc& to, const CFloatHandle& toData )
{
    ASSERT_EXPR( fromData.GetMathEngine() == this );
	ASSERT_EXPR( toData.GetMathEngine() == this ); 

	const int geom = to.Height() * to.Width();
	const int totalChannels = to.Channels() * to.Depth();
   
    C3DKernel kernel( *queue, "cubeKernelBlobResizeImage", 1, 1, BlobResizeImageCombine, to.ObjectCount(), totalChannels, geom );
    kernel.SetParam( from, 0 );
    kernel.SetParam( fromData, 1 );
    kernel.SetParam( deltaLeft, 2 );
    kernel.SetParam( deltaRight, 3 );
    kernel.SetParam( deltaTop, 4 );
    kernel.SetParam( deltaBottom, 5 );
    kernel.SetParam( defaultValue, 6 );
    kernel.SetParam( to, 7 );
    kernel.SetParam( toData, 8 );
    ASSERT_EXPR( kernel.Run() );
}

static const int BlobGetSubSequenceCombine = 16;
    
void CMetalMathEngine::BlobGetSubSequence( const CBlobDesc& from, const CFloatHandle& fromData, const CIntHandle& indexHandle,
	const CBlobDesc& to, const CFloatHandle& toData, int startPos, bool isRev )
{
    ASSERT_EXPR( fromData.GetMathEngine() == this );
	ASSERT_EXPR( indexHandle.IsNull() || indexHandle.GetMathEngine() == this );
	ASSERT_EXPR( toData.GetMathEngine() == this );    

    const char* kernelName = indexHandle.IsNull() ? "cubeKernelBlobGetSubSequenceNoIndex" : "cubeKernelBlobGetSubSequence";
    C3DKernel kernel( *queue, kernelName, 1, 1, BlobGetSubSequenceCombine, to.BatchLength(), to.BatchWidth() * to.ListSize(), from.ObjectSize() );
    kernel.SetParam( from, 0 );
    kernel.SetParam( fromData, 1 );
    kernel.SetParam( to, 2 );
    kernel.SetParam( toData, 3 );
    kernel.SetParam( startPos, 4 );
    kernel.SetParam( isRev ? 1 : 0, 5 );
    if( !indexHandle.IsNull() ) {
        kernel.SetParam( indexHandle, 6 );
    }
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::Upsampling2DForward( const CBlobDesc& input, const CFloatHandle& inputData, int heightCopyCount,
	int widthCopyCount, const CBlobDesc& result, const CFloatHandle& resultData )
{
    ASSERT_EXPR( inputData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
    ASSERT_EXPR( heightCopyCount > 0 );
    ASSERT_EXPR( widthCopyCount > 0 );
    ASSERT_EXPR( input.BatchLength() == result.BatchLength() );
    ASSERT_EXPR( input.BatchWidth() == result.BatchWidth() );
    ASSERT_EXPR( input.Channels() == result.Channels() );
    ASSERT_EXPR( input.Depth() == result.Depth() );
    ASSERT_EXPR( input.Height() * heightCopyCount == result.Height() );
    ASSERT_EXPR( input.Width() * widthCopyCount == result.Width() );
    
    const int inputHeight = input.Height();
    const int inputRowSize = input.Width() * input.Depth() * input.Channels();
    const int pixelSize = input.Depth() * input.Channels();
    const int resultHeight = result.Height();
    const int resultRowSize = result.Width() * result.Depth() * result.Channels();
    
    C2DKernel kernel( *queue, "matrixKernelUpsampling2DForward", 1, 1, resultHeight, resultRowSize );
    kernel.SetParam( heightCopyCount, 0 );
    kernel.SetParam( widthCopyCount, 1 );
    kernel.SetParam( pixelSize, 2 );
    kernel.SetParam( input.ObjectCount(), 3 );
    kernel.SetParam( inputHeight, 4 );
    kernel.SetParam( inputRowSize, 5 );
    kernel.SetParam( inputData, 6 );
    kernel.SetParam( resultHeight, 7 );
    kernel.SetParam( resultRowSize, 8 );
    kernel.SetParam( resultData, 9    );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::Upsampling2DBackward( const CBlobDesc&, const CFloatHandle&, int,
	int, const CBlobDesc&, const CFloatHandle& )
{
	ASSERT_EXPR( false );
}

void CMetalMathEngine::BuildIntegerHist( const CConstIntHandle& numbersHandle, int numbersCount,
    const CIntHandle& resultHandle, int maxNumber )
{
	ASSERT_EXPR( numbersHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    VectorFill( resultHandle, 0, maxNumber );

    C1DKernel kernel( *queue, "vectorKernelBuildIntegerHist", 1, numbersCount );
    kernel.SetParam( numbersHandle, 0 );
    kernel.SetParam( numbersCount, 1 );
    kernel.SetParam( resultHandle, 2 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::Reorg( const CBlobDesc& source, const CFloatHandle& sourceData, int stride, bool isForward,
	const CBlobDesc& result, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	const CBlobDesc& input = isForward ? source : result;
	C2DKernel kernel( *queue, "blobReorgFloat", 1, 1, source.ObjectCount() * input.Height(), input.Channels() * input.Width() );
	kernel.SetParam( sourceData, 0 );
	kernel.SetParam( input.Width(), 1 );
	kernel.SetParam( input.Height(), 2 );
	kernel.SetParam( input.Channels(), 3 );
	kernel.SetParam( source.ObjectCount(), 4 );
	kernel.SetParam( stride, 5 );
	kernel.SetParam( isForward, 6 );
	kernel.SetParam( resultData, 7 );
	ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::Reorg( const CBlobDesc& source, const CIntHandle& sourceData, int stride, bool isForward,
	const CBlobDesc& result, const CIntHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

	const CBlobDesc& input = isForward ? source : result;
	C2DKernel kernel( *queue, "blobReorgInt", 1, 1, source.ObjectCount() * input.Height(), input.Channels() * input.Width()  );
	kernel.SetParam( sourceData, 0 );
	kernel.SetParam( input.Width(), 1 );
	kernel.SetParam( input.Height(), 2 );
	kernel.SetParam( input.Channels(), 3 );
	kernel.SetParam( source.ObjectCount(), 4 );
	kernel.SetParam( stride, 5 );
	kernel.SetParam( isForward, 6 );
	kernel.SetParam( resultData, 7 );
	ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::AddWidthIndex( const CBlobDesc& source, const CFloatHandle& sourceData, bool isForward,
	const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelAddWidthIndexFloat", 1, source.ObjectCount() * source.Channels() * source.Width() * source.Height() );
    kernel.SetParam( sourceData, 0 );
    kernel.SetParam( source.Width(), 1 );
    kernel.SetParam( source.Height(), 2 );
    kernel.SetParam( source.Channels(), 3 );
    kernel.SetParam( source.ObjectCount(), 4 );
    kernel.SetParam( isForward, 5 );
    kernel.SetParam( resultData, 6 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::AddWidthIndex( const CBlobDesc& source, const CIntHandle& sourceData, bool isForward,
	const CIntHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelAddWidthIndexInt", 1, source.ObjectCount() * source.Channels() * source.Width() * source.Height() );
    kernel.SetParam( sourceData, 0 );
    kernel.SetParam( source.Width(), 1 );
    kernel.SetParam( source.Height(), 2 );
    kernel.SetParam( source.Channels(), 3 );
    kernel.SetParam( source.ObjectCount(), 4 );
    kernel.SetParam( isForward, 5 );
    kernel.SetParam( resultData, 6 );
    ASSERT_EXPR( kernel.Run() );    
}

void CMetalMathEngine::AddHeightIndex( const CBlobDesc& source, const CFloatHandle& sourceData, bool isForward,
	const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelAddHeightIndexFloat", 1, source.ObjectCount() * source.Channels() * source.Width() * source.Height() );
    kernel.SetParam( sourceData, 0 );
    kernel.SetParam( source.Width(), 1 );
    kernel.SetParam( source.Height(), 2 );
    kernel.SetParam( source.Channels(), 3 );
    kernel.SetParam( source.ObjectCount(), 4 );
    kernel.SetParam( isForward, 5 );
    kernel.SetParam( resultData, 6 );
    ASSERT_EXPR( kernel.Run() );
}

void CMetalMathEngine::AddHeightIndex( const CBlobDesc& source, const CIntHandle& sourceData, bool isForward,
	const CIntHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );

    C1DKernel kernel( *queue, "vectorKernelAddHeightIndexInt", 1, source.ObjectCount() * source.Channels() * source.Width() * source.Height() );
    kernel.SetParam( sourceData, 0 );
    kernel.SetParam( source.Width(), 1 );
    kernel.SetParam( source.Height(), 2 );
    kernel.SetParam( source.Channels(), 3 );
    kernel.SetParam( source.ObjectCount(), 4 );
    kernel.SetParam( isForward, 5 );
    kernel.SetParam( resultData, 6 );
    ASSERT_EXPR( kernel.Run() ); 
}

void CMetalMathEngine::MatrixRowsToVectorSquaredL2Distance( const CConstFloatHandle& matrixHandle, const int matrixHeight,
    const int matrixWidth, const CConstFloatHandle& vectorHandle, const CFloatHandle& resultHandle )
{
	ASSERT_EXPR( matrixHandle.GetMathEngine() == this );
	ASSERT_EXPR( vectorHandle.GetMathEngine() == this );
    ASSERT_EXPR( resultHandle.GetMathEngine() == this );

    VectorFill( resultHandle, 0, matrixHeight );

    C2DKernel kernel( *queue, "matrixKernelMatrixRowsToVectorSquaredL2Distance", 1, ( matrixWidth + 7 ) / 8, matrixHeight, matrixWidth );
    kernel.SetParam( matrixHandle, 0 );
    kernel.SetParam( matrixHeight, 1 );
    kernel.SetParam( matrixWidth, 2 );
    kernel.SetParam( vectorHandle, 3 );
    kernel.SetParam( resultHandle, 4 );
    kernel.SetSharedParam( kernel.GetThreadCount() * sizeof(float), 5 );
    ASSERT_EXPR( kernel.Run() );
}

CActivationDesc* CMetalMathEngine::InitActivation( const CActivationInfo& activation, int dataSize )
{
	size_t bufferSize = 0;
	if( activation.Type == AF_Abs || activation.Type == AF_HSwish || activation.Type == AF_GELU ) {
		bufferSize = 2 * dataSize;
	}
	return new CCommonActivationDesc( *this, activation, bufferSize );
}

bool CMetalMathEngine::IsInPlaceActivation( TActivationFunction activation )
{
	return activation != AF_Abs && activation != AF_HSwish && activation != AF_GELU;
}

void CMetalMathEngine::Activation( const CActivationDesc& activationDesc, const CConstFloatHandle& input,
	const CFloatHandle& output, int dataSize )
{
	const CCommonActivationDesc& desc = static_cast<const CCommonActivationDesc&>( activationDesc );
	switch( desc.Type ) {
		case AF_Linear:
		{
			CConstFloatHandle currData = input;
			if( desc.Param1 != 1.f ) {
				VectorMultiply( currData, output, dataSize, desc.Param1Handle );
				currData = output;
			}
			if( desc.Param2 != 0.f ) {
				VectorAddValue( currData, output, dataSize, desc.Param2Handle );
				currData = output;
			}
			if( currData != output ) {
				VectorCopy( output, currData, dataSize );
			}
			return;
		}
		case AF_ELU:
			VectorELU( input, output, dataSize, desc.Param1Handle );
			return;
		case AF_ReLU:
			VectorReLU( input, output, dataSize, desc.Param1Handle );
			return;
		case AF_LeakyReLU:
			VectorLeakyReLU( input, output, dataSize, desc.Param1Handle );
			return;
		case AF_Abs:
			VectorAbs( input, output, dataSize );
			return;
		case AF_Sigmoid:
			VectorSigmoid( input, output, dataSize );
			return;
		case AF_Tanh:
			VectorTanh( input, output, dataSize );
			return;
		case AF_HardTanh:
			VectorHardTanh( input, output, dataSize );
			return;
		case AF_HardSigmoid:
			VectorHardSigmoid( input, output, dataSize, desc.Param1Handle, desc.Param2Handle );
			return;
		case AF_Power:
			VectorPower( desc.Param1, input, output, dataSize );
			return;
		case AF_HSwish:
			VectorHSwish( input, output, dataSize );
			return;
		case AF_GELU:
			VectorMultiply( input, desc.Buffer, dataSize, desc.Param1Handle );
			VectorSigmoid( desc.Buffer, desc.Buffer, dataSize );
			VectorEltwiseMultiply( input, desc.Buffer, output, dataSize );
			return;
		case AF_None:
			return;
		case AF_Count:
		default:
			ASSERT_EXPR( false );
	}
}

void CMetalMathEngine::ActivationBackward( const CActivationDesc& activationDesc, const CConstFloatHandle& input,
	const CConstFloatHandle& output, const CConstFloatHandle& outputDiff, const CFloatHandle& inputDiff, int dataSize )
{
	const CCommonActivationDesc& desc = static_cast<const CCommonActivationDesc&>( activationDesc );
	switch( desc.Type ) {
		case AF_Linear:
			if( desc.Param1 != 1.f ) {
				VectorMultiply( outputDiff, inputDiff, dataSize, desc.Param1Handle );
			} else if( outputDiff != inputDiff ) {
				VectorCopy( inputDiff, outputDiff, dataSize );
			}
			return;
		case AF_ELU:
			VectorELUDiffOp( output, outputDiff, inputDiff, dataSize, desc.Param1Handle );
			return;
		case AF_ReLU:
			VectorReLUDiffOp( output, outputDiff, inputDiff, dataSize, desc.Param1Handle );
			return;
		case AF_LeakyReLU:
			VectorLeakyReLUDiffOp( output, outputDiff, inputDiff, dataSize, desc.Param1Handle );
			return;
		case AF_Abs:
			VectorAbsDiff( desc.Buffer, outputDiff, inputDiff, dataSize );
			return;
		case AF_Sigmoid:
			VectorSigmoidDiffOp( output, outputDiff, inputDiff, dataSize );
			return;
		case AF_Tanh:
			VectorTanhDiffOp( output, outputDiff, inputDiff, dataSize );
			return;
		case AF_HardTanh:
			VectorHardTanhDiffOp( output, outputDiff, inputDiff, dataSize );
			return;
		case AF_HardSigmoid:
			VectorHardSigmoidDiffOp( output, outputDiff, inputDiff, dataSize, desc.Param1Handle, desc.Param2Handle );
			return;
		case AF_Power:
			VectorPowerDiffOp( desc.Param1, output, outputDiff, inputDiff, dataSize );
			return;
		case AF_HSwish:
			VectorHSwishDiff( input, outputDiff, inputDiff, dataSize );
			return;
		case AF_GELU:
		{
			CFloatHandle multipliedInput = desc.Buffer;
			CFloatHandle sigmoidMultipliedInput = desc.Buffer.GetHandle() + dataSize;
			VectorMultiply( input, multipliedInput, dataSize, desc.Param1Handle );
			VectorSigmoid( multipliedInput, sigmoidMultipliedInput, dataSize );
			VectorSigmoidDiff( multipliedInput, input, inputDiff, dataSize );
			VectorMultiply( inputDiff, inputDiff, dataSize, desc.Param1Handle );
			VectorAdd( inputDiff, sigmoidMultipliedInput, inputDiff, dataSize );
			VectorEltwiseMultiply( inputDiff, outputDiff, inputDiff, dataSize );
			return;
		}
		case AF_None:
			return;
		case AF_Count:
		default:
			ASSERT_EXPR( false );
	}
}

} // namespace NeoML

#endif // NEOML_USE_METAL
