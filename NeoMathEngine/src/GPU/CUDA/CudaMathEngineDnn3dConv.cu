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

#ifdef NEOML_USE_CUDA

#include <CudaMathEngine.h>
#include <CudaMathEngineDnnConvs.h>
#include <MemoryHandleInternal.h>

#include <Kernels/CudaDnn3dConvKernels.h>

namespace NeoML {

// Temporary matrix height
static inline int tempMatrixHeight( const CCuda3dConvolutionDescInternal& desc )
{
	return desc.Source.ObjectCount() * desc.Result.Height() * desc.Result.Width() * desc.Result.Depth();
}

// Temporary matrix width
static inline int tempMatrixWidth( const CCuda3dConvolutionDescInternal& desc )
{
	return desc.Filter.ObjectSize();
}

C3dConvolutionDesc* CCudaMathEngine::InitBlob3dConvolution( const CBlobDesc& input, int paddingHeight,
	int paddingWidth, int paddingDepth, int strideHeight, int strideWidth, int strideDepth,
	const CBlobDesc& filter, const CBlobDesc& output )
{
	CCuda3dConvolutionDesc* desc = new CCuda3dConvolutionDesc();
	desc->Internal.Source = input;
	desc->Internal.Filter = filter;
	desc->Internal.Result = output;
	desc->Internal.StrideHeight = strideHeight;
	desc->Internal.StrideWidth = strideWidth;
	desc->Internal.StrideDepth = strideDepth;
	desc->Internal.PaddingHeight = paddingHeight;
	desc->Internal.PaddingWidth = paddingWidth;
	desc->Internal.PaddingDepth = paddingDepth;
	return desc;
}

void CCudaMathEngine::Blob3dConvolution( const C3dConvolutionDesc& convDesc,
	const CFloatHandle& source, const CFloatHandle& filter, const CFloatHandle* freeTerm,
	const CFloatHandle& result )
{
	const CCuda3dConvolutionDescInternal& desc = static_cast<const CCuda3dConvolutionDesc&>( convDesc ).Internal;

	if( freeTerm != 0 ) {
		// Fill the output with the free term values
		SetVectorToMatrixRows( result,
			desc.Result.ObjectCount() * desc.Result.Height() * desc.Result.Width() * desc.Result.Depth(),
			desc.Filter.ObjectCount(), *freeTerm );
	}

	if( desc.Filter.Height() == 1 && desc.Filter.Width() == 1
		&& desc.StrideHeight == 1 && desc.StrideWidth == 1 && desc.StrideDepth == 1
		&& desc.PaddingHeight == 0 && desc.PaddingWidth == 0 && desc.PaddingDepth == 1 )
	{
		// Convolution is a matrix product without building a temporary matrix
		if( freeTerm != 0 ) {
			multiplyMatrixByTransposedMatrixAndAdd( source,
				desc.Source.ObjectCount() * desc.Result.Height() * desc.Result.Width() * desc.Result.Depth(),
				desc.Filter.ObjectSize(), desc.Filter.ObjectSize(), filter,
				desc.Filter.ObjectCount(), desc.Filter.ObjectSize(), result,
				desc.Filter.ObjectCount() );
		} else {
			MultiplyMatrixByTransposedMatrix( source,
				desc.Source.ObjectCount() * desc.Result.Height() * desc.Result.Width() * desc.Result.Depth(),
				desc.Filter.ObjectSize(), desc.Filter.ObjectSize(), filter,
				desc.Filter.ObjectCount(), desc.Filter.ObjectSize(), result,
				desc.Filter.ObjectCount(), desc.Result.BlobSize() );
		}
		return;
	}
	
	// Build the temporary matrix
	const int matrixHeight = tempMatrixHeight( desc );
	const int matrixWidth = tempMatrixWidth( desc );
	CFloatHandleStackVar tempMatrix( *this, matrixHeight * matrixWidth );
	{
		dim3 blockCount;
		dim3 threadCount;
		const int widthNorm = ( matrixWidth + BuildTempMatrixCombine - 1 ) / BuildTempMatrixCombine;
		getCudaTaskGrid2D( blockCount, threadCount, matrixHeight, widthNorm );
		BuildTempMatrixKernel<<<blockCount, threadCount, 0, cudaStream>>>( desc, GetRaw( source ), matrixHeight,
			matrixWidth, GetRaw( tempMatrix.GetHandle() ), widthNorm );
	}
	
	// Multiply the temporary matrix by the filter
	if( freeTerm != 0 ) {
		multiplyMatrixByTransposedMatrixAndAdd( tempMatrix,
			desc.Source.ObjectCount() * desc.Result.Height() * desc.Result.Width() * desc.Result.Depth(),
			desc.Filter.ObjectSize(), desc.Filter.ObjectSize(), filter,
			desc.Filter.ObjectCount(), desc.Filter.ObjectSize(), result,
			desc.Filter.ObjectCount() );
	} else {
		MultiplyMatrixByTransposedMatrix( tempMatrix,
			desc.Source.ObjectCount() * desc.Result.Height() * desc.Result.Width() * desc.Result.Depth(),
			desc.Filter.ObjectSize(), desc.Filter.ObjectSize(), filter,
			desc.Filter.ObjectCount(), desc.Filter.ObjectSize(), result,
			desc.Filter.ObjectCount(), desc.Result.BlobSize() );
	}
}

void CCudaMathEngine::Blob3dConvolutionBackward( const C3dConvolutionDesc& convDesc, const CFloatHandle& outputDiff,
	const CFloatHandle& filter, const CFloatHandle* freeTerm, const CFloatHandle& inputDiff )
{
	const CCuda3dConvolutionDescInternal& desc = static_cast<const CCuda3dConvolutionDesc&>( convDesc ).Internal;
	const int filterCount = desc.Filter.ObjectCount();
	const int filterObjectSize = desc.Filter.ObjectSize();

	if( desc.Filter.Height() == 1 && desc.Filter.Width() == 1 && desc.Filter.Depth() == 1
		&& desc.StrideHeight == 1 && desc.StrideWidth == 1 && desc.StrideDepth == 1
		&& desc.PaddingHeight == 0 && desc.PaddingWidth == 0 && desc.PaddingDepth == 1 )
	{
		// The convolution backward pass is a matrix product without creating a temporary matrix
		MultiplyMatrixByMatrix( 1, outputDiff, desc.Result.BlobSize() / filterCount, filterCount,
			filter, filterObjectSize, inputDiff, desc.Source.BlobSize() );
		if( freeTerm != 0 ) {
			AddVectorToMatrixRows( 1, inputDiff, inputDiff, desc.Source.ObjectCount() * desc.Source.Height() * desc.Source.Width(),
				desc.Source.Channels() * desc.Source.Depth(), *freeTerm );
		}
		return;
	}

	if( freeTerm != 0 ) {
		// Fill the input gradients with the free term values
		SetVectorToMatrixRows( inputDiff, desc.Source.ObjectCount() * desc.Source.Height() * desc.Source.Width() * desc.Source.Depth(),
			desc.Source.Channels(), *freeTerm );
	} else {
		VectorFill( inputDiff, 0.f, desc.Source.BlobSize() );
	}

	TBackwardOperationType operation = BOT_AtomicAdd;
	if( desc.Filter.Depth() <= desc.StrideDepth && desc.Filter.Width() <= desc.StrideWidth && desc.Filter.Height() <= desc.StrideHeight ) {
		// The filter areas do not intersect, so atomic operations are not needed
		operation = freeTerm == 0 ? BOT_Set : BOT_Add;
	}
	
	// Get the temporary matrix
	const int matrixHeight = tempMatrixHeight( desc );
	const int matrixWidth = tempMatrixWidth( desc );
	CFloatHandleStackVar tempMatrix( *this, matrixHeight * matrixWidth );
	MultiplyMatrixByMatrix( 1, outputDiff, desc.Result.BlobSize() / filterCount, filterCount,
		filter, filterObjectSize, tempMatrix, tempMatrix.Size() );
		
	// Get the input gradients from the temporary matrix data
	dim3 blockCount;
	dim3 threadCount;
	int widthNorm = ( matrixWidth + BuildInputFromTempMatrixCombine - 1 ) / BuildInputFromTempMatrixCombine;
	getCudaTaskGrid2D( blockCount, threadCount, matrixHeight, widthNorm );
	BuildInputFromTempMatrixKernel<<<blockCount, threadCount, 0, cudaStream>>>( desc, GetRaw( tempMatrix.GetHandle() ),
		matrixHeight, matrixWidth, GetRaw( inputDiff ), operation, widthNorm );
}

void CCudaMathEngine::Blob3dConvolutionLearnAdd( const C3dConvolutionDesc& convDesc,
	const CFloatHandle& input, const CFloatHandle& outputDiff, const CFloatHandle& filterDiff,
	const CFloatHandle* freeTermDiff, bool isFreeTermDiffFromInput )
{
	const CCuda3dConvolutionDescInternal& desc = static_cast<const CCuda3dConvolutionDesc&>( convDesc ).Internal;

	if( freeTermDiff != 0 ) {
		const CConstFloatHandle& freeTermSrc = isFreeTermDiffFromInput ? input : outputDiff;
		// Get the free term gradient
		if( !isFreeTermDiffFromInput ) {
			SumMatrixRowsAdd( 1, *freeTermDiff, outputDiff, desc.Result.BlobSize() / desc.Filter.ObjectCount(),
				desc.Filter.ObjectCount() );
		} else {
			SumMatrixRowsAdd( 1, *freeTermDiff, input, desc.Source.BlobSize() / desc.Source.Channels(),
				desc.Source.Channels() );
		}
	}
	
	// Build the temporary matrix
	const int matrixHeight = tempMatrixHeight( desc );
	const int matrixWidth = tempMatrixWidth( desc );
	const int filterCount = desc.Filter.ObjectCount();
	CFloatHandleStackVar tempMatrix( *this, matrixHeight * matrixWidth );
	{
		dim3 blockCount;
		dim3 threadCount;
		const int widthNorm = ( matrixWidth + BuildTempMatrixCombine - 1 ) / BuildTempMatrixCombine;
		getCudaTaskGrid2D( blockCount, threadCount, matrixHeight, widthNorm );
		BuildTempMatrixKernel<<<blockCount, threadCount, 0, cudaStream>>>( desc, GetRaw( input ), matrixHeight,
			matrixWidth, GetRaw( tempMatrix.GetHandle() ), widthNorm );
	}
	
	// Get the filter gradients by multiplying the temporary matrix and the output gradients
	MultiplyTransposedMatrixByMatrixAndAdd( outputDiff, matrixHeight, filterCount, filterCount,
		tempMatrix, matrixWidth, matrixWidth, filterDiff, matrixWidth, desc.Filter.BlobSize() );
}

} // namespace NeoML

#endif // NEOML_USE_CUDA
