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

#include <algorithm>
#include <CpuMathEngine.h>
#include <CpuExecutionScope.h>
#include <float.h>
#include <MathEngineCommon.h>
#include <MathEngineDnnConv.h>
#include <MemoryHandleInternal.h>
#include <CpuMathEnginePrivate.h>

namespace NeoML {

void CCpuMathEngine::blob3dConvolution1x1x1( const CBlobDesc& source, const CBlobDesc& /*filter*/, const CBlobDesc& result,
	int strideHeight, int strideWidth, int strideDepth,
	const float* sourceData, const float* filterData, const float* freeTermData, float* resultData )
{
	const int channels = source.Channels();
	const int geomSize = result.ObjectCount() * result.GeometricalSize();
	const int newChannels = result.Channels();
	// Convolution is matrix product:
	// [geomSize x channels] * [newChannels x channels]T
	// then add the free term if necessary

	// Split the first matrix by rows
	if( freeTermData != nullptr ) {
		setVectorToMatrixRows( resultData, geomSize, newChannels, freeTermData );
	} else {
		NeoML::vectorFill0( resultData, geomSize * newChannels );
	}

	if( strideHeight == 1 && strideWidth == 1 && strideDepth == 1 ) {
		multiplyMatrixByTransposedMatrixAndAdd( sourceData, geomSize, channels, channels,
			filterData, newChannels, channels, resultData, newChannels, nullptr );
	} else {
		CFloatHandleVar repackedHolder( mathEngine(), geomSize * channels );
		float* repackedData = GetRaw( repackedHolder.GetHandle() );
		// Repack the input blob, removing the unused data
		for( int out = 0; out < geomSize; ++out ) {
			int objNum = out;
			const int outK = objNum % result.Depth();
			objNum /= result.Depth();
			const int outI = objNum % result.Width();
			objNum /= result.Width();
			const int outJ = objNum % result.Height();
			objNum /= result.Height();

			float* const sourceDataPtr = repackedData + out * channels;
			const float* const inputData = sourceData + ( ( ( objNum * source.Height() + outJ * strideHeight )
				* source.Width() + outI * strideWidth ) * source.Depth() + outK * strideDepth ) * channels;
			dataCopy( sourceDataPtr, inputData, channels );
		}
		multiplyMatrixByTransposedMatrixAndAdd( repackedData, geomSize, channels, channels,
			filterData, newChannels, channels, resultData, newChannels, nullptr );
	}
}

void CCpuMathEngine::blob3dConvolution1x1x1Backward( const CCommon3dConvolutionDesc& desc,
	const float* outputDiffData, const float* filterData, const CConstFloatHandle* freeTermData,
	float* inputDiffData )
{
	const float* freeTermDataRaw = freeTermData == nullptr ? nullptr : GetRaw( *freeTermData );

	const CBlobDesc& inputDiff = desc.Source;
	const CBlobDesc& outputDiff = desc.Result;

	bool isRepackNeeded = desc.StrideHeight > 1 || desc.StrideWidth > 1 || desc.StrideDepth > 1;

	CBlobDesc resultBlob = inputDiff;
	float* resultData = inputDiffData;
	int resultBlobHolderSize = 0;
	if( isRepackNeeded ) {
		resultBlob = outputDiff;
		resultBlob.SetDimSize( BD_Channels, inputDiff.Channels() );
		resultBlobHolderSize = resultBlob.BlobSize();
	}

	CFloatHandleVar resultBlobHolder( mathEngine(), resultBlobHolderSize );

	if( isRepackNeeded ) {
		resultData = GetRaw( resultBlobHolder.GetHandle() );
	}

	const int batchCount = outputDiff.ObjectCount();
	float* inputDiffDataPtr = inputDiffData;

	if( freeTermData != nullptr ) {
		setVectorToMatrixRows( inputDiffDataPtr, batchCount * inputDiff.GeometricalSize(),
			inputDiff.Channels(), freeTermDataRaw );
	}

	if( isRepackNeeded || freeTermData == nullptr ) {
		multiplyMatrixByMatrix( outputDiffData,
			batchCount * outputDiff.GeometricalSize(), outputDiff.Channels(), outputDiff.Channels(),
			filterData, resultBlob.Channels(), resultBlob.Channels(),
			resultData, resultBlob.Channels(), nullptr );
	} else {
		multiplyMatrixByMatrixAndAdd( outputDiffData,
			batchCount * outputDiff.GeometricalSize(), outputDiff.Channels(), outputDiff.Channels(),
			filterData, resultBlob.Channels(), resultBlob.Channels(),
			resultData, resultBlob.Channels(), nullptr );
	}

	if( isRepackNeeded ) {
		// Repack the result blob
		int inputColSize = inputDiff.Channels() * inputDiff.Depth();
		int inputRowSize = inputColSize * inputDiff.Width();
		int inputObjSize = inputRowSize * inputDiff.Height();

		if( freeTermData == nullptr ) {
			vectorFill0( inputDiffDataPtr, inputObjSize * batchCount );
		}

		for( int b = 0; b < batchCount; ++b ) {
			float* inputDiffRow = inputDiffDataPtr;
			for( int j = 0; j < resultBlob.Height(); ++j ) {
				float* inputDiffCol = inputDiffRow;
				for( int i = 0; i < resultBlob.Width(); ++i ) {
					float* inputDiffPixel = inputDiffCol;
					for( int k = 0; k < resultBlob.Depth(); ++k ) {
						NeoML::vectorAdd( inputDiffPixel, resultData, inputDiffPixel, inputDiff.Channels() );
						inputDiffPixel += inputDiff.Channels() * desc.StrideDepth;
						resultData += inputDiff.Channels();
					}
					inputDiffCol += inputColSize * desc.StrideWidth;
				}
				inputDiffRow += inputRowSize * desc.StrideHeight;
			}
			inputDiffDataPtr += inputObjSize;
		}
	}
}

void CCpuMathEngine::blob3dConvolution1x1x1LearnAdd( const CCommon3dConvolutionDesc& desc, const CConstFloatHandle& inputData,
	const CConstFloatHandle& outputDiffData, const CFloatHandle& filterDiffData, const CFloatHandle* freeTermDiffData )
{
	const CBlobDesc& input = desc.Source;
	const CBlobDesc& outputDiff = desc.Result;
	const CBlobDesc& filterDiff = desc.Filter;

	bool isRepackNeeded = desc.StrideHeight > 1 || desc.StrideWidth > 1 || desc.StrideDepth > 1;
	CBlobDesc inputBlob = input;
	CConstFloatHandle inputBlobData = inputData;

	int inputBlobHolderSize = 0;
	if( isRepackNeeded ) {
		inputBlob = outputDiff;
		inputBlob.SetDimSize( BD_Channels, input.Channels() );
		inputBlobHolderSize = inputBlob.BlobSize();	
	}

	CFloatHandleVar inputBlobHolder( mathEngine(), inputBlobHolderSize );

	if( isRepackNeeded ) {
		inputBlobData = inputBlobHolder.GetHandle();

		// Repack the input
		const float* inputDataPtr = GetRaw( inputData );
		float* inputBlobDataPtr = GetRaw( inputBlobHolder.GetHandle() );
		for( int b = 0; b < inputBlob.ObjectCount(); ++b ) {
			const float* inputRowData = inputDataPtr;
			for( int j = 0; j < inputBlob.Height(); ++j ) {
				const float* inputColData = inputRowData;
				for( int i = 0; i < inputBlob.Width(); ++i ) {
					const float* inputPixelData = inputColData;
					for( int k = 0; k < inputBlob.Depth(); ++k ) {
						dataCopy( inputBlobDataPtr, inputPixelData, input.Channels() );
						inputBlobDataPtr += input.Channels();
						inputPixelData += input.Channels() * desc.StrideDepth;
					}
					inputColData += input.Depth() * input.Channels() * desc.StrideWidth;
				}
				inputRowData += input.Width() * input.Depth() * input.Channels() * desc.StrideHeight;
			}
			inputDataPtr += input.ObjectSize();
		}
	}

	int batchSize = outputDiff.ObjectCount();

	// Train the filter
	MultiplyTransposedMatrixByMatrixAndAdd( outputDiffData,
		batchSize * outputDiff.GeometricalSize(), filterDiff.BatchWidth(), filterDiff.BatchWidth(),
		inputBlobData, inputBlob.Channels(), inputBlob.Channels(),
		filterDiffData, inputBlob.Channels(),
		filterDiff.BatchWidth() * inputBlob.Channels(), nullptr );

	if( freeTermDiffData != 0 ) {
		// Train the free term
		SumMatrixRowsAdd( 1, *freeTermDiffData, outputDiffData, batchSize * outputDiff.GeometricalSize(), filterDiff.BatchWidth() );
	}
}

void CCpuMathEngine::blob3dConvolutionPrepareInput( const CCommon3dConvolutionDesc& desc, float* inputPreparedData,
	const float* inputBlobData, int inputObject, int outputHeight, int outputWidthExStart, int outputWidthExCount )
{
	const CBlobDesc& inputBlob = desc.Source;
	const CBlobDesc& outputBlob = desc.Result;
	const CBlobDesc& filterBlob = desc.Filter;

	const float paddingFill = 0;

	int filterDepthSize = filterBlob.Depth() * inputBlob.Channels();
	int filterRowSize = filterBlob.Width() * filterDepthSize;

	const float* input = inputBlobData + inputObject * inputBlob.ObjectSize();
	float* inputPrepared = inputPreparedData;

	for( int wEx = outputWidthExStart; wEx < outputWidthExStart + outputWidthExCount; ++wEx ) {
		int outputK = wEx % outputBlob.Depth();
		int outputI = wEx / outputBlob.Depth();
		int inputI = outputI * desc.StrideWidth - desc.PaddingWidth;
		int inputK = outputK * desc.StrideDepth - desc.PaddingDepth;

		for( int j = 0; j < outputHeight; ++j ) {
			int rowsToComplete = filterBlob.Height();
			int inputJ = j * desc.StrideHeight - desc.PaddingHeight;
			if( j > 0 ) {
				// Copy part of the data from the previous rows
				int rowsToCopy = filterBlob.Height() - desc.StrideHeight;
				if( rowsToCopy > 0 ) {
					dataCopy( inputPrepared, inputPrepared - rowsToCopy * filterRowSize,
						rowsToCopy * filterRowSize );
					rowsToComplete -= rowsToCopy;
					inputJ += rowsToCopy;
					inputPrepared += rowsToCopy * filterRowSize;
				}
			}

			// Copy data from the input to inputPrepared row by row
			for( int row = 0; row < rowsToComplete; ++row ) {
				if( inputJ < 0 || inputJ >= inputBlob.Height() ) {
					// padding either top or bottom
					vectorFill( inputPrepared, paddingFill, filterRowSize );
					inputPrepared += filterRowSize;
				} else {
					int colsToComplete = filterBlob.Width();
					int startI = inputI;
					if( startI < 0 ) {
						PRESUME_EXPR( -startI < colsToComplete );
						// padding left
						vectorFill( inputPrepared, paddingFill, -startI * filterDepthSize );
						inputPrepared += -startI * filterDepthSize;
						colsToComplete += startI;
						startI = 0;
					}

					int endI = startI + colsToComplete;
					if( endI > inputBlob.Width() ) {
						endI = inputBlob.Width();
					}
					int validI = endI - startI;

					if( validI > 0 ) {
						// Copy data from the input to inputPrepared column by column
						for( ; startI < endI; ++startI ) {
							int depthToComplete = filterBlob.Depth();
							int startK = inputK;
							if( startK < 0 ) {
								PRESUME_EXPR( -startK < depthToComplete );
								// padding front
								vectorFill( inputPrepared, paddingFill, -startK * inputBlob.Channels() );
								inputPrepared += -startK * inputBlob.Channels();
								depthToComplete += startK;
								startK = 0;
							}

							int endK = startK + depthToComplete;
							if( endK > inputBlob.Depth() ) {
								endK = inputBlob.Depth();
							}
							int validK = endK - startK;

							if( validK > 0 ) {
								// Copy the main data
								dataCopy( inputPrepared,
									input + ( ( inputJ * inputBlob.Width() + startI ) * inputBlob.Depth() + startK )
									* inputBlob.Channels(),
									validK * inputBlob.Channels() );
								depthToComplete -= validK;
								inputPrepared += validK * inputBlob.Channels();
							}

							if( depthToComplete > 0 ) {
								// padding back
								vectorFill( inputPrepared, paddingFill, depthToComplete * inputBlob.Channels() );
								inputPrepared += depthToComplete * inputBlob.Channels();
							}
						}

						colsToComplete -= validI;
					}

					if( colsToComplete > 0 ) {
						// padding right
						vectorFill( inputPrepared, paddingFill, colsToComplete * filterDepthSize );
						inputPrepared += colsToComplete * filterDepthSize;
					}
				}
				++inputJ;
			}
		}
	}
}

void CCpuMathEngine::blob3dConvolution( const CCommon3dConvolutionDesc& desc, const float* sourceData,
	const float* filterData, const CConstFloatHandle* freeTermData, float* resultData )
{
	const float* freeTermDataRaw = freeTermData == nullptr ? nullptr : GetRaw( *freeTermData );

	const CBlobDesc& source = desc.Source;
	const CBlobDesc& result = desc.Result;
	const CBlobDesc& filter = desc.Filter;

	const int objectCount = source.ObjectCount();
	const int preparedWidth = filter.GeometricalSize();

	const int inputPreparedObjectSize = result.Width() * result.Depth() * result.Height() * preparedWidth * source.Channels();
	CFloatHandleStackVar inputPreparedData( mathEngine(), inputPreparedObjectSize );

	const int outputTempObjectSize = result.Width() * result.Depth() * result.Height() * result.Channels();
	CFloatHandleStackVar outputTempData( mathEngine(), outputTempObjectSize );

	const int outputCount = result.Width() * result.Depth();
	const int workingPreparedHeight = outputCount * result.Height();
	const int outputTempGeometricalSize = outputCount * result.Height();

	float* const inputPrepared = GetRaw( inputPreparedData.GetHandle() );
	float* const outputTemp = GetRaw( outputTempData.GetHandle() );

	for( int b = 0; b < objectCount; ++b ) {
		blob3dConvolutionPrepareInput( desc, inputPrepared, sourceData, b, result.Height(), /*outputStart*/0, outputCount );

		multiplyMatrixByTransposedMatrix( inputPrepared, workingPreparedHeight, preparedWidth * source.Channels(),
			preparedWidth * source.Channels(), filterData, filter.BatchWidth(), preparedWidth * source.Channels(),
			outputTemp, filter.BatchWidth(), nullptr );

		if( freeTermData != nullptr ) {
			addVectorToMatrixRows( outputTemp, outputTemp, outputTempGeometricalSize, result.Channels(),
				result.Channels(), result.Channels(), freeTermDataRaw );
		}

		// Transpose outputTemp to a part of result
		const float* tempData = outputTemp;
		float* outputDataPtr = resultData + b * result.ObjectSize();
		const int outputRowSize = result.Width() * result.Depth() * result.Channels();
		for( int tj = 0; tj < outputCount; ++tj ) {
			float* outputRowData = outputDataPtr;
			for( int ti = 0; ti < result.Height(); ++ti ) {
				dataCopy( outputRowData, tempData, result.Channels() );
				tempData += result.Channels();
				outputRowData += outputRowSize;
			}
			outputDataPtr += result.Channels();
		}
	}
}

void CCpuMathEngine::addMatrixToMatrix( float* first, int height,
	int width, int firstRowSize, const float* second, int secondRowSize )
{
	for( int j = 0; j < height; ++j ) {
		vectorAdd( first, second, first, width );
		first += firstRowSize;
		second += secondRowSize;
	}
}

void CCpuMathEngine::sumMatrixRowsAdd( float* result, const float* matrix,
	int matrixHeight, int matrixWidth )
{
	for( int i = 0; i < matrixHeight; i++ ) {
		vectorAdd(result, matrix, result, matrixWidth);
		matrix += matrixWidth;
	}
}

void CCpuMathEngine::blob3dConvolutionBackward( const CCommon3dConvolutionDesc& desc, const float* sourceData,
	const CConstFloatHandle& filterData, const CConstFloatHandle* freeTermData, float* resultData )
{
	const float* freeTermDataRaw = freeTermData == nullptr ? nullptr : GetRaw( *freeTermData );

	const CBlobDesc& source = desc.Result;
	const CBlobDesc& filter = desc.Filter;
	const CBlobDesc& result = desc.Source;

	// Transpose the filter for more convenient calculations
	const int filterForwardChannelsCount = filter.BatchWidth();
	const int filterForwardGeometricalSize = filter.GeometricalSize() * filter.Channels();
	const int filterForwardDataSize = filterForwardChannelsCount * filterForwardGeometricalSize;

	CFloatHandleStackVar filterForward( mathEngine(), filterForwardDataSize );
	TransposeMatrix( 1, filterData, filter.BatchWidth(), 1, filter.ObjectSize(), 1, filterForward.GetHandle(),
		filterForwardDataSize );
	float* const filterForwardPtr = GetRaw( filterForward.GetHandle() );

	const int inputGeo = source.GeometricalSize();
	// The number of rows in the output
	const int outputLineY = result.ObjectCount() * result.Height();
	// The output row size
	const int outputRowSize = result.Width() * result.Depth() * result.Channels();
	// The element size, including depth
	const int outputZSize = result.Depth() * result.Channels();

	const int tempDataSize = source.ObjectCount() * inputGeo * filterForwardGeometricalSize;
	CFloatHandleStackVar temp( mathEngine(), tempDataSize );
	float* const tempPtr = GetRaw( temp.GetHandle() );

	// The first step is to multiply the input and filter matrices
	const int sourceCount = source.ObjectCount() * inputGeo;
	multiplyMatrixByTransposedMatrix( sourceData,
		sourceCount, filterForwardChannelsCount, filterForwardChannelsCount,
		filterForwardPtr, filterForwardGeometricalSize, filterForwardChannelsCount,
		tempPtr, filterForwardGeometricalSize, nullptr );

	// The second step is to add the subvectors of the resulting 
	// matrix to the corresponding places in the output
	if( freeTermData == nullptr ) {
		vectorFill0( resultData, outputLineY * outputRowSize );
	}

	for( int step = 0; step < outputLineY; ++step ) {
		float* const outputData = resultData + step * outputRowSize;

		// Set the free term
		if( freeTermData != nullptr ) {
			setVectorToMatrixRows( outputData, result.Width() * result.Depth(),
				result.Channels(), freeTermDataRaw );
		}

		const int batch = step / result.Height();
		const int row = step % result.Height();
		const int inputRowStart = std::max( 0, ( row + desc.PaddingHeight - filter.Height() + desc.StrideHeight ) / desc.StrideHeight );
		const int filterRowBackStart = row - inputRowStart * desc.StrideHeight + desc.PaddingHeight;
		if( 0 > filterRowBackStart || filterRowBackStart >= filter.Height() ) {
			continue;
		}
		const int filterRowBackEnd = std::max( 0, filter.Height() + row - result.Height() - desc.PaddingHeight );

		int inputRow = inputRowStart;
		for( int filterRow = filterRowBackStart;
			filterRow >= filterRowBackEnd;
			filterRow -= desc.StrideHeight, ++inputRow ) {

			// temp stores the rows of filter multiplied by input; add them to the output rows in correct positions
			const float* tempRowData = tempPtr + ( ( batch * source.Height() + inputRow )
				* source.Width() * source.Depth() * filter.Height() + filterRow )
				* filter.Width() * filter.Depth() * filter.Channels();

			for( int inputX = 0; inputX < source.Width(); ++inputX ) {
				int xStart = inputX * desc.StrideWidth - desc.PaddingWidth;
				const int xEnd = std::min( result.Width(), xStart + filter.Width() );
				int xTempDataShift = 0;
				if( xStart < 0 ) {
					xTempDataShift = -xStart;
					xStart = 0;
				}

				// The start of the place in the output where to copy
				float* const outputLine = outputData + xStart * outputZSize;
				const float* tempData = tempRowData + inputX * source.Depth() *  filter.ObjectSize()
					+ xTempDataShift * filter.Depth() * filter.Channels();

				for( int z = -desc.PaddingDepth;
					z <= result.Depth() + desc.PaddingDepth - filter.Depth();
					z += desc.StrideDepth ) {
					int tempDataShift = 0;
					int toCopy = filter.Depth();
					int pos = z;
					if( pos < 0 ) {
						tempDataShift = -pos * filter.Channels();
						toCopy += pos;
						pos = 0;
					}
					if( pos + toCopy > result.Depth() ) {
						toCopy = result.Depth() - pos;
					}
					PRESUME_EXPR( toCopy > 0 );

					toCopy *= filter.Channels();
					float* outputVec = outputLine + pos * filter.Channels();
					addMatrixToMatrix( outputVec, xEnd - xStart, toCopy, outputZSize,
						tempData + tempDataShift, filter.Depth() * filter.Channels() );

					tempData += filter.ObjectSize();
				}
			}
		}
	}
}

void CCpuMathEngine::blob3dConvolutionLearnAdd( const CCommon3dConvolutionDesc& desc, const float* inputData,
	const float* outputDiffData, const CFloatHandle& filterDiffData, const CFloatHandle* freeTermDiffData,
	bool isFreeTermDiffFromInput )
{
	const CBlobDesc& input = desc.Source;
	const CBlobDesc& outputDiff = desc.Result;
	const CBlobDesc& filterDiff = desc.Filter;

	const int inputPreparedTempSize = outputDiff.GeometricalSize() * filterDiff.GeometricalSize() * input.Channels();
	CFloatHandleStackVar inputPreparedTemp( mathEngine(), inputPreparedTempSize );

	const int outputTempSize = outputDiff.ObjectSize();
	CFloatHandleStackVar outputTemp( mathEngine(), outputTempSize );

	float* const inputPreparedDataPtr = GetRaw( inputPreparedTemp.GetHandle() );
	float* const outputTempDataPtr = GetRaw( outputTemp.GetHandle() );
	float* const filterDiffDataPtr = GetRaw( filterDiffData );

	for( int b = 0; b < input.ObjectCount(); ++b ) {
		const float* outputDiffDataPtr = outputDiffData + b * outputDiff.ObjectSize();

		blob3dConvolutionPrepareInput( desc, inputPreparedDataPtr, inputData, b,
			outputDiff.Height(), 0, outputDiff.Width() * outputDiff.Depth() );

		transposeMatrix( 1, outputDiffDataPtr,
			outputDiff.Height(), 1, outputDiff.Width() * outputDiff.Depth(),
			outputDiff.Channels(), outputTempDataPtr );

		// Filter diff
		multiplyTransposedMatrixByMatrixAndAdd( outputTempDataPtr,
			outputDiff.GeometricalSize(), outputDiff.Channels(), outputDiff.Channels(),
			inputPreparedDataPtr, filterDiff.GeometricalSize() * input.Channels(), filterDiff.GeometricalSize() * input.Channels(),
			filterDiffDataPtr, filterDiff.GeometricalSize() * input.Channels(), nullptr );

		if( freeTermDiffData != nullptr ) {
			// Free term diff
			float* const freeTermDiffDataPtr = GetRaw( *freeTermDiffData );
			if( isFreeTermDiffFromInput ) {
				sumMatrixRowsAdd( freeTermDiffDataPtr, inputData + b * input.ObjectSize(), input.GeometricalSize(), input.Channels() );
			} else {
				sumMatrixRowsAdd( freeTermDiffDataPtr, outputDiffDataPtr, outputDiff.GeometricalSize(), outputDiff.Channels() );
			}
		}
	}
}

C3dConvolutionDesc* CCpuMathEngine::InitBlob3dConvolution( const CBlobDesc& source,
	int paddingHeight, int paddingWidth, int paddingDepth,
	int strideHeight, int strideWidth, int strideDepth,
	const CBlobDesc& filter, const CBlobDesc& result )
{
	CCommon3dConvolutionDesc *desc = new CCommon3dConvolutionDesc( source, result, filter,
		paddingHeight, paddingWidth, paddingDepth, strideHeight, strideWidth, strideDepth );
	return desc;
}

void CCpuMathEngine::Blob3dConvolution( const C3dConvolutionDesc& convDesc, const CConstFloatHandle& sourceData,
	const CConstFloatHandle& filterData, const CConstFloatHandle* freeTermData, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( filterData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	ASSERT_EXPR( freeTermData == 0 || freeTermData->GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* sourceDataRaw = GetRaw( sourceData );
	const float* filterDataRaw = GetRaw( filterData );
	const float* freeTermDataRaw = freeTermData == nullptr ? nullptr : GetRaw( *freeTermData );
	float* resultDataRaw = GetRaw( resultData );

	const CCommon3dConvolutionDesc& desc = static_cast<const CCommon3dConvolutionDesc&>( convDesc );

	if( desc.PaddingHeight == 0 && desc.PaddingWidth == 0 && desc.PaddingDepth == 0 && desc.Filter.ObjectSize() == desc.Filter.Channels() ) {
		blob3dConvolution1x1x1( desc.Source, desc.Filter, desc.Result, desc.StrideHeight, desc.StrideWidth, desc.StrideDepth,
			sourceDataRaw, filterDataRaw, freeTermDataRaw, resultDataRaw );
	} else {
		blob3dConvolution( desc, sourceDataRaw, filterDataRaw, freeTermData, resultDataRaw );
	}
}

void CCpuMathEngine::Blob3dConvolutionBackward( const C3dConvolutionDesc& convDesc, const CConstFloatHandle& sourceData,
	const CConstFloatHandle& filterData, const CConstFloatHandle* freeTermData, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( filterData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	ASSERT_EXPR( freeTermData == 0 || freeTermData->GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* sourceDataRaw = GetRaw( sourceData );
	const float* filterDataRaw = GetRaw( filterData );
	float* resultDataRaw = GetRaw( resultData );

	const CCommon3dConvolutionDesc& desc = static_cast<const CCommon3dConvolutionDesc&>( convDesc );

	if( desc.PaddingHeight == 0 && desc.PaddingWidth == 0 && desc.PaddingDepth == 0 && desc.Filter.ObjectSize() == desc.Filter.Channels() ) {
		blob3dConvolution1x1x1Backward( desc, sourceDataRaw, filterDataRaw, freeTermData, resultDataRaw );
	} else {
		blob3dConvolutionBackward( desc, sourceDataRaw, filterData, freeTermData, resultDataRaw );
	}
}

void CCpuMathEngine::Blob3dConvolutionLearnAdd( const C3dConvolutionDesc& convDesc, const CConstFloatHandle& inputData,
	const CConstFloatHandle& outputDiffData, const CFloatHandle& filterDiffData, const CFloatHandle* freeTermDiffData,
	bool isFreeTermDiffFromInput )
{
	ASSERT_EXPR( inputData.GetMathEngine() == this );
	ASSERT_EXPR( outputDiffData.GetMathEngine() == this );
	ASSERT_EXPR( filterDiffData.GetMathEngine() == this );
	ASSERT_EXPR( freeTermDiffData == 0 || freeTermDiffData->GetMathEngine() == this );
	CCpuExecutionScope scope;

	const CCommon3dConvolutionDesc& desc = static_cast<const CCommon3dConvolutionDesc&>( convDesc );

	if( desc.PaddingHeight == 0 && desc.PaddingWidth == 0 && desc.PaddingDepth == 0 && desc.Filter.ObjectSize() == desc.Filter.Channels() ) {
		blob3dConvolution1x1x1LearnAdd( desc, inputData, outputDiffData, filterDiffData, freeTermDiffData );
	} else {
		blob3dConvolutionLearnAdd( desc, GetRaw( inputData ), GetRaw( outputDiffData ), filterDiffData, freeTermDiffData, isFreeTermDiffFromInput );
	}
}

} // namespace NeoML
