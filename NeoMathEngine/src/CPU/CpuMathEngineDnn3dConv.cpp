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

#include <CpuMathEngine.h>
#include <float.h>
#include <CpuMathEngineOmp.h>
#include <MathEngineCommon.h>
#include <MathEngineDnnConv.h>
#include <MemoryHandleInternal.h>
#include <CpuMathEnginePrivate.h>

namespace NeoML {

void CCpuMathEngine::blob3dConvolution1x1x1Backward( const CCommon3dConvolutionDesc& desc,
	const float* outputDiffData, const float* filterData, const CFloatHandle* freeTermData,
	float* inputDiffData )
{
	float* freeTermDataRaw = freeTermData == nullptr ? nullptr : GetRaw( *freeTermData );

	const CBlobDesc& inputDiff = desc.Source;
	const CBlobDesc& outputDiff = desc.Result;

	bool isRepackNeeded = desc.StrideHeight > 1 || desc.StrideWidth > 1 || desc.StrideDepth > 1;

	CBlobDesc resultBlob = inputDiff;
	float* resultBlobData = inputDiffData;
	int resultBlobHolderSize = 0;
	if( isRepackNeeded ) {
		resultBlob = outputDiff;
		resultBlob.SetDimSize( BD_Channels, inputDiff.Channels() );
		resultBlobHolderSize = resultBlob.BlobSize();
	}

	CFloatHandleVar resultBlobHolder( mathEngine(), resultBlobHolderSize );

	if( isRepackNeeded ) {
		resultBlobData = GetRaw( resultBlobHolder.GetHandle() );
	}

	int objectCount = outputDiff.ObjectCount();

	const int curThreadCount = IsOmpRelevant( objectCount ) ? threadCount : 1;
	NEOML_OMP_NUM_THREADS( curThreadCount )
	{
		int batchStart;
		int batchCount;
		if( OmpGetTaskIndexAndCount( objectCount, batchStart, batchCount ) ) {
			float* resultData = resultBlobData + batchStart * resultBlob.ObjectSize();
			float* inputDiffDataPtr = inputDiffData + batchStart * inputDiff.ObjectSize();

			if( freeTermData != nullptr ) {
				setVectorToMatrixRows( inputDiffDataPtr, batchCount * inputDiff.GeometricalSize(),
					inputDiff.Channels(), freeTermDataRaw );
			}

			if( isRepackNeeded || freeTermData == nullptr ) {
				multiplyMatrixByMatrix( outputDiffData + batchStart * outputDiff.ObjectSize(),
					batchCount * outputDiff.GeometricalSize(), outputDiff.Channels(), outputDiff.Channels(),
					filterData, resultBlob.Channels(), resultBlob.Channels(),
					resultData, resultBlob.Channels() );
			} else {
				multiplyMatrixByMatrixAndAdd( outputDiffData + batchStart * outputDiff.ObjectSize(),
					batchCount * outputDiff.GeometricalSize(), outputDiff.Channels(), outputDiff.Channels(),
					filterData, resultBlob.Channels(), resultBlob.Channels(),
					resultData, resultBlob.Channels() );
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
	}
}

void CCpuMathEngine::blob3dConvolution1x1x1LearnAdd( const CCommon3dConvolutionDesc& desc, const CFloatHandle& inputData,
	const CFloatHandle& outputDiffData, const CFloatHandle& filterDiffData, const CFloatHandle* freeTermDiffData )
{
	const CBlobDesc& input = desc.Source;
	const CBlobDesc& outputDiff = desc.Result;
	const CBlobDesc& filterDiff = desc.Filter;

	bool isRepackNeeded = desc.StrideHeight > 1 || desc.StrideWidth > 1 || desc.StrideDepth > 1;
	CBlobDesc inputBlob = input;
	CFloatHandle inputBlobData = inputData;

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
		float* inputBlobDataPtr = GetRaw( inputBlobData );
		for( int b = 0; b < inputBlob.ObjectCount(); ++b ) {
			const float* inputRowData = inputDataPtr;
			for( int j = 0; j < inputBlob.Height(); ++j ) {
				const float* inputColData = inputRowData;
				for( int i = 0; i < inputBlob.Width(); ++i ) {
					const float* inputPixelData = inputColData;
					for( int k = 0; k < inputBlob.Depth(); ++k ) {
						vectorCopy( inputBlobDataPtr, inputPixelData, input.Channels() );
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
		filterDiff.BatchWidth() * inputBlob.Channels() );

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
					vectorCopy( inputPrepared, inputPrepared - rowsToCopy * filterRowSize,
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
						assert( -startI < colsToComplete );
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
								assert( -startK < depthToComplete );
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
								vectorCopy( inputPrepared,
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
	const float* filterData, const CFloatHandle* freeTermData, float* resultData )
{
	float* freeTermDataRaw = freeTermData == nullptr ? nullptr : GetRaw( *freeTermData );

	const CBlobDesc& source = desc.Source;
	const CBlobDesc& result = desc.Result;
	const CBlobDesc& filter = desc.Filter;

	const int objectCount = source.ObjectCount();

	int preparedWidth = filter.GeometricalSize();

	const int curThreadCount = IsOmpRelevant( objectCount * result.Width() * result.Depth(),
		static_cast<int64_t>( source.BlobSize() ) * filter.BlobSize() ) ? threadCount : 1;
	const int tempObjectCount = min( source.ObjectCount(), curThreadCount );

	const int inputPreparedObjectSize = result.Width() * result.Depth() * result.Height() * preparedWidth * source.Channels();
	CFloatHandleStackVar inputPreparedData( mathEngine(), tempObjectCount * inputPreparedObjectSize );
	float* inputPreparedDataPtr = GetRaw( inputPreparedData.GetHandle() );

	const int outputTempObjectSize = result.Width() * result.Depth() * result.Height() * result.Channels();
	CFloatHandleStackVar outputTempData( mathEngine(), tempObjectCount * outputTempObjectSize );
	float* outputTempDataPtr = GetRaw( outputTempData.GetHandle() );

	NEOML_OMP_NUM_THREADS( curThreadCount )
	{
		int batchStart;
		int batchCount;
		int outputStart;
		int outputCount;
		if( OmpGetTaskIndexAndCount2D( objectCount, result.Width() * result.Depth(),
			batchStart, batchCount, outputStart, outputCount ) )
		{
			int workingPreparedHeight = outputCount * result.Height();

			const int outputTempGeometricalSize = outputCount * result.Height();
			const int tempObjectIndex = source.ObjectCount() <= tempObjectCount ? batchStart : OmpGetThreadNum();

			float* inputPrepared = inputPreparedDataPtr + tempObjectIndex * inputPreparedObjectSize
				+ outputStart * result.Height() * preparedWidth * source.Channels();
			float* outputTemp = outputTempDataPtr + tempObjectIndex * outputTempObjectSize
				+ outputStart * result.Height() * result.Channels();

			for( int b = batchStart; b < batchStart + batchCount; ++b ) {
				blob3dConvolutionPrepareInput( desc, inputPrepared, sourceData, b, result.Height(), outputStart, outputCount );

				multiplyMatrixByTransposedMatrix( inputPrepared, workingPreparedHeight, preparedWidth * source.Channels(),
					preparedWidth * source.Channels(), filterData, filter.BatchWidth(), preparedWidth * source.Channels(),
					outputTemp, filter.BatchWidth() );

				if( freeTermData != nullptr ) {
					addVectorToMatrixRows( outputTemp, outputTemp, outputTempGeometricalSize, result.Channels(),
						result.Channels(), result.Channels(), freeTermDataRaw );
				}

				// Transpose outputTemp to a part of result
				const float* tempData = outputTemp;
				float* outputDataPtr = resultData + b * result.ObjectSize() + outputStart * result.Channels();
				int outputRowSize = result.Width() * result.Depth() * result.Channels();
				for( int tj = 0; tj < outputCount; ++tj ) {
					float* outputRowData = outputDataPtr;
					for( int ti = 0; ti < result.Height(); ++ti ) {
						vectorCopy( outputRowData, tempData, result.Channels() );
						tempData += result.Channels();
						outputRowData += outputRowSize;
					}
					outputDataPtr += result.Channels();
				}
			}
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
	result += matrixWidth;
}

void CCpuMathEngine::blob3dConvolutionBackward( const CCommon3dConvolutionDesc& desc, const float* sourceData,
	const CFloatHandle& filterData, const CFloatHandle* freeTermData, float* resultData )
{
	float* freeTermDataRaw = freeTermData == nullptr ? nullptr : GetRaw( *freeTermData );

	const CBlobDesc& source = desc.Result;
	const CBlobDesc& filter = desc.Filter;
	const CBlobDesc& result = desc.Source;

	// Transpose the filter for more convenient calculations
	const int filterForwardChannelsCount = filter.BatchWidth();
	const int filterForwardGeometricalSize = filter.GeometricalSize() * filter.Channels();
	const int filterForwardDataSize = filterForwardChannelsCount * filterForwardGeometricalSize;

	CFloatHandleVar filterForward( mathEngine(), filterForwardDataSize );
	TransposeMatrix( 1, filterData, filter.BatchWidth(), 1, filter.ObjectSize(), 1, filterForward.GetHandle(),
		filterForwardDataSize );
	float* filterForwardPtr = GetRaw( filterForward.GetHandle() );

	int inputGeo = source.GeometricalSize();
	// The number of rows in the output
	int outputLineY = result.ObjectCount() * result.Height();
	// The output row size
	int outputRowSize = result.Width() * result.Depth() * result.Channels();
	// The element size, including depth
	int outputZSize = result.Depth() * result.Channels();

	const int tempDataSize = source.ObjectCount() * inputGeo * filterForwardGeometricalSize;
	const int tempWidth = filterForwardGeometricalSize;
	CFloatHandleVar temp( mathEngine(), tempDataSize );
	float* tempPtr = GetRaw( temp.GetHandle() );

	const int curThreadCount = IsOmpRelevant( outputLineY, static_cast<int64_t>( source.BlobSize() ) * filter.BlobSize() )
		? threadCount : 1;

	NEOML_OMP_NUM_THREADS( curThreadCount )
	{
		// The first step is to multiply the input and filter matrices
		int inputStart;
		int inputCount;
		if( OmpGetTaskIndexAndCount( source.ObjectCount() * inputGeo, inputStart, inputCount ) ) {
			multiplyMatrixByTransposedMatrix( sourceData + inputStart * filterForwardChannelsCount,
				inputCount, filterForwardChannelsCount, filterForwardChannelsCount,
				filterForwardPtr, filterForwardGeometricalSize, filterForwardChannelsCount,
				tempPtr + inputStart * tempWidth, filterForwardGeometricalSize );
		}

		do {
			if( curThreadCount > 1 ) {
#pragma omp barrier
			}
		} while( 0 );

		// The second step is to add the subvectors of the resulting 
		// matrix to the corresponding places in the output
		int outputLineStart;
		int outputLineCount;
		if( OmpGetTaskIndexAndCount( outputLineY, outputLineStart, outputLineCount ) ) {
			if( freeTermData == 0 ) {
				vectorFill( resultData + outputLineStart * outputRowSize,
					0, outputLineCount * outputRowSize );
			}

			int outputLineEnd = outputLineStart + outputLineCount;
			for( int step = outputLineStart; step < outputLineEnd; ++step ) {
				float* outputData = resultData + step * outputRowSize;

				// Set the free term
				if( freeTermData != nullptr ) {
					setVectorToMatrixRows( outputData, result.Width() * result.Depth(),
						result.Channels(), freeTermDataRaw );
				}

				int batch = step / result.Height();
				int row = step % result.Height();
				int inputRowStart = ( row + desc.PaddingHeight - filter.Height() + desc.StrideHeight ) / desc.StrideHeight;
				if( inputRowStart < 0 ) {
					inputRowStart = 0;
				}
				int filterRowBackStart = row - inputRowStart * desc.StrideHeight + desc.PaddingHeight;
				if( 0 > filterRowBackStart || filterRowBackStart >= filter.Height() ) {
					continue;
				}
				int filterRowBackEnd = filter.Height() + row - result.Height() - desc.PaddingHeight;
				if( filterRowBackEnd < 0 ) {
					filterRowBackEnd = 0;
				}

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
						int xEnd = xStart + filter.Width();
						int xTempDataShift = 0;
						if( xStart < 0 ) {
							xTempDataShift = -xStart;
							xStart = 0;
						}
						if( xEnd > result.Width() ) {
							xEnd = result.Width();
						}

						// The start of the place in the output where to copy
						float* outputLine = outputData + xStart * outputZSize;
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
							assert( toCopy > 0 );

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
	}
}

void CCpuMathEngine::blob3dConvolutionLearnAdd( const CCommon3dConvolutionDesc& desc, const float* inputData,
	const float* outputDiffData, const CFloatHandle& filterDiffData, const CFloatHandle* freeTermDiffData,
	bool isFreeTermDiffFromInput )
{
	const CBlobDesc& input = desc.Source;
	const CBlobDesc& outputDiff = desc.Result;
	const CBlobDesc& filterDiff = desc.Filter;

	const int objectCount = input.ObjectCount();
	const int freeTermDiffSize = isFreeTermDiffFromInput ? filterDiff.Channels() : filterDiff.ObjectCount();
	const int inputPreparedTempSize = outputDiff.GeometricalSize() * filterDiff.GeometricalSize() * input.Channels();
	const int outputTempSize = outputDiff.ObjectSize();

	const int curThreadCount = IsOmpRelevant( objectCount ) ? threadCount : 1;

	COmpPrivate1DData inputPreparedTemp( curThreadCount, mathEngine(), inputPreparedTempSize );
	COmpPrivate1DData outputTemp( curThreadCount, mathEngine(), outputTempSize );
	COmpReduction1DData filterDiffItem( mathEngine(), filterDiffData, desc.Filter.BlobSize() );
	COmpReduction<COmpReduction1DData> filterDiffReduction( curThreadCount, filterDiffItem );
	
	unique_ptr<COmpReduction1DData> freeTermDiffItem( nullptr );
	unique_ptr<COmpReduction<COmpReduction1DData>> freeTermDiffReduction( nullptr );
	
	if( freeTermDiffData != nullptr ) {
		freeTermDiffItem.reset( new COmpReduction1DData( mathEngine(), *freeTermDiffData, freeTermDiffSize ) );
		freeTermDiffReduction.reset( new COmpReduction<COmpReduction1DData>( curThreadCount, *freeTermDiffItem ) );
	}

	NEOML_OMP_FOR_NUM_THREADS( curThreadCount )
		for( int b = 0; b < objectCount; ++b ) {
			const float* outputDiffDataPtr = outputDiffData + b * outputDiff.ObjectSize();
			float* inputPreparedDataPtr = GetRaw( inputPreparedTemp.GetPrivateData() );
			float* filterDiffReductionDataPtr = GetRaw( filterDiffReduction.GetPrivate().Data );
			float* outputTempDataPtr = GetRaw( outputTemp.GetPrivateData() );

			blob3dConvolutionPrepareInput( desc, inputPreparedDataPtr, inputData, b,
				outputDiff.Height(), 0, outputDiff.Width() * outputDiff.Depth() );

			transposeMatrix( 1, outputDiffDataPtr,
				outputDiff.Height(), 1, outputDiff.Width() * outputDiff.Depth(),
				outputDiff.Channels(), outputTempDataPtr );

			// Filter diff
			multiplyTransposedMatrixByMatrixAndAdd( outputTempDataPtr,
				outputDiff.GeometricalSize(), outputDiff.Channels(), outputDiff.Channels(),
				inputPreparedDataPtr, filterDiff.GeometricalSize() * input.Channels(), filterDiff.GeometricalSize() * input.Channels(),
				filterDiffReductionDataPtr, filterDiff.GeometricalSize() * input.Channels() );

			if( freeTermDiffData != nullptr ) {
				// Free term diff
				float* freeTermDiffReductionDataPtr = GetRaw( freeTermDiffReduction->GetPrivate().Data );
				if( isFreeTermDiffFromInput ) {
					sumMatrixRowsAdd( freeTermDiffReductionDataPtr,
						inputData + b * input.ObjectSize(), input.GeometricalSize(), input.Channels() );
				} else {
					sumMatrixRowsAdd( freeTermDiffReductionDataPtr,
						outputDiffDataPtr, outputDiff.GeometricalSize(), outputDiff.Channels() );
				}
			}
		}

	filterDiffReduction.Reduce();
	if( freeTermDiffData != 0 ) {
		freeTermDiffReduction->Reduce();
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

void CCpuMathEngine::Blob3dConvolution( const C3dConvolutionDesc& convDesc, const CFloatHandle& sourceData,
	const CFloatHandle& filterData, const CFloatHandle* freeTermData, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( filterData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	ASSERT_EXPR( freeTermData == 0 || freeTermData->GetMathEngine() == this );

	const float* sourceDataRaw = GetRaw( sourceData );
	const float* filterDataRaw = GetRaw( filterData );
	float* freeTermDataRaw = freeTermData == nullptr ? nullptr : GetRaw( *freeTermData );
	float* resultDataRaw = GetRaw( resultData );

	const CCommon3dConvolutionDesc& desc = static_cast<const CCommon3dConvolutionDesc&>( convDesc );

	if( desc.PaddingHeight == 0 && desc.PaddingWidth == 0 && desc.PaddingDepth == 0 && desc.Filter.ObjectSize() == desc.Filter.Channels() ) {
		blob3dConvolution1x1x1( desc.Source, desc.Filter, desc.Result, desc.StrideHeight, desc.StrideWidth, desc.StrideDepth,
			sourceDataRaw, filterDataRaw, freeTermDataRaw, resultDataRaw );
	} else {
		blob3dConvolution( desc, sourceDataRaw, filterDataRaw, freeTermData, resultDataRaw );
	}
}

void CCpuMathEngine::Blob3dConvolutionBackward( const C3dConvolutionDesc& convDesc, const CFloatHandle& sourceData,
	const CFloatHandle& filterData, const CFloatHandle* freeTermData, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( filterData.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	ASSERT_EXPR( freeTermData == 0 || freeTermData->GetMathEngine() == this );

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

void CCpuMathEngine::Blob3dConvolutionLearnAdd( const C3dConvolutionDesc& convDesc, const CFloatHandle& inputData,
	const CFloatHandle& outputDiffData, const CFloatHandle& filterDiffData, const CFloatHandle* freeTermDiffData, bool isFreeTermDiffFromInput )
{
	ASSERT_EXPR( inputData.GetMathEngine() == this );
	ASSERT_EXPR( outputDiffData.GetMathEngine() == this );
	ASSERT_EXPR( filterDiffData.GetMathEngine() == this );
	ASSERT_EXPR( freeTermDiffData == 0 || freeTermDiffData->GetMathEngine() == this );

	const CCommon3dConvolutionDesc& desc = static_cast<const CCommon3dConvolutionDesc&>( convDesc );

	if( desc.PaddingHeight == 0 && desc.PaddingWidth == 0 && desc.PaddingDepth == 0 && desc.Filter.ObjectSize() == desc.Filter.Channels() ) {
		blob3dConvolution1x1x1LearnAdd( desc, inputData, outputDiffData, filterDiffData, freeTermDiffData );
	} else {
		blob3dConvolutionLearnAdd( desc, GetRaw( inputData ), GetRaw( outputDiffData ), filterDiffData, freeTermDiffData, isFreeTermDiffFromInput );
	}
}

} // namespace NeoML
