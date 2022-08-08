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
#include <CpuMathEnginePrivate.h>
#include <CpuExecutionScope.h>
#include <MemoryHandleInternal.h>
#include <MathEngineCommon.h>
#include <math.h>

namespace NeoML {

static void subVectorFromMatrixRows(CCpuMathEngine* engine, const CConstFloatHandle& matrixHandle, const CFloatHandle& resultHandle,
	int matrixHeight, int matrixWidth, const CConstFloatHandle& vectorHandle)
{
	CConstFloatHandle matrix = matrixHandle;
	CFloatHandle result = resultHandle;

	for(int i = 0; i < matrixHeight; i++) {
		engine->VectorSub( matrix, vectorHandle, result, matrixWidth );
		matrix += matrixWidth;
		result += matrixWidth;
	}
}

static inline void batchTransposePlainMatrix( int batchSize, const float* first,
	int height, int width, float* result )
{
	int objectSize = height * width;
	int firstRowSize = width;
	int resultRowSize = height;

	int height4 = GetCount4( height );
	int width4 = GetCount4( width );

	CMatrixBlock4x4 block;

	for( int b = 0; b < batchSize; ++b ) {
		const float* firstStart = first;
		float* resultStart = result;
		for( int j = 0; j < height4; ++j ) {
			const float* firstData = firstStart;
			float* resultData = resultStart;
			for( int i = 0; i < width4; ++i ) {
				block.Load4x4( firstData, firstRowSize );
				block.Transpose();
				block.Store4x4( resultData, resultRowSize );

				firstData += 4;
				resultData += resultRowSize * 4;
			}

			if( width > 0 ) {
				block.Load4xX( firstData, width, firstRowSize );
				block.Transpose();
				block.StoreYx4( resultData, width, resultRowSize );
			}

			firstStart += firstRowSize * 4;
			resultStart += 4;
		}

		if( height > 0 ) {
			const float* firstData = firstStart;
			float* resultData = resultStart;
			for( int i = 0; i < width4; ++i ) {
				block.LoadYx4( firstData, height, firstRowSize );
				block.Transpose();
				block.Store4xX( resultData, height, resultRowSize );

				firstData += 4;
				resultData += resultRowSize * 4;
			}

			if( width > 0 ) {
				block.LoadYxX( firstData, height, width, firstRowSize );
				block.Transpose();
				block.StoreYxX( resultData, width, height, resultRowSize );
			}

			firstStart += firstRowSize * 4;
			resultStart += 4;
		}

		first += objectSize;
		result += objectSize;
	}
}

template<class T>
inline void CCpuMathEngine::transposeMatrixImpl( int batchSize, const T* first,
	int height, int medium, int width, int channels, T* result )
{
	// Transpose B x 1 x M x W x C -> B x W x M x 1 x C
	// is equivalent to B x M x 1 x W x C -> B x W x 1 x M x C
	if( medium != 1 && height == 1 ) {
		swap( medium, height );
	}

	// Same goes for W == 1 && H != 1
	if( medium != 1 && width == 1 ) {
		swap( medium, width );
	}

	if( medium == 1 && ( height == 1 || width == 1 ) ) {
		dataCopy( result, first, batchSize * height * medium * width * channels );
		return;
	}

	if( medium == 1 && channels == 1 ) {
		static_assert( sizeof(float) == sizeof(T), "Size of float isn't equal to size of T." );
		batchTransposePlainMatrix( batchSize, reinterpret_cast<const float*>( first ),
			height, width, reinterpret_cast<float*>( result ) );
		return;
	}

	int objectSize = height * width * medium * channels;

	int resultRowSize = height * medium * channels;

	for( int b = 0; b < batchSize; ++b ) {
		T* resultColumnStart = result;
		for( int j = 0; j < height; ++j ) {
			T* resultMediumStart = resultColumnStart;
			for( int m = 0; m < medium; ++m ) {
				T* resultItem = resultMediumStart;
				for( int i = 0; i < width; ++i ) {
					dataCopy( resultItem, first, channels );
					resultItem += resultRowSize;
					first += channels;
				}
				resultMediumStart += channels * height;
			}
			resultColumnStart += channels;
		}
		result += objectSize;
	}
}

void CCpuMathEngine::TransposeMatrix( int batchSize, const CConstFloatHandle& firstHandle,
	int height, int medium, int width, int channels, const CFloatHandle& resultHandle, int )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	transposeMatrix( batchSize, GetRaw( firstHandle ), height, medium, width, channels, GetRaw( resultHandle ) );
}

void CCpuMathEngine::TransposeMatrix( int batchSize, const CConstIntHandle& firstHandle,
	int height, int medium, int width, int channels, const CIntHandle& resultHandle, int )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	transposeMatrix( batchSize, GetRaw( firstHandle ), height, medium, width, channels, GetRaw( resultHandle ) );
}

void CCpuMathEngine::transposeMatrix( int batchSize, const float* firstHandle,
	int height, int medium, int width, int channels, float* resultHandle )
{
	transposeMatrixImpl( batchSize, firstHandle, height, medium, width, channels, resultHandle );
}

void CCpuMathEngine::transposeMatrix( int batchSize, const int* firstHandle,
	int height, int medium, int width, int channels, int* resultHandle )
{
	transposeMatrixImpl( batchSize, firstHandle, height, medium, width, channels, resultHandle );
}

void CCpuMathEngine::addVectorToMatrixRows( const float* matrix, float* result,
	int matrixHeight, int matrixWidth, int matrixRowSize, int resultRowSize, const float* vector)
{
	for(int i = 0; i < matrixHeight; i++) {
		vectorAdd( matrix, vector, result, matrixWidth );
		matrix += matrixRowSize;
		result += resultRowSize;
	}
}

void CCpuMathEngine::SetVectorToMatrixRows( const CFloatHandle& resultHandle,
	int matrixHeight, int matrixWidth, const CConstFloatHandle& vectorHandle )
{
	CCpuExecutionScope scope;
	
	float* result = GetRaw( resultHandle );
	const float* vector = GetRaw( vectorHandle );

	const int curThreadCount = IsOmpRelevant( matrixHeight, matrixHeight * matrixWidth ) ? threadCount : 1;
	NEOML_OMP_FOR_NUM_THREADS( curThreadCount )
	for(int i = 0; i < matrixHeight; i++) {
		dataCopy( result + i * matrixWidth, vector, matrixWidth );
	}
}

void CCpuMathEngine::setVectorToMatrixRows( float* result,
	int matrixHeight, int matrixWidth, const float* vector)
{
	for(int i = 0; i < matrixHeight; i++) {
		dataCopy( result, vector, matrixWidth );
		result += matrixWidth;
	}
}

void CCpuMathEngine::AddVectorToMatrixColumns(const CConstFloatHandle& matrixHandle, const CFloatHandle& resultHandle,
	int matrixHeight, int matrixWidth, const CConstFloatHandle& vectorHandle)
{
	ASSERT_EXPR( matrixHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( vectorHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* matrix = GetRaw( matrixHandle );
	float* result = GetRaw( resultHandle );
	const float* vector = GetRaw( vectorHandle );

	for(int i = 0; i < matrixHeight; ++i) {
		vectorAddValue(matrix, result, matrixWidth, *vector);
		matrix += matrixWidth;
		result += matrixWidth;
		++vector;
	}
}

void CCpuMathEngine::AddVectorToMatrixRows( int batchSize, const CConstFloatHandle& matrixHandle, const CFloatHandle& resultHandle,
	int matrixHeight, int matrixWidth, const CConstFloatHandle& vectorHandle )
{
	CCpuExecutionScope scope;

	float* result = GetRaw( resultHandle );
	const float* matrix = GetRaw( matrixHandle );
	const float* vector = GetRaw( vectorHandle );

	const int matrixSize = matrixHeight * matrixWidth;
	const int tasks = batchSize * matrixSize;
	const int curThreadCount = IsOmpRelevant(tasks, tasks) ? threadCount : 1;
	NEOML_OMP_NUM_THREADS( curThreadCount )
	{
		int batchStart;
		int batchCount;
		int heightStart;
		int heightCount;
		int widthStart;
		int widthCount;
		if( OmpGetTaskIndexAndCount3D(batchSize, 1, matrixHeight, 1, matrixWidth, 1, batchStart, batchCount, heightStart, heightCount, widthStart, widthCount) ) {
			const int offset = batchStart * matrixSize + heightStart * matrixWidth + widthStart;
			float* outputData = result + offset;
			const float* inputData = matrix + offset;
			const float* vectorData = vector + batchStart* matrixWidth + widthStart;

			for( int i = 0; i < batchCount; ++i ) {
				addVectorToMatrixRows(inputData, outputData, heightCount, widthCount,
					matrixWidth, matrixWidth, vectorData);
				inputData += matrixSize;
				outputData += matrixSize;
				vectorData += matrixWidth;
			}
		}
	}
}

void CCpuMathEngine::RowMultiplyMatrixByMatrix(const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, int height, int width, const CFloatHandle& resultHandle)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw( firstHandle );
	const float* second = GetRaw( secondHandle );
	float* result = GetRaw( resultHandle );

	for(int i = 0; i < height; ++i) {
		vectorDotProduct(first, second, width, result);
		first += width;
		second += width;
		++result;
	}
}

static void ColumnMultiplyMatrixByMatrix(CCpuMathEngine* engine, const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, int height, int width, const CFloatHandle& resultHandle)
{
	CCpuExecutionScope scope;

	CConstFloatHandle first = firstHandle;
	CConstFloatHandle second = secondHandle;

	engine->VectorEltwiseMultiply(first, second, resultHandle, width);
	for(int j = 1; j < height; ++j) {
		first += width;
		second += width;
		engine->VectorEltwiseMultiplyAdd(first, second, resultHandle, width);
	}
}

void CCpuMathEngine::AddVectorToMatrixColumns(const CConstIntHandle& matrixHandle, const CIntHandle& resultHandle,
	int matrixHeight, int matrixWidth, const CConstIntHandle& vectorHandle)
{
	CCpuExecutionScope scope;

	CConstIntHandle matrix = matrixHandle;
	CIntHandle result = resultHandle;
	CConstIntHandle vector = vectorHandle;

	for(int i = 0; i < matrixHeight; ++i) {
		VectorAddValue(matrix, result, matrixWidth, vector);
		matrix += matrixWidth;
		result += matrixWidth;
		++vector;
	}
}

void CCpuMathEngine::SubVectorFromMatrixColumns(const CConstFloatHandle& matrixHandle, const CFloatHandle& resultHandle,
	int matrixHeight, int matrixWidth, const CConstFloatHandle& vectorHandle)
{
	CCpuExecutionScope scope;

	CConstFloatHandle matrix = matrixHandle;
	CFloatHandle result = resultHandle;
	const float* vector = GetRaw( vectorHandle );

	for(int i = 0; i < matrixHeight; ++i) {
		float value = -(*vector++);
		VectorAddValue(matrix, result, matrixWidth, CConstFloatHandle( CMemoryHandleInternal::CreateMemoryHandle( this, &value )));
		matrix += matrixWidth;
		result += matrixWidth;
	}
}

void CCpuMathEngine::SumMatrixColumns(const CFloatHandle& resultHandle, const CConstFloatHandle& matrixHandle,
	int matrixHeight, int matrixWidth)
{
	CCpuExecutionScope scope;

	CConstFloatHandle matrix = matrixHandle;
	CFloatHandle result = resultHandle;

	for(int j = 0; j < matrixHeight; ++j) {
		VectorSum(matrix, matrixWidth, result);
		matrix += matrixWidth;
		++result;
	}
}

void CCpuMathEngine::MatrixColumnsEltwiseDivide( const CConstFloatHandle& matrixHandle, int matrixHeight, int matrixWidth,
	const CConstFloatHandle& vectorHandle, const CFloatHandle& resultHandle )
{
	CCpuExecutionScope scope;

	const float* matrix = GetRaw( matrixHandle );
	const float* vector = GetRaw( vectorHandle );
	float* result = GetRaw( resultHandle );

	for( int i = 0; i < matrixHeight; i++ ) {
		for( int j = 0; j < matrixWidth; j++ ) {
			*result = *matrix / *vector;
			result++;
			matrix++;
		}
		vector++;
	}
}

void CCpuMathEngine::sumMatrixColumnsAdd(const CFloatHandle& resultHandle, const CConstFloatHandle& matrixHandle,
	int matrixHeight, int matrixWidth)
{
	CConstFloatHandle matrix = matrixHandle;

	CFloatHandle result = resultHandle;
	for(int j = 0; j < matrixHeight; ++j) {
		VectorSumAdd(matrix, matrixWidth, result);
		matrix += matrixWidth;
		++result;
	}
}

void CCpuMathEngine::SumMatrixRows(int batchSize,
	const CFloatHandle& resultHandle, const CConstFloatHandle& matrixHandle, int matrixHeight, int matrixWidth)
{
	CCpuExecutionScope scope;

	VectorFill(resultHandle, 0.f, batchSize * matrixWidth);
	SumMatrixRowsAdd(batchSize, resultHandle, matrixHandle, matrixHeight, matrixWidth);
}

void CCpuMathEngine::SumMatrixRows(int batchSize, const CIntHandle& resultHandle, const CConstIntHandle& matrixHandle,
	int matrixHeight, int matrixWidth)
{
	CCpuExecutionScope scope;

	VectorFill( resultHandle, 0, batchSize * matrixWidth );
	CConstIntHandle matrix = matrixHandle;
	CIntHandle result = resultHandle;
	for( int i = 0; i < batchSize; ++i ) {
		for( int j = 0; j < matrixHeight; j++ ) {
			VectorAdd(result, matrix, result, matrixWidth);
			matrix += matrixWidth;
		}
		result += matrixWidth;
	}
}

void CCpuMathEngine::SumMatrixRowsAdd(int batchSize,
	const CFloatHandle& resultHandle, const CConstFloatHandle& matrixHandle, int matrixHeight, int matrixWidth)
{
	CCpuExecutionScope scope;

	CConstFloatHandle matrix = matrixHandle;
	CFloatHandle result = resultHandle;
	for( int i = 0; i < batchSize; ++i ) {
		for( int j = 0; j < matrixHeight; j++ ) {
			VectorAdd(result, matrix, result, matrixWidth);
			matrix += matrixWidth;
		}
		result += matrixWidth;
	}
}

void CCpuMathEngine::findMaxValueInColumns( float* resultHandle, const float* matrixHandle,
	int matrixHeight, int matrixWidth )
{
	if( matrixHeight == 1 ) {
		dataCopy( resultHandle, matrixHandle, matrixWidth );
		return;
	}

	const float* nextRow = matrixHandle + matrixWidth;
	vectorEltwiseMax( matrixHandle, nextRow, resultHandle, matrixWidth );

	for( int i = 2; i < matrixHeight; ++i ) {
		nextRow += matrixWidth;
		vectorEltwiseMax( resultHandle, nextRow, resultHandle, matrixWidth );
	}
}

void CCpuMathEngine::VectorMultichannelLookupAndCopy( int batchSize, int channelCount, const CConstFloatHandle& inputHandle,
	const CConstFloatHandle* lookupHandles, const CLookupDimension* lookupDimensions, int lookupCount,
	const CFloatHandle& outputHandle, int outputChannels )
{
	ASSERT_EXPR(lookupCount <= channelCount);
	CCpuExecutionScope scope;

	const float* inputStart = GetRaw(inputHandle);
	float* outputStart = GetRaw(outputHandle);

	const int curThreadCount = IsOmpRelevant( batchSize, batchSize * outputChannels ) ? threadCount : 1;
	NEOML_OMP_FOR_NUM_THREADS( curThreadCount )
	for(int i = 0; i < batchSize; ++i) {
		const float* input = inputStart + i * channelCount;
		float* output = outputStart + i * outputChannels;
		for(int j = 0; j < lookupCount; ++j) {
			int index = (int)*input;
			input++;
			PRESUME_EXPR(0 <= index && index < lookupDimensions[j].VectorCount);
			int vectorSize = lookupDimensions[j].VectorSize;
			dataCopy(output, GetRaw(lookupHandles[j]) + index * vectorSize, vectorSize);
			output += vectorSize;
		}
		int remained = channelCount - lookupCount;
		if(remained > 0) {
			dataCopy(output, input, remained);
		}
	}
}

void CCpuMathEngine::VectorMultichannelLookupAndCopy(int batchSize, int channelCount, const CConstIntHandle& inputHandle,
	const CConstFloatHandle* lookupHandles, const CLookupDimension* lookupDimensions, int lookupCount,
	const CFloatHandle& outputHandle, int outputChannels)
{
	ASSERT_EXPR(lookupCount == channelCount);
	CCpuExecutionScope scope;

	const int* inputStart = GetRaw( inputHandle );
	float* outputStart = GetRaw( outputHandle );

	const int curThreadCount = IsOmpRelevant( batchSize, batchSize * outputChannels ) ? threadCount : 1;
	NEOML_OMP_FOR_NUM_THREADS( curThreadCount )
	for(int i = 0; i < batchSize; ++i) {
		const int* input = inputStart + i * channelCount;
		float* output = outputStart + i * outputChannels;
		for(int j = 0; j < lookupCount; ++j) {
			if(j < channelCount) {
				int index = *input;
				input++;
				PRESUME_EXPR(0 <= index && index < lookupDimensions[j].VectorCount);
				int vectorSize = lookupDimensions[j].VectorSize;
				dataCopy(output, GetRaw(lookupHandles[j]) + index * vectorSize, vectorSize);
				output += vectorSize;
			}
		}
	}
}

void CCpuMathEngine::VectorMultichannelLookupAndCopy(int batchSize, int channelCount, const CConstIntHandle& inputHandle,
	const CConstIntHandle* lookupHandles, const CLookupDimension* lookupDimensions, int lookupCount,
	const CIntHandle& outputHandle, int outputChannels)
{
	ASSERT_EXPR(lookupCount <= channelCount);
	CCpuExecutionScope scope;

	const int* inputStart = GetRaw(inputHandle);
	int* outputStart = GetRaw(outputHandle);

	const int curThreadCount = IsOmpRelevant( batchSize, batchSize * outputChannels ) ? threadCount : 1;
	NEOML_OMP_FOR_NUM_THREADS( curThreadCount )
	for(int i = 0; i < batchSize; ++i) {
		const int* input = inputStart + i * channelCount;
		int* output = outputStart + i * outputChannels;
		for(int j = 0; j < lookupCount; ++j) {
			int index = *input;
			input++;
			PRESUME_EXPR(0 <= index && index < lookupDimensions[j].VectorCount);
			int vectorSize = lookupDimensions[j].VectorSize;
			dataCopy(output, GetRaw(lookupHandles[j]) + index * vectorSize, vectorSize);
			output += vectorSize;
		}
		int remained = channelCount - lookupCount;
		if(remained > 0) {
			dataCopy(output, input, remained);
		}
	}
}

void CCpuMathEngine::VectorMultichannelLookupAndAddToTable(int batchSize, int channelCount, const CConstFloatHandle& inputHandle,
	const CFloatHandle* lookupHandles, const CLookupDimension* lookupDimensions, int lookupCount,
	const CConstFloatHandle& multHandle, const CConstFloatHandle& matrixHandle, int /*outputChannels*/)
{
	ASSERT_EXPR(lookupCount <= channelCount);
	CCpuExecutionScope scope;

	CConstFloatHandle input = inputHandle;
	CConstFloatHandle matrix = matrixHandle;

	for(int i = 0; i < batchSize; ++i) {
		for(int j = 0; j < lookupCount; ++j) {
			if(j < channelCount) {
				int index = (int)input.GetValue();
				input++;
				PRESUME_EXPR(0 <= index && index < lookupDimensions[j].VectorCount);
				int vectorSize = lookupDimensions[j].VectorSize;
				CFloatHandle pos = lookupHandles[j] + index * vectorSize;
				VectorMultiplyAndAdd(pos, matrix, pos, vectorSize, multHandle);
				matrix += vectorSize;
			}
		}
		// skip unmapped updates
		int remained = channelCount - lookupCount;
		input += remained;
		matrix += remained;
	}
}

void CCpuMathEngine::VectorMultichannelLookupAndAddToTable(int batchSize, int channelCount, const CConstIntHandle& inputHandle,
	const CFloatHandle* lookupHandles, const CLookupDimension* lookupDimensions, int lookupCount,
	const CConstFloatHandle& multHandle,
	const CConstFloatHandle& matrixHandle, int /*outputChannels*/)
{
	ASSERT_EXPR(lookupCount <= channelCount);
	CCpuExecutionScope scope;

	CConstIntHandle input = inputHandle;
	CConstFloatHandle matrix = matrixHandle;

	for(int i = 0; i < batchSize; ++i) {
		for(int j = 0; j < lookupCount; ++j) {
			if(j < channelCount) {
				int index = input.GetValue();
				input++;
				PRESUME_EXPR(0 <= index && index < lookupDimensions[j].VectorCount);
				int vectorSize = lookupDimensions[j].VectorSize;
				CFloatHandle pos = lookupHandles[j] + index * vectorSize;
				VectorMultiplyAndAdd(pos, matrix, pos, vectorSize, multHandle);
				matrix += vectorSize;
			}
		}
		// skip unmapped updates
		int remained = channelCount - lookupCount;
		input += remained;
		matrix += remained;
	}
}

void CCpuMathEngine::EnumBinarization(int batchSize,
	const CConstFloatHandle& inputHandle, int enumSize, const CFloatHandle& resultHandle)
{
	CCpuExecutionScope scope;

	const float* input = GetRaw(inputHandle);
	float* result = GetRaw(resultHandle);

	VectorFill(resultHandle, 0, batchSize * enumSize);

	for(int i = 0; i < batchSize; ++i) {
		int enumValue = (int)(*input++);
		if(enumValue >= 0) {
			PRESUME_EXPR(enumValue < enumSize);
			result[enumValue] = 1;
		}
		result += enumSize;
	}
}

void CCpuMathEngine::EnumBinarization(int batchSize,
	const CConstIntHandle& inputHandle, int enumSize, const CFloatHandle& resultHandle)
{
	CCpuExecutionScope scope;

	const int* input = GetRaw(inputHandle);
	float* result = GetRaw(resultHandle);

	VectorFill(resultHandle, 0, batchSize * enumSize);

	for(int i = 0; i < batchSize; ++i) {
		int enumValue = *input++;
		if(enumValue >= 0) {
			PRESUME_EXPR(enumValue < enumSize);
			result[enumValue] = 1;
		}
		result += enumSize;
	}
}

void CCpuMathEngine::AddMatrixElementsToVector(const CConstFloatHandle& matrixHandle, int height, int width,
	const CConstIntHandle& indicesHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(vectorSize >= height);
	CCpuExecutionScope scope;

	const float* matrix = GetRaw(matrixHandle);
	const int* indices = GetRaw(indicesHandle);
	float* result = GetRaw(resultHandle);

	for(int j = 0; j < height; ++j) {
		int index = *indices++;
		if(index >= 0 && index < width) {
			*result += matrix[index];
		}
		++result;
		matrix += width;
	}
}

void CCpuMathEngine::AddDiagMatrixToMatrix( const CConstFloatHandle& diagMatrix, const CConstFloatHandle& matrix,
	int height, int width, const CFloatHandle& result )
{
	CCpuExecutionScope scope;

	const float* diagMatrixPtr = GetRaw(diagMatrix);
	const float* matrixPtr = GetRaw(matrix);
	float* resultPtr = GetRaw(result);
	for( int i = 0; i < height; i++ ) {
		for( int j = 0; j < width; j++ ) {
			*resultPtr = *matrixPtr;
			if( i == j ) {
				*resultPtr += *diagMatrixPtr;
			}
			resultPtr++;
			matrixPtr++;
		}
		diagMatrixPtr++;
	}
}

void CCpuMathEngine::AddMatrixElementsToMatrix(const CConstFloatHandle& matrixHandle, int height, int width,
	const CFloatHandle& resultHandle, const CConstIntHandle& indicesHandle)
{
	CCpuExecutionScope scope;

	const float* matrix = GetRaw(matrixHandle);
	const int* indices = GetRaw(indicesHandle);
	float* result = GetRaw(resultHandle);

	for(int j = 0; j < height; ++j) {
		int index = *indices++;
		if(index >= 0 && index < width) {
			result[index] += matrix[index];
		}
		result += width;
		matrix += width;
	}
}

void CCpuMathEngine::AddMatrixElementsToVector(const CConstFloatHandle& matrixHandle, int /*height*/, int width,
	const CConstIntHandle& rowIndicesHandle, const CConstIntHandle& columnIndicesHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	CCpuExecutionScope scope;

	const float* matrix = GetRaw(matrixHandle);
	const int* rowIndices = GetRaw(rowIndicesHandle);
	const int* columnIndices = GetRaw(columnIndicesHandle);
	float* result = GetRaw(resultHandle);

	for(int i = 0; i < vectorSize; ++i) {
		*result += matrix[*rowIndices * width + *columnIndices];
		++result;
		++rowIndices;
		++columnIndices;
	}
}

void CCpuMathEngine::AddVectorToMatrixElements(const CFloatHandle& matrixHandle, int height, int width,
	const CConstIntHandle& indicesHandle, const CConstFloatHandle& vectorHandle)
{
	CCpuExecutionScope scope;

	float* matrix = GetRaw(matrixHandle);
	const int* indices = GetRaw(indicesHandle);
	const float* vector = GetRaw(vectorHandle);

	for(int j = 0; j < height; ++j) {
		int index = *indices++;
		if(index < 0 || index >= width) {
			++vector;
		} else {
			matrix[index] += *vector++;
		}
		matrix += width;
	}
}

void CCpuMathEngine::AddVectorToMatrixElements(const CFloatHandle& matrixHandle, int /*height*/, int width,
	const CConstIntHandle& rowIndicesHandle, const CConstIntHandle& columnIndicesHandle,
	const CConstFloatHandle& vectorHandle, int vectorSize)
{
	CCpuExecutionScope scope;

	float* matrix = GetRaw(matrixHandle);
	const int* rowIndices = GetRaw(rowIndicesHandle);
	const int* columnIndices = GetRaw(columnIndicesHandle);
	const float* vector = GetRaw(vectorHandle);

	for(int i = 0; i < vectorSize; ++i) {
		matrix[rowIndices[i] * width + columnIndices[i]] += vector[i];
	}
}

void CCpuMathEngine::LookupAndSum(const CConstIntHandle& indicesHandle, int batchSize, int indexCount,
	const CConstFloatHandle& tableHandle, int vectorSize, const CFloatHandle& result)
{
	CCpuExecutionScope scope;

	const int* indicesStart = GetRaw(indicesHandle);
	float* outputStart = GetRaw(result);
	const float* table = GetRaw(tableHandle);

	const int curThreadCount = IsOmpRelevant( batchSize, batchSize * indexCount * vectorSize ) ? threadCount : 1;
	NEOML_OMP_FOR_NUM_THREADS( curThreadCount )
	for(int b = 0; b < batchSize; ++b) {
		float* output = outputStart + b * vectorSize;
		const int* indices = indicesStart + b * indexCount;
		int index = *indices;
		indices++;
		if(index >= 0) {
			dataCopy(output, table + vectorSize * index, vectorSize);
		} else {
			vectorFill(output, 0.f, vectorSize);
		}
		for(int elem = 1; elem < indexCount; ++elem) {
			index = *indices;
			indices++;
			if(index >= 0) {
				vectorAdd(output, table + vectorSize * index, output, vectorSize);
			}
		}
	}
}

void CCpuMathEngine::LookupAndAddToTable(const CConstIntHandle& indicesHandle, int batchSize, int indexCount,
	const CConstFloatHandle& additionsHandle, int vectorSize, const CFloatHandle& tableHandle, int vectorCount)
{
	ASSERT_EXPR( indicesHandle.GetMathEngine() == this );
	ASSERT_EXPR( tableHandle.GetMathEngine() == this );
	ASSERT_EXPR( additionsHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const int* indices = GetRaw( indicesHandle );
	const float* additions = GetRaw( additionsHandle );
	float* table = GetRaw( tableHandle );

	vectorFill( table, 0.f, vectorCount * vectorSize );

	for( int b = 0; b < batchSize; ++b ) {
		for( int elem = 0; elem < indexCount; ++elem ) {
			int index = *indices;
			indices++;
			if( index >= 0 ) {
				vectorAdd( table + index * vectorSize, additions, table + index * vectorSize, vectorSize );
			}
		}
		additions += vectorSize;
	}
}

void CCpuMathEngine::findMaxValueInColumns( float* result, int* rowIndices,
	const float* matrix, int matrixHeight, int matrixWidth )
{
	// Copy the first row
	dataCopy( result, matrix, matrixWidth );
	memset( rowIndices, 0, matrixWidth * sizeof( *rowIndices ) );
	matrix += matrixWidth;
	// Process the rest
	for( int i = 0; i < matrixHeight - 1; i++ ) {
		float* vectorPtr = result;
		int* indicesPtr = rowIndices;
		for( int j = 0; j < matrixWidth; j++ ) {
			if( *matrix > *vectorPtr ) {
				*vectorPtr = *matrix;
				*indicesPtr = i + 1;
			}
			matrix += 1;
			vectorPtr += 1;
			indicesPtr += 1;
		}
	}
}

void CCpuMathEngine::MultiplyDiagMatrixByMatrix( const CConstFloatHandle& firstHandle, int firstSize,
	const CConstFloatHandle& secondHandle, int secondWidth,
	const CFloatHandle& resultHandle, int resultBufferSize )
{
	ASSERT_EXPR( resultBufferSize >= firstSize * secondWidth );
	CCpuExecutionScope scope;

	const float* first = GetRaw( firstHandle );
	const float* second = GetRaw( secondHandle );
	float* result = GetRaw( resultHandle );

	const int curThreadCount = IsOmpRelevant( firstSize, firstSize * secondWidth ) ? threadCount : 1;
	NEOML_OMP_FOR_NUM_THREADS( curThreadCount )
	for( int i = 0; i < firstSize; i++ ) {
		const float multiplier = *( first + i );
		vectorMultiply( second + i * secondWidth, result + i * secondWidth, multiplier, secondWidth );
	}
}

void CCpuMathEngine::Multiply1DiagMatrixByMatrix( int batchSize, const CConstFloatHandle& firstHandle, int firstSize,
	const CConstFloatHandle& secondHandle, int secondWidth,
	const CFloatHandle& resultHandle, int resultBufferSize )
{
	ASSERT_EXPR( resultBufferSize >= batchSize * firstSize * secondWidth );
	CCpuExecutionScope scope;

	CConstFloatHandle second = secondHandle;
	CFloatHandle result = resultHandle;

	for( int b = 0; b < batchSize; ++b ) {
		CConstFloatHandle first = firstHandle;
		for( int j = 0; j < firstSize; ++j ) {
			VectorMultiply( second, result, secondWidth, first );
			second += secondWidth;
			result += secondWidth;
			++first;
		}
	}
}

void CCpuMathEngine::MultiplyMatrixByMatrix( int batchSize, const CConstFloatHandle& firstHandle, int firstHeight,
	int firstWidth, const CConstFloatHandle& secondHandle, int secondWidth,
	const CFloatHandle& resultHandle, int resultBufferSize )
{
	ASSERT_EXPR( resultBufferSize >= batchSize * firstHeight * secondWidth );
	CCpuExecutionScope scope;

	const float* first = GetRaw( firstHandle );
	const float* second = GetRaw( secondHandle );
	float* result = GetRaw( resultHandle );

	for( int b = 0; b < batchSize; ++b ) {
		multiplyMatrixByMatrix( first, firstHeight, firstWidth, firstWidth, second, 
			secondWidth, secondWidth, result, secondWidth );
		first += firstHeight * firstWidth;
		second += firstWidth * secondWidth;
		result += firstHeight * secondWidth;
	}
}

void CCpuMathEngine::MultiplyTransposedMatrixByMatrixAndAdd(const CConstFloatHandle& firstHandle,
	int firstHeight, int firstWidth, int firstRowSize,
	const CConstFloatHandle& secondHandle, int secondWidth, int secondRowSize,
	const CFloatHandle& resultHandle, int resultRowSize, int resultBufferSize)
{
	ASSERT_EXPR((firstWidth - 1) * resultRowSize + secondWidth <= resultBufferSize);
	CCpuExecutionScope scope;

	multiplyTransposedMatrixByMatrixAndAdd( GetRaw( firstHandle ),
		firstHeight, firstWidth, firstRowSize, GetRaw( secondHandle ), secondWidth, secondRowSize,
		GetRaw( resultHandle ), resultRowSize );
}

void CCpuMathEngine::MultiplyTransposedMatrixByMatrix(int batchSize, const CConstFloatHandle& firstHandle, int firstHeight,
	int firstWidth, const CConstFloatHandle& secondHandle, int secondWidth, const CFloatHandle& resultHandle, int resultBufferSize)
{
	ASSERT_EXPR( resultBufferSize >= batchSize * firstWidth * secondWidth );
	CCpuExecutionScope scope;
	
	batchMultiplyTransposedMatrixByMatrix( batchSize, GetRaw( firstHandle ), firstHeight, firstWidth,
		GetRaw( secondHandle ), secondWidth, GetRaw( resultHandle ) );
}

void CCpuMathEngine::batchMultiplyMatrixByTransposedMatrix( int batchSize, const CConstFloatHandle& firstHandle, int firstHeight,
	int firstWidth, const CConstFloatHandle& secondHandle, int secondHeight,
	const CFloatHandle& resultHandle )
{
	CConstFloatHandle first = firstHandle;
	CConstFloatHandle second = secondHandle;
	CFloatHandle result = resultHandle;

	for( int b = 0; b < batchSize; ++b ) {
		MultiplyMatrixByTransposedMatrix( first, firstHeight, firstWidth, firstWidth, second, secondHeight, firstWidth, result,
			secondHeight, firstHeight * secondHeight );
		first += firstHeight * firstWidth;
		second += firstWidth * secondHeight;
		result += firstHeight * secondHeight;
	}
}

void CCpuMathEngine::MultiplyMatrixByTransposedMatrix(const CConstFloatHandle& firstHandle, int firstHeight,
	int firstWidth, int firstRowSize, const CConstFloatHandle& secondHandle, int secondHeight, int secondRowSize,
	const CFloatHandle& resultHandle, int resultRowSize, int)
{
	CCpuExecutionScope scope;

	const float* first = GetRaw( firstHandle );
	const float* second = GetRaw( secondHandle );
	float* result = GetRaw( resultHandle );

	const int curThreadCount = IsOmpRelevant( firstHeight * secondHeight, firstWidth * firstHeight * secondHeight )
		? threadCount : 1;
	NEOML_OMP_NUM_THREADS( curThreadCount )
	{
		int firstHeightStart;
		int firstHeightCount;
		int secondHeightStart;
		int secondHeightCount;
		if( OmpGetTaskIndexAndCount2D( firstHeight, 1, secondHeight, floatAlignment,
			firstHeightStart, firstHeightCount, secondHeightStart, secondHeightCount ) )
		{
			const float* firstData = first + firstHeightStart * firstWidth;
			float* resultData = result + firstHeightStart * secondHeight + secondHeightStart;
			const float* secondData = second + secondHeightStart * firstWidth;

			multiplyMatrixByTransposedMatrix( firstData, firstHeightCount, firstWidth, firstRowSize,
				secondData, secondHeightCount, secondRowSize,
				resultData, resultRowSize );
		}
	}
}

void CCpuMathEngine::MultiplyMatrixByTransposedMatrix( int batchSize, const CConstFloatHandle& firstHandle,
	int firstHeight, int firstWidth, const CConstFloatHandle& secondHandle, int secondHeight,
	const CFloatHandle& resultHandle, int resultBufferSize )
{
	ASSERT_EXPR( resultBufferSize >= batchSize * firstHeight * secondHeight );
	CCpuExecutionScope scope;

	CConstFloatHandle first = firstHandle;
	CConstFloatHandle second = secondHandle;
	CFloatHandle result = resultHandle;

	for( int b = 0; b < batchSize; ++b ) {
		MultiplyMatrixByTransposedMatrix( first, firstHeight, firstWidth, firstWidth, second, secondHeight,
			firstWidth, result, secondHeight, firstHeight * secondHeight );
		first += firstHeight * firstWidth;
		second += firstWidth * secondHeight;
		result += firstHeight * secondHeight;
	}
}

void CCpuMathEngine::batchMultiplyTransposedMatrixByMatrix( int batchSize,
	const float* first, int firstHeight, int firstWidth,
	const float* second, int secondWidth,
	float* result )
{
	for( int b = 0; b < batchSize; ++b ) {
		multiplyTransposedMatrixByMatrix( first, firstHeight, firstWidth, second, secondWidth,
			result );

		first += firstHeight * firstWidth;
		second += firstHeight * secondWidth;
		result += firstWidth * secondWidth;
	}
}

void CCpuMathEngine::MultiplyMatrixByDiagMatrix( const CConstFloatHandle& firstHandle, int firstHeight, int firstWidth,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int resultBufferSize )
{
	ASSERT_EXPR( resultBufferSize >= firstHeight * firstWidth );
	CCpuExecutionScope scope;

	CConstFloatHandle first = firstHandle;
	CFloatHandle result = resultHandle;

	for( int j = 0; j < firstHeight; ++j ) {
		VectorEltwiseMultiply( first, secondHandle, result, firstWidth );
		first += firstWidth;
		result += firstWidth;
	}
}

void CCpuMathEngine::MatrixSpreadRows( const CConstFloatHandle& sourceHandle, int height, int width,
	const CFloatHandle& resultHandle, int resultHeight, const CConstIntHandle& indexHandle,
	const CConstFloatHandle& fillValue )
{
	CCpuExecutionScope scope;

	float val = fillValue.IsNull() ? 0 : *GetRaw( fillValue );
	const int* indices = GetRaw( indexHandle );

	VectorFill( resultHandle, val, resultHeight * width );

	const float* source = GetRaw(sourceHandle);
	float* result = GetRaw(resultHandle);

	const int curThreadCount = IsOmpRelevant( height, height * width ) ? threadCount : 1;
	NEOML_OMP_FOR_NUM_THREADS( curThreadCount )
	for( int j = 0; j < height; ++j ) {
		if( indices[j] >= 0 ) {
			dataCopy( result + indices[j] * width, source + j * width, width );
		}
	}
}

void CCpuMathEngine::MatrixSpreadRows( const CConstIntHandle& sourceHandle, int height, int width,
	const CIntHandle& resultHandle, int resultHeight, const CConstIntHandle& indexHandle,
	const CConstIntHandle& fillValue )
{
	CCpuExecutionScope scope;

	int val = fillValue.IsNull() ? 0 : *GetRaw( fillValue );
	const int* indices = GetRaw( indexHandle );

	VectorFill( resultHandle, val, resultHeight * width );

	const int* source = GetRaw( sourceHandle );
	int* result = GetRaw( resultHandle );

	const int curThreadCount = IsOmpRelevant( height, height * width ) ? threadCount : 1;
	NEOML_OMP_FOR_NUM_THREADS( curThreadCount )
	for( int j = 0; j < height; ++j ) {
		if( indices[j] >= 0 ) {
			dataCopy( result + indices[j] * width, source + j * width, width );
		}
	}
}

void CCpuMathEngine::MatrixSpreadRowsAdd( const CConstFloatHandle& sourceHandle, int height, int width,
	const CFloatHandle& resultHandle, int /*resultHeight*/, const CConstIntHandle& indexHandle )
{
	CCpuExecutionScope scope;

	CConstFloatHandle source = sourceHandle;
	const int* indices = GetRaw( indexHandle );

	for( int j = 0; j < height; ++j ) {
		if( *indices >= 0 ) {
			CFloatHandle row = resultHandle + *indices * width;
			VectorAdd( row, source, row, width );
		}
		source += width;
		++indices;
	}
}

void CCpuMathEngine::MultiplyTransposedLookupMatrixByVector( int batchSize, const CLookupMatrix& matrix,
	const CConstFloatHandle& vectorHandle, const CFloatHandle& resultHandle, int resultSize )
{
	ASSERT_EXPR( matrix.RowCount > 0 );
	ASSERT_EXPR( resultSize >= batchSize * matrix.Width() );
	CCpuExecutionScope scope;

	CConstFloatHandle vector = vectorHandle;
	CFloatHandle result = resultHandle;
	const int* rows = GetRaw( matrix.Rows );

	for( int b = 0; b < batchSize; ++b ) {
		VectorMultiply( matrix.Table + ( *rows++ ) * matrix.Width(), result, matrix.Width(), vector++ );
		for( int j = 1; j < matrix.RowCount; ++j ) {
			VectorMultiplyAndAdd( result, matrix.Table + ( *rows++ ) * matrix.Width(), result, matrix.Width(), vector++ );
		}

		result += matrix.Width();
	}
}

void CCpuMathEngine::MultiplyTransposedLookupMatrixByVectorAndAdd( int batchSize, const CLookupMatrix& matrix,
	const CConstFloatHandle& vectorHandle, const CFloatHandle& resultHandle, int resultSize )
{
	ASSERT_EXPR( resultSize >= batchSize * matrix.Width() );
	CCpuExecutionScope scope;

	CConstFloatHandle vector = vectorHandle;
	CFloatHandle result = resultHandle;
	const int* rows = GetRaw( matrix.Rows );

	for( int b = 0; b < batchSize; ++b ) {
		for( int j = 0; j < matrix.RowCount; ++j ) {
			VectorMultiplyAndAdd( result, matrix.Table + ( *rows++ ) * matrix.Width(), result, matrix.Width(), vector++ );
		}

		result += matrix.Width();
	}
}

void CCpuMathEngine::MultiplyVectorByTransposedLookupVectorAndAddToTable( int batchSize,
	const CFloatHandle& table, int /*vectorCount*/, int vectorSize, const CConstIntHandle& indexHandle,
	const CConstFloatHandle& firstHandle, int firstSize, const CLookupVector& second )
{
	ASSERT_EXPR( vectorSize == second.VectorSize() );
	CCpuExecutionScope scope;

	CConstFloatHandle first = firstHandle;
	const int* index = GetRaw( indexHandle );
	const int* vectorIndex = GetRaw( second.Vector );

	for( int b = 0; b < batchSize; ++b ) {
		CConstFloatHandle secondVec = second.Table + ( *vectorIndex++ ) * vectorSize;
		for( int j = 0; j < firstSize; ++j ) {
			CFloatHandle tableRow = table + ( *index++ ) * vectorSize;
			VectorMultiplyAndAdd( tableRow, secondVec, tableRow, vectorSize, first++ );
		}
	}
}

void CCpuMathEngine::MatrixLogSumExpByRows( const CConstFloatHandle& matrixHandle,
	int height, int width, const CFloatHandle& resultHandle, int resultSize )
{
	ASSERT_EXPR( resultSize >= height );
	CCpuExecutionScope scope;

	CFloatHandleStackVar temp( mathEngine(), height * width );
	CFloatHandleStackVar tempVec( mathEngine(), height );

	// Find maximum in each row
	FindMaxValueInRows( matrixHandle, height, width, resultHandle, height );

	// Subtract the maximum and save the result to a temporary variable
	SubVectorFromMatrixColumns( matrixHandle, temp, height, width, resultHandle );

	// exp
	VectorExp( temp, temp, height * width );

	// Add up the columns, putting the result into tempVec
	SumMatrixColumns( tempVec, temp, height, width );

	// log
	VectorLog( tempVec, tempVec, height );

	// Add the logarithm to the maximum
	VectorAdd( resultHandle, tempVec, resultHandle, height );
}

void CCpuMathEngine::MatrixSoftmaxByRows( const CConstFloatHandle& matrixHandle, int height, int width,
	const CFloatHandle& resultHandle )
{
	CCpuExecutionScope scope;

	CFloatHandleStackVar temp( mathEngine(), height );

	// Find maximum in each row
	FindMaxValueInRows( matrixHandle, height, width, temp, height );

	// Subtract the maximum and save the result to a temporary variable
	SubVectorFromMatrixColumns( matrixHandle, resultHandle, height, width, temp );

	// exp
	VectorExp( resultHandle, resultHandle, height * width );

	// Add up the columns, putting the result into tempVec (exp(x0) + exp(x1) + ...)
	SumMatrixColumns( temp, resultHandle, height, width );

	// Calculate the denominator 1. / (exp(x0) + exp(x1) + ...)
	VectorInv( temp, temp, height );

	// Multiply the result matrix rows by 1. / (exp(x0) + exp(x1) + ...)
	MultiplyDiagMatrixByMatrix( temp, height, resultHandle, width, resultHandle, height * width );
}

void CCpuMathEngine::MatrixSoftmaxDiffOpByRows( const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, int height, int width, const CFloatHandle& resultHandle )
{
	CCpuExecutionScope scope;

	// The formula: first - y, second - dE/dy, result - dE/dx
	// dE/dxi = yi * (dE/dyi - <dE/dy, y>)

	CFloatHandleStackVar temp( mathEngine(), height );

	// <dE/dy, y>
	RowMultiplyMatrixByMatrix( firstHandle, secondHandle, height, width, temp );

	// dE/dyi - <dE/dy, y>
	SubVectorFromMatrixColumns( secondHandle, resultHandle, height, width, temp );

	// dE/dxi = yi * (dE/dyi - <dE/dy, y>)
	VectorEltwiseMultiply( resultHandle, firstHandle, resultHandle, height * width );
}

void CCpuMathEngine::MatrixSoftmaxByColumns( const CConstFloatHandle& matrix, int height, int width,
	const CFloatHandle& result )
{
	CCpuExecutionScope scope;

	CFloatHandleStackVar temp( mathEngine(), width );

	// Find maximum in each column
	findMaxValueInColumns( GetRaw( temp.GetHandle() ), GetRaw( matrix ), height, width );

	// Subtract the maximum and save the result to a temporary variable
	subVectorFromMatrixRows( this, matrix, result, height, width, temp );

	// exp
	VectorExp( result, result, height * width );

	// Add up the rows, putting the result into temp (exp(x0) + exp(x1) + ...)
	SumMatrixRows( 1, temp, result, height, width );

	// Calculate the denominator 1. / (exp(x0) + exp(x1) + ...)
	VectorInv( temp, temp, width );

	// Multiply the result matrix rows by 1. / (exp(x0) + exp(x1) + ...)
	MultiplyMatrixByDiagMatrix( result, height, width, temp, result, height * width );
}

void CCpuMathEngine::MatrixSoftmaxDiffOpByColumns( const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, int height, int width, const CFloatHandle& resultHandle )
{
	CCpuExecutionScope scope;

	// The formula: first - y, second - dE/dy, result - dE/dx
	// dE/dxi = yi * (dE/dyi - <dE/dy, y>)

	CFloatHandleStackVar temp( mathEngine(), width );

	// <dE/dy, y>
	ColumnMultiplyMatrixByMatrix( this, firstHandle, secondHandle, height, width, temp );

	// dE/dyi - <dE/dy, y>
	subVectorFromMatrixRows( this, secondHandle, resultHandle, height, width, temp );

	// dE/dxi = yi * (dE/dyi - <dE/dy, y>)
	VectorEltwiseMultiply( resultHandle, firstHandle, resultHandle, height * width );
}

void CCpuMathEngine::BitSetBinarization( int batchSize, int bitSetSize,
	const CConstIntHandle& inputHandle, int outputVectorSize, const CFloatHandle& resultHandle )
{
	CCpuExecutionScope scope;

	const int BitsPerElement = sizeof( int ) * CHAR_BIT;
	ASSERT_EXPR( static_cast<int>( bitSetSize * BitsPerElement ) >= outputVectorSize );

	const int* input = GetRaw( inputHandle );
	float* result = GetRaw( resultHandle );

	VectorFill( resultHandle, 0, batchSize * outputVectorSize );

	for( int batchIndex = 0; batchIndex < batchSize; ++batchIndex ) {
		const int* batchBegin = input + batchIndex * bitSetSize;
		for( int elementIndex = 0; elementIndex < outputVectorSize; elementIndex += BitsPerElement ) {
			unsigned int element = batchBegin[elementIndex / BitsPerElement];
			int offset = 0;
			while( element != 0 ) {
				unsigned long enabledBit;
#if FINE_PLATFORM(FINE_WINDOWS)
				_BitScanForward( &enabledBit, element );
#elif FINE_PLATFORM(FINE_LINUX) || FINE_PLATFORM(FINE_DARWIN) || FINE_PLATFORM(FINE_ANDROID) || FINE_PLATFORM(FINE_IOS)
				enabledBit = __builtin_ffsll( element ) - 1;
#else 
	#error "Platform isn't supported!"
#endif
				PRESUME_EXPR( ( enabledBit + offset + elementIndex ) < ( unsigned int ) outputVectorSize );
				result[enabledBit + offset] = 1.0f;
				element = ( element >> enabledBit ) >> 1;
				offset += ( enabledBit + 1 );
			}
			result += min( BitsPerElement, outputVectorSize - elementIndex );
		}
	}
}

} // namespace NeoML
