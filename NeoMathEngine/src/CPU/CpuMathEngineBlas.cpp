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
#include <MemoryHandleInternal.h>
#include <MathEngineCommon.h>
#include <math.h>

namespace NeoML {

// LogSumExp for two inputs
inline float LogSumExpFunc(float f, float s)
{
	if(f >= s) {
		return f + log1pf(expf(s - f));
	} else {
		return s + log1pf(expf(f - s));
	}
}

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

void CCpuMathEngine::SetVectorToMatrixRows( const CFloatHandle& resultHandle,
	int matrixHeight, int matrixWidth, const CConstFloatHandle& vectorHandle )
{
	CFloatHandle result = resultHandle;

	const int curThreadCount = IsOmpRelevant( matrixHeight, matrixHeight * matrixWidth ) ? threadCount : 1;
	NEOML_OMP_FOR_NUM_THREADS( curThreadCount )
	for(int i = 0; i < matrixHeight; i++) {
		VectorCopy( result + i * matrixWidth, vectorHandle, matrixWidth );
	}
}

void CCpuMathEngine::setVectorToMatrixRows(const CFloatHandle& resultHandle,
	int matrixHeight, int matrixWidth, const CConstFloatHandle& vectorHandle)
{
	CFloatHandle result = resultHandle;

	for(int i = 0; i < matrixHeight; i++) {
		vectorCopy( result, vectorHandle, matrixWidth );
		result += matrixWidth;
	}
}

void CCpuMathEngine::AddVectorToMatrixColumns(const CConstFloatHandle& matrixHandle, const CFloatHandle& resultHandle,
	int matrixHeight, int matrixWidth, const CConstFloatHandle& vectorHandle)
{
	CConstFloatHandle matrix = matrixHandle;
	CFloatHandle result = resultHandle;
	CConstFloatHandle vector = vectorHandle;

	for(int i = 0; i < matrixHeight; ++i) {
		VectorAddValue(matrix, result, matrixWidth, vector);
		matrix += matrixWidth;
		result += matrixWidth;
		++vector;
	}
}

void CCpuMathEngine::AddVectorToMatrixRows( int batchSize, const CConstFloatHandle& matrixHandle, const CFloatHandle& resultHandle,
	int matrixHeight, int matrixWidth, const CConstFloatHandle& vectorHandle )
{
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
			CFloatHandle outputData = resultHandle + offset;
			CConstFloatHandle inputData = matrixHandle + offset;
			CConstFloatHandle vectorData = vectorHandle + batchStart* matrixWidth + widthStart;

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
	CConstFloatHandle first = firstHandle;
	CConstFloatHandle second = secondHandle;
	CFloatHandle result = resultHandle;

	for(int i = 0; i < height; ++i) {
		VectorDotProduct(first, second, width, result);
		first += width;
		second += width;
		++result;
	}
}

static void ColumnMultiplyMatrixByMatrix(CCpuMathEngine* engine, const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, int height, int width, const CFloatHandle& resultHandle)
{
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
	CConstFloatHandle matrix = matrixHandle;
	CFloatHandle result = resultHandle;
	const float* vector = GetRaw(vectorHandle);

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
	CConstFloatHandle matrix = matrixHandle;
	CFloatHandle result = resultHandle;

	for(int j = 0; j < matrixHeight; ++j) {
		VectorSum(matrix, matrixWidth, result);
		matrix += matrixWidth;
		++result;
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
	VectorFill(resultHandle, 0.f, batchSize * matrixWidth);
	SumMatrixRowsAdd(batchSize, resultHandle, matrixHandle, matrixHeight, matrixWidth);
}

void CCpuMathEngine::SumMatrixRowsAdd(int batchSize,
	const CFloatHandle& resultHandle, const CConstFloatHandle& matrixHandle, int matrixHeight, int matrixWidth)
{
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

void CCpuMathEngine::findMaxValueInColumns(const CFloatHandle& resultHandle, const CConstFloatHandle& matrixHandle,
	int matrixHeight, int matrixWidth)
{
	if(matrixHeight == 1) {
		VectorCopy(resultHandle, matrixHandle, matrixWidth);
		return;
	}

	CConstFloatHandle nextRow = matrixHandle + matrixWidth;
	VectorEltwiseMax(matrixHandle, nextRow, resultHandle, matrixWidth);

	for(int i = 2; i < matrixHeight; ++i) {
		nextRow += matrixWidth;
		VectorEltwiseMax(resultHandle, nextRow, resultHandle, matrixWidth);
	}
}

// Sets the matrix elements to the values from a vector: matrix[rowIndices[i], columnIndices[i]] = vector[i].
void CCpuMathEngine::SetVectorToMatrixElements(
	const CFloatHandle& matrixHandle, int /*height*/, int width,
	const CConstIntHandle& rowIndicesHandle, const CConstIntHandle& columnIndicesHandle,
	const CConstFloatHandle& vectorHandle, int vectorSize )
{
	float* matrix = GetRaw( matrixHandle );
	const int* rowIndices = GetRaw( rowIndicesHandle );
	const int* columnIndices = GetRaw( columnIndicesHandle );
	const float* vector = GetRaw( vectorHandle );

	for( int i = 0; i < vectorSize; i++ ) {
		matrix[rowIndices[i] * width + columnIndices[i]] = vector[i];
	}
}

void CCpuMathEngine::VectorMultichannelLookupAndCopy( int batchSize, int channelCount, const CConstFloatHandle& inputHandle,
	const CConstFloatHandle* lookupHandles, const CLookupDimension* lookupDimensions, int lookupCount,
	const CFloatHandle& outputHandle, int /*outputChannels*/ )
{
	ASSERT_EXPR(lookupCount <= channelCount);

	CConstFloatHandle input = inputHandle;
	CFloatHandle output = outputHandle;

	for(int i = 0; i < batchSize; ++i) {
		for(int j = 0; j < lookupCount; ++j) {
			int index = (int)input.GetValue();
			input++;
			ASSERT_EXPR(0 <= index && index < lookupDimensions[j].VectorCount);
			int vectorSize = lookupDimensions[j].VectorSize;
			VectorCopy(output, lookupHandles[j] + index * vectorSize, vectorSize);
			output += vectorSize;
		}
		int remained = channelCount - lookupCount;
		if(remained > 0) {
			VectorCopy(output, input, remained);
			input += remained;
			output += remained;
		}
	}
}

void CCpuMathEngine::VectorMultichannelLookupAndCopy(int batchSize, int channelCount, const CConstIntHandle& inputHandle,
	const CConstFloatHandle* lookupHandles, const CLookupDimension* lookupDimensions, int lookupCount,
	const CFloatHandle& outputHandle, int /*outputChannels*/)
{
	ASSERT_EXPR(lookupCount <= channelCount);

	CConstIntHandle input = inputHandle;
	CFloatHandle output = outputHandle;

	for(int i = 0; i < batchSize; ++i) {
		for(int j = 0; j < lookupCount; ++j) {
			if(j < channelCount) {
				int index = input.GetValue();
				input++;
				ASSERT_EXPR(0 <= index && index < lookupDimensions[j].VectorCount);
				int vectorSize = lookupDimensions[j].VectorSize;
				VectorCopy(output, lookupHandles[j] + index * vectorSize, vectorSize);
				output += vectorSize;
			}
		}
		int remained = channelCount - lookupCount;
		ASSERT_EXPR(remained == 0);
	}
}

void CCpuMathEngine::VectorMultichannelLookupAndAddToTable(int batchSize, int channelCount, const CConstFloatHandle& inputHandle,
	const CFloatHandle* lookupHandles, const CLookupDimension* lookupDimensions, int lookupCount,
	const CConstFloatHandle& multHandle, const CConstFloatHandle& matrixHandle, int /*outputChannels*/)
{
	ASSERT_EXPR(lookupCount <= channelCount);

	CConstFloatHandle input = inputHandle;
	CConstFloatHandle matrix = matrixHandle;

	for(int i = 0; i < batchSize; ++i) {
		for(int j = 0; j < lookupCount; ++j) {
			if(j < channelCount) {
				int index = (int)input.GetValue();
				input++;
				ASSERT_EXPR(0 <= index && index < lookupDimensions[j].VectorCount);
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

	CConstIntHandle input = inputHandle;
	CConstFloatHandle matrix = matrixHandle;

	for(int i = 0; i < batchSize; ++i) {
		for(int j = 0; j < lookupCount; ++j) {
			if(j < channelCount) {
				int index = input.GetValue();
				input++;
				ASSERT_EXPR(0 <= index && index < lookupDimensions[j].VectorCount);
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
	const float* input = GetRaw(inputHandle);
	float* result = GetRaw(resultHandle);

	VectorFill(resultHandle, 0, batchSize * enumSize);

	for(int i = 0; i < batchSize; ++i) {
		int enumValue = (int)(*input++);
		if(enumValue >= 0) {
			ASSERT_EXPR(enumValue < enumSize);
			result[enumValue] = 1;
		}
		result += enumSize;
	}
}

void CCpuMathEngine::EnumBinarization(int batchSize,
	const CConstIntHandle& inputHandle, int enumSize, const CFloatHandle& resultHandle)
{
	const int* input = GetRaw(inputHandle);
	float* result = GetRaw(resultHandle);

	VectorFill(resultHandle, 0, batchSize * enumSize);

	for(int i = 0; i < batchSize; ++i) {
		int enumValue = *input++;
		if(enumValue >= 0) {
			ASSERT_EXPR(enumValue < enumSize);
			result[enumValue] = 1;
		}
		result += enumSize;
	}
}

void CCpuMathEngine::AddMatrixElementsToVector(const CConstFloatHandle& matrixHandle, int height, int width,
	const CConstIntHandle& indicesHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR(vectorSize >= height);

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

void CCpuMathEngine::AddMatrixElementsToMatrix(const CConstFloatHandle& matrixHandle, int height, int width,
	const CFloatHandle& resultHandle, const CConstIntHandle& indicesHandle)
{
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
	float* matrix = GetRaw(matrixHandle);
	const int* rowIndices = GetRaw(rowIndicesHandle);
	const int* columnIndices = GetRaw(columnIndicesHandle);
	const float* vector = GetRaw(vectorHandle);

	for(int i = 0; i < vectorSize; ++i) {
		matrix[rowIndices[i] * width + columnIndices[i]] += vector[i];
	}
}

void CCpuMathEngine::EltwiseLogSumExpVectorToMatrixElements(const CFloatHandle& matrixHandle, int height, int width,
	const CConstIntHandle& indicesHandle, const CConstFloatHandle& vectorHandle)
{
	float* matrix = GetRaw(matrixHandle);
	const int* indices = GetRaw(indicesHandle);
	const float* vector = GetRaw(vectorHandle);

	for(int j = 0; j < height; ++j) {
		int index = *indices++;
		if(index < 0 || index >= width) {
			++vector;
		} else {
			matrix[index] = LogSumExpFunc(*vector++, matrix[index]);
		}
		matrix += width;
	}
}

void CCpuMathEngine::EltwiseLogSumExpVectorToMatrixElements(const CFloatHandle& matrixHandle,
	int height, int width,
	const CConstIntHandle& rowIndicesHandle, const CConstIntHandle& columnIndicesHandle,
	const CConstFloatHandle& vectorHandle, int vectorSize)
{
	float* matrix = GetRaw(matrixHandle);
	const int* rowIndices = GetRaw(rowIndicesHandle);
	const int* columnIndices = GetRaw(columnIndicesHandle);
	const float* vector = GetRaw(vectorHandle);

	for(int i = 0; i < vectorSize; i++) {
		const int rowIndex = rowIndices[i];
		const int columnIndex = columnIndices[i];
		if(rowIndex >= 0 && rowIndex < height &&
			columnIndex >= 0 && columnIndex < width) {
			const int matrixIndex = rowIndex * width + columnIndex;
			matrix[matrixIndex] = LogSumExpFunc(vector[i], matrix[matrixIndex]);
		}
	}
}

void CCpuMathEngine::LookupAndSum(const CConstIntHandle& indicesHandle, int batchSize, int indexCount,
	const CConstFloatHandle& tableHandle, int vectorSize, const CFloatHandle& result)
{
	CConstIntHandle indices = indicesHandle;
	CFloatHandle output = result;
	for(int b = 0; b < batchSize; ++b) {
		int index = (int)indices.GetValue();
		indices++;
		if(index >= 0) {
			VectorCopy(output, tableHandle + vectorSize * index, vectorSize);
		} else {
			VectorFill(output, 0.f, vectorSize);
		}
		for(int elem = 1; elem < indexCount; ++elem) {
			index = (int)indices.GetValue();
			indices++;
			if(index >= 0) {
				VectorAdd(output, tableHandle + vectorSize * index, output, vectorSize);
			}
		}
		output += vectorSize;
	}
}

void CCpuMathEngine::LookupAndAddToTable(const CConstIntHandle& indicesHandle, int batchSize, int indexCount,
	const CConstFloatHandle& additionsHandle, int vectorSize, const CFloatHandle& tableHandle, int vectorCount)
{
	VectorFill(tableHandle, 0.f, vectorCount * vectorSize);

	CConstIntHandle indices = indicesHandle;
	CConstFloatHandle additions = additionsHandle;
	for(int b = 0; b < batchSize; ++b) {
		for(int elem = 0; elem < indexCount; ++elem) {
			int index = (int)indices.GetValue();
			indices++;
			if(index >= 0) {
				VectorAdd(tableHandle + index * vectorSize, additions, tableHandle + index * vectorSize, vectorSize);
			}
		}
		additions += vectorSize;
	}
}

void CCpuMathEngine::findMaxValueInColumns( const CFloatHandle& resultHandle, const CIntHandle& rowIndicesHandle,
	const CConstFloatHandle& matrixHandle, int matrixHeight, int matrixWidth )
{
	const float* matrix = GetRaw( matrixHandle );
	float* result = GetRaw( resultHandle );
	int* rowIndices = GetRaw( rowIndicesHandle );

	// Copy the first row
	VectorCopy( resultHandle, matrixHandle, matrixWidth );
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

	CConstFloatHandle first = firstHandle;
	CConstFloatHandle second = secondHandle;
	CFloatHandle result = resultHandle;

	for( int j = 0; j < firstSize; ++j ) {
		VectorMultiply( second, result, secondWidth, first );
		second += secondWidth;
		result += secondWidth;
		++first;
	}
}

void CCpuMathEngine::Multiply1DiagMatrixByMatrix( int batchSize, const CConstFloatHandle& firstHandle, int firstSize,
	const CConstFloatHandle& secondHandle, int secondWidth,
	const CFloatHandle& resultHandle, int resultBufferSize )
{
	ASSERT_EXPR( resultBufferSize >= batchSize * firstSize * secondWidth );

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

	CConstFloatHandle first = firstHandle;
	CConstFloatHandle second = secondHandle;
	CFloatHandle result = resultHandle;

	for( int b = 0; b < batchSize; ++b ) {
		multiplyMatrixByMatrix( first, firstHeight, firstWidth, firstWidth, second, secondWidth, secondWidth, result,
			secondWidth, firstHeight * secondWidth );
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
	multiplyTransposedMatrixByMatrixAndAdd( firstHandle,
		firstHeight, firstWidth, firstRowSize, secondHandle, secondWidth, secondRowSize,
		resultHandle, resultRowSize, resultBufferSize );
}

void CCpuMathEngine::MultiplyTransposedMatrixByMatrix(int batchSize, const CConstFloatHandle& firstHandle, int firstHeight,
	int firstWidth, const CConstFloatHandle& secondHandle, int secondWidth, const CFloatHandle& resultHandle, int resultBufferSize)
{
	batchMultiplyTransposedMatrixByMatrix(batchSize, firstHandle, firstHeight, firstWidth, secondHandle, secondWidth, resultHandle, resultBufferSize);
}

void CCpuMathEngine::batchMultiplyMatrixByTransposedMatrix( int batchSize, const CConstFloatHandle& firstHandle, int firstHeight,
	int firstWidth, const CConstFloatHandle& secondHandle, int secondHeight,
	const CFloatHandle& resultHandle, int resultBufferSize )
{
	ASSERT_EXPR( resultBufferSize >= batchSize * firstHeight * secondHeight );

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
			CConstFloatHandle firstData = firstHandle + firstHeightStart * firstWidth;
			CFloatHandle resultData = resultHandle + firstHeightStart * secondHeight + secondHeightStart;
			CConstFloatHandle secondData = secondHandle + secondHeightStart * firstWidth;

			multiplyMatrixByTransposedMatrix( firstData, firstHeightCount, firstWidth, firstRowSize,
				secondData, secondHeightCount, secondRowSize,
				resultData, resultRowSize, resultRowSize * firstHeight );
		}
	}
}

void CCpuMathEngine::MultiplyMatrixByTransposedMatrix( int batchSize, const CConstFloatHandle& firstHandle,
	int firstHeight, int firstWidth, const CConstFloatHandle& secondHandle, int secondHeight,
	const CFloatHandle& resultHandle, int resultBufferSize )
{
	ASSERT_EXPR( resultBufferSize >= batchSize * firstHeight * secondHeight );

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
	const CConstFloatHandle& firstHandle, int firstHeight, int firstWidth,
	const CConstFloatHandle& secondHandle, int secondWidth,
	const CFloatHandle& resultHandle, int resultBufferSize )
{
	ASSERT_EXPR( resultBufferSize >= batchSize * firstWidth * secondWidth );

	CConstFloatHandle first = firstHandle;
	CConstFloatHandle second = secondHandle;
	CFloatHandle result = resultHandle;

	for( int b = 0; b < batchSize; ++b ) {
		multiplyTransposedMatrixByMatrix( first, firstHeight, firstWidth, second, secondWidth,
			result, firstWidth * secondWidth );

		first += firstHeight * firstWidth;
		second += firstHeight * secondWidth;
		result += firstWidth * secondWidth;
	}
}

void CCpuMathEngine::MultiplyMatrixByDiagMatrix( const CConstFloatHandle& firstHandle, int firstHeight, int firstWidth,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int resultBufferSize )
{
	ASSERT_EXPR( resultBufferSize >= firstHeight * firstWidth );

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
	float val = fillValue.IsNull() ? 0 : *GetRaw( fillValue );
	const int* indices = GetRaw( indexHandle );

	VectorFill( resultHandle, val, resultHeight * width );

	CConstFloatHandle source = sourceHandle;
	for( int j = 0; j < height; ++j ) {
		if( *indices >= 0 ) {
			VectorCopy( resultHandle + *indices * width, source, width );
		}
		source += width;
		++indices;
	}
}

void CCpuMathEngine::MatrixSpreadRows( const CConstIntHandle& sourceHandle, int height, int width,
	const CIntHandle& resultHandle, int resultHeight, const CConstIntHandle& indexHandle,
	const CConstIntHandle& fillValue )
{
	int val = fillValue.IsNull() ? 0 : *GetRaw( fillValue );
	const int* indices = GetRaw( indexHandle );

	VectorFill( resultHandle, val, resultHeight * width );

	CConstIntHandle source = sourceHandle;
	for( int j = 0; j < height; ++j ) {
		if( *indices >= 0 ) {
			VectorCopy( resultHandle + *indices * width, source, width );
		}
		source += width;
		++indices;
	}
}

void CCpuMathEngine::MatrixSpreadRowsAdd( const CConstFloatHandle& sourceHandle, int height, int width,
	const CFloatHandle& resultHandle, int /*resultHeight*/, const CConstIntHandle& indexHandle )
{
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

void CCpuMathEngine::FindMinValueInColumns( const CConstFloatHandle& matrixHandle, int matrixHeight,
	int matrixWidth, const CFloatHandle& resultHandle, const CIntHandle& columnIndices )
{
	const float* matrix = GetRaw( matrixHandle );
	float* result = GetRaw( resultHandle );
	int* rowIndices = GetRaw( columnIndices );

	// Copy the first row
	VectorCopy( resultHandle, matrixHandle, matrixWidth );
	VectorFill( columnIndices, 0, matrixWidth );
	matrix += matrixWidth;
	// Process the rest
	for( int i = 0; i < matrixHeight - 1; i++ ) {
		float* vectorPtr = result;
		int* indicesPtr = rowIndices;
		for( int j = 0; j < matrixWidth; j++ ) {
			if( *matrix < *vectorPtr ) {
				*vectorPtr = *matrix;
				*indicesPtr = i + 1;
			}
			matrix += 1;
			vectorPtr += 1;
			indicesPtr += 1;
		}
	}
}

void CCpuMathEngine::MatrixLogSumExpByRows( const CConstFloatHandle& matrixHandle,
	int height, int width, const CFloatHandle& resultHandle, int resultSize )
{
	ASSERT_EXPR( resultSize >= height );

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

void CCpuMathEngine::MatrixLogSumExpByColumns( const CConstFloatHandle& matrixHandle,
	int height, int width, const CFloatHandle& resultHandle, int resultSize )
{
	ASSERT_EXPR( resultSize >= width );

	CFloatHandleStackVar temp( mathEngine(), height * width );
	CFloatHandleStackVar tempVec( mathEngine(), width );

	// Find maximum in each column
	findMaxValueInColumns( resultHandle, matrixHandle, height, width );

	// Subtract the maximum and save the result to a temporary variable
	subVectorFromMatrixRows( this, matrixHandle, temp, height, width, resultHandle );

	// exp
	VectorExp( temp, temp, height * width );

	// Add up the rows, putting the result into tempVec
	SumMatrixRows( 1, tempVec, temp, height, width );

	// log
	VectorLog( tempVec, tempVec, width );

	// Add the logarithm to the maximum
	VectorAdd( resultHandle, tempVec, resultHandle, width );
}

void CCpuMathEngine::MatrixSoftmaxByColumns( const CConstFloatHandle& matrix, int height, int width,
	const CFloatHandle& result )
{
	CFloatHandleStackVar temp( mathEngine(), width );

	// Find maximum in each column
	findMaxValueInColumns( temp, matrix, height, width );

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
				ASSERT_EXPR( ( enabledBit + offset + elementIndex ) < ( unsigned int ) outputVectorSize );
				result[enabledBit + offset] = 1.0f;
				element = ( element >> enabledBit ) >> 1;
				offset += ( enabledBit + 1 );
			}
			result += min( BitsPerElement, outputVectorSize - elementIndex );
		}
	}
}

} // namespace NeoML
