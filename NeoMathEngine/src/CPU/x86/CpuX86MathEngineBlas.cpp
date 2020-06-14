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

#include <NeoMathEngine/NeoMathEngineDefs.h>

#ifdef NEOML_USE_SSE

#include <CpuMathEngine.h>
#include <CpuX86.h>
#include <float.h>
#include <MemoryHandleInternal.h>
#include <MathEngineCommon.h>
#include <CpuX86MathEngineVectorMathPrivate.h>

namespace NeoML {

// Split the vector length into registers of 4 elements each + the remainder
static inline int GetCount4(int& size)
{
	int ret = size / 4;
	size %= 4;
	return ret;
}

static inline void batchTransposePlainMatrix( int batchSize, const float* first,
	int height, int width, float* result, int )
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
void CCpuMathEngine::transposeMatrixImpl( int batchSize, const CTypedMemoryHandle<const T>& firstHandle,
	int height, int medium, int width, int channels, const CTypedMemoryHandle<T>& resultHandle, int resultBufferSize )
{
	if( medium == 1 && channels == 1 ) {
		static_assert( sizeof(float) == sizeof(T), "Size of float isn't equal to size of T." );
		batchTransposePlainMatrix( batchSize, ( const float* ) GetRaw( firstHandle ),
			height, width, ( float* ) GetRaw( resultHandle ), resultBufferSize );
		return;
	}

	int objectSize = height * width * medium * channels;
	ASSERT_EXPR( resultBufferSize >= batchSize * objectSize );

	CTypedMemoryHandle<const T> source = firstHandle;
	CTypedMemoryHandle<T> result = resultHandle;

	int resultRowSize = height * medium * channels;

	for( int b = 0; b < batchSize; ++b ) {
		CTypedMemoryHandle<T> resultColumnStart = result;
		for( int j = 0; j < height; ++j ) {
			CTypedMemoryHandle<T> resultMediumStart = resultColumnStart;
			for( int m = 0; m < medium; ++m ) {
				CTypedMemoryHandle<T> resultItem = resultMediumStart;
				for( int i = 0; i < width; ++i ) {
					VectorCopy( resultItem, source, channels );
					resultItem += resultRowSize;
					source += channels;
				}
				resultMediumStart += channels * height;
			}
			resultColumnStart += channels;
		}
		result += objectSize;
	}
}

template void CCpuMathEngine::transposeMatrixImpl<float>( int batchSize, const CTypedMemoryHandle<const float>& firstHandle,
	int height, int medium, int width, int channels, const CTypedMemoryHandle<float>& resultHandle, int resultBufferSize );

template void CCpuMathEngine::transposeMatrixImpl<int>( int batchSize, const CTypedMemoryHandle<const int>& firstHandle,
	int height, int medium, int width, int channels, const CTypedMemoryHandle<int>& resultHandle, int resultBufferSize );

void CCpuMathEngine::TransposeMatrix( int batchSize, const CConstFloatHandle& firstHandle,
	int height, int medium, int width, int channels, const CFloatHandle& resultHandle, int resultBufferSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

	transposeMatrixImpl( batchSize, firstHandle, height, medium, width, channels, resultHandle, resultBufferSize );
}

void CCpuMathEngine::TransposeMatrix( int batchSize, const CConstIntHandle& firstHandle,
	int height, int medium, int width, int channels, const CIntHandle& resultHandle, int resultBufferSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

	transposeMatrixImpl( batchSize, firstHandle, height, medium, width, channels, resultHandle, resultBufferSize );
}

void CCpuMathEngine::addVectorToMatrixRows(const CConstFloatHandle& matrixHandle, const CFloatHandle& resultHandle,
	int matrixHeight, int matrixWidth, int matrixRowSize, int resultRowSize, const CConstFloatHandle& vectorHandle)
{
	const float* matrix = GetRaw( matrixHandle );
	float* result = GetRaw( resultHandle );
	const float* vector = GetRaw( vectorHandle );

	for(int i = 0; i < matrixHeight; i++) {
		vectorAdd( matrix, vector, result, matrixWidth );
		matrix += matrixRowSize;
		result += resultRowSize;
	}
}

void CCpuMathEngine::MatrixRowsToVectorSquaredL2Distance( const CConstFloatHandle& matrixHandle,
	const int matrixHeight, const int matrixWidth, const CConstFloatHandle& vectorHandle,
	const CFloatHandle& resultHandle )
{
	ASSERT_EXPR( matrixHandle.GetMathEngine() == this );
	ASSERT_EXPR( vectorHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

	const float* matrix = GetRaw( matrixHandle );
	float* result = GetRaw( resultHandle );

	for( int i = 0; i < matrixHeight; i++ ) {
		const float* rawVector = GetRaw( vectorHandle );
		*result++ = euclidianSSE( matrix, rawVector, matrixWidth );
		matrix += matrixWidth;
	}
}

static void findMaxValueWorker(const __m128& value, const __m128i& index, __m128& maxValue, __m128i& maxIndex)
{
	__m128i cmp = _mm_castps_si128(_mm_cmpgt_ps(value, maxValue));
	maxIndex = _mm_or_si128(_mm_andnot_si128(cmp, maxIndex), _mm_and_si128(cmp, index));
	maxValue = _mm_max_ps(value, maxValue);
}

void CCpuMathEngine::FindMaxValueInRows(const CConstFloatHandle& matrixHandle, int matrixHeight, int matrixWidth,
	const CFloatHandle& resultHandle, const CIntHandle& columnIndices, int vectorSize)
{
	ASSERT_EXPR( matrixHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( columnIndices.GetMathEngine() == this );
	ASSERT_EXPR( vectorSize >= matrixHeight );

	const float* matrix = GetRaw(matrixHandle);
	float* result = GetRaw(resultHandle);
	int* indices = GetRaw(columnIndices);

	int sseSize;
	int nonSseSize;
	checkSse(matrixWidth, sseSize, nonSseSize);

	__m128i iStep = _mm_set1_epi32(4);
	__m128 maxValueAcc = _mm_setzero_ps();
	__m128i maxIndexAcc = _mm_setzero_si128();
	for(int j = 0; j < matrixHeight; ++j) {
		// Find the maximum in the row
		__m128i index = _mm_set_epi32(3, 2, 1, 0);
		__m128 maxValue = _mm_set1_ps(-FLT_MAX);
		__m128i maxIndex = index;
		for(int i = 0; i < sseSize; ++i) {
			__m128 value = LoadSse4(matrix);
			findMaxValueWorker(value, index, maxValue, maxIndex);

			index = _mm_add_epi32(index, iStep);
			matrix += 4;
		}

		if(nonSseSize > 0) {
			__m128 value = LoadSse(matrix, nonSseSize, -FLT_MAX);
			findMaxValueWorker(value, index, maxValue, maxIndex);

			matrix += nonSseSize;
		}
		
		// Find the maximum inside maxValue
		__m128 value = _mm_shuffle_ps(maxValue, maxValue, _MM_SHUFFLE(1, 0, 3, 2));
		index = _mm_shuffle_epi32(maxIndex, _MM_SHUFFLE(1, 0, 3, 2));
		findMaxValueWorker(value, index, maxValue, maxIndex);

		value = _mm_shuffle_ps(maxValue, maxValue, _MM_SHUFFLE(2, 3, 0, 1));
		index = _mm_shuffle_epi32(maxIndex, _MM_SHUFFLE(2, 3, 0, 1));
		findMaxValueWorker(value, index, maxValue, maxIndex);

		// Maximum is stored in maxValue fields, put it into maxValueAcc
		int phase = j % 4;
		__m128 mask = GetPhaseMask4(phase);
		maxValueAcc = _mm_or_ps(_mm_andnot_ps(mask, maxValueAcc), _mm_and_ps(mask, maxValue));
		maxIndexAcc = _mm_or_si128(_mm_andnot_si128(_mm_castps_si128(mask), maxIndexAcc),
			_mm_and_si128(_mm_castps_si128(mask), maxIndex));

		// Save the result if necessary
		if(phase == 3) {
			StoreSse4(maxValueAcc, result);
			StoreIntSse4(maxIndexAcc, indices);
			result += 4;
			indices += 4;
		} else if(j == matrixHeight - 1) {
			StoreSse(maxValueAcc, result, phase + 1);
			StoreIntSse(maxIndexAcc, indices, phase + 1);
		}
	}
}

void CCpuMathEngine::FindMaxValueInRows(const CConstFloatHandle& matrixHandle, int matrixHeight, int matrixWidth,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( matrixHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( vectorSize >= matrixHeight );

	const float* matrix = GetRaw(matrixHandle);
	float* result = GetRaw(resultHandle);

	int sseSize;
	int nonSseSize;
	checkSse(matrixWidth, sseSize, nonSseSize);

	__m128 maxValueAcc = _mm_setzero_ps();
	for(int j = 0; j < matrixHeight; ++j) {
		// Find the maximum in the row
		__m128 maxValue = _mm_set1_ps(-FLT_MAX);
		for(int i = 0; i < sseSize; ++i) {
			__m128 value = LoadSse4(matrix);
			maxValue = _mm_max_ps(value, maxValue);

			matrix += 4;
		}

		if(nonSseSize > 0) {
			__m128 value = LoadSse(matrix, nonSseSize, -FLT_MAX);
			maxValue = _mm_max_ps(value, maxValue);

			matrix += nonSseSize;
		}

		// Find the maximum inside maxValue
		__m128 value = _mm_shuffle_ps(maxValue, maxValue, _MM_SHUFFLE(1, 0, 3, 2));
		maxValue = _mm_max_ps(value, maxValue);

		value = _mm_shuffle_ps(maxValue, maxValue, _MM_SHUFFLE(2, 3, 0, 1));
		maxValue = _mm_max_ps(value, maxValue);

		// Maximum is stored in maxValue fields, put it into maxValueAcc
		int phase = j % 4;
		__m128 mask = GetPhaseMask4(phase);
		maxValueAcc = _mm_or_ps(_mm_andnot_ps(mask, maxValueAcc), _mm_and_ps(mask, maxValue));

		// Save the result if necessary
		if(phase == 3) {
			StoreSse4(maxValueAcc, result);
			result += 4;
		} else if(j == matrixHeight - 1) {
			StoreSse(maxValueAcc, result, phase + 1);
		}
	}
}

void CCpuMathEngine::FindMaxValueInColumns( int batchSize, const CConstFloatHandle& matrixHandle,
	int matrixHeight, int matrixWidth, const CFloatHandle& resultHandle, const CIntHandle& rowIndices, int vectorSize )
{
	ASSERT_EXPR( matrixHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( rowIndices.GetMathEngine() == this );

	if( matrixWidth == 1 ) {
		// For x86 FindMaxValueInRows would be more optimal,
		// because it uses SSE differently
		FindMaxValueInRows( matrixHandle, batchSize, matrixHeight, resultHandle, rowIndices, vectorSize );
		return;
	}

	ASSERT_EXPR( vectorSize >= batchSize * matrixWidth );

	int sseSize;
	int nonSseSize;
	checkSse( matrixWidth, sseSize, nonSseSize );

	// The pointer moves over the first row of the matrix
	const float* firstRow = GetRaw( matrixHandle );
	float* result = GetRaw( resultHandle );
	int* indices = GetRaw( rowIndices );

	for( int b = 0; b < batchSize; ++b ) {
		// Process 4 columns at once
		for( int i = 0; i < sseSize; ++i ) {
			const float* data = firstRow;
			__m128 maxValue = LoadSse4( data );
			__m128i maxIndices = _mm_set1_epi32( 0 );
			data += matrixWidth;
			for( int h = 1; h < matrixHeight; ++h ) {
				__m128 value = LoadSse4( data );
				__m128i currIndices = _mm_set1_epi32( h );
				findMaxValueWorker( value, currIndices, maxValue, maxIndices );
				data += matrixWidth;
			}
			StoreSse4( maxValue, result );
			StoreIntSse4( maxIndices, indices );
			result += 4;
			indices += 4;
			firstRow += 4;
		}

		if( nonSseSize > 0 ) {
			// Process the rest of the columns in the same way
			const float* data = firstRow;
			__m128 maxValue = LoadSse( data, nonSseSize );
			__m128i maxIndices = _mm_set1_epi32( 0 );
			data += matrixWidth;
			for( int h = 1; h < matrixHeight; ++h ) {
				__m128 value = LoadSse( data, nonSseSize );
				__m128i currIndices = _mm_set1_epi32( h );
				findMaxValueWorker( value, currIndices, maxValue, maxIndices );
				data += matrixWidth;
			}
			StoreSse( maxValue, result, nonSseSize );
			StoreIntSse( maxIndices, indices, nonSseSize );
			result += nonSseSize;
			indices += nonSseSize;
			firstRow += nonSseSize;
		}

		// firstRow points to the start of the second row of the current matrix
		// Move it to the start of the first row of the next matrix in the batch
		firstRow += matrixWidth * ( matrixHeight - 1 );
	}
}

void CCpuMathEngine::MultiplyDiagMatrixByMatrixAndAdd(int batchSize, const CConstFloatHandle& firstHandle,
	int firstSize, const CConstFloatHandle& secondHandle, int secondWidth, const CFloatHandle& resultHandle)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

	const float* first = GetRaw(firstHandle);
	const float* second = GetRaw(secondHandle);
	float* resultStart = GetRaw(resultHandle);

	int sseSize;
	int nonSseSize;
	checkSse(secondWidth, sseSize, nonSseSize);

	for(int b = 0; b < batchSize; ++b) {
		float* result = resultStart;
		__m128 firstValBuf = _mm_setzero_ps();
		for(int j = 0; j < firstSize; ++j) {
			int phase = j % 4;
			if(phase == 0) {
				int loadSize = firstSize - j;
				if(loadSize >= 4) {
					loadSize = 4;
					firstValBuf = LoadSse4(first);
				} else {
					firstValBuf = LoadSse(first, loadSize);
				}
				first += loadSize;
			}
			__m128 firstVal = GetPhaseValue4(firstValBuf, phase);

			for(int i = 0; i < sseSize; ++i) {
				__m128 secondVal = LoadSse4(second);
				__m128 val = _mm_mul_ps(firstVal, secondVal);
				__m128 resultVal = LoadSse4(result);
				resultVal = _mm_add_ps(resultVal, val);
				StoreSse4(resultVal, result);

				second += 4;
				result += 4;
			}

			if(nonSseSize > 0) {
				__m128 secondVal = LoadSse(second, nonSseSize);
				__m128 val = _mm_mul_ps(firstVal, secondVal);
				__m128 resultVal = LoadSse(result, nonSseSize);
				resultVal = _mm_add_ps(resultVal, val);
				StoreSse(resultVal, result, nonSseSize);

				second += nonSseSize;
				result += nonSseSize;
			}
		}
	}
}

void CCpuMathEngine::MultiplyLookupMatrixByLookupVector(int batchSize, const CLookupMatrix& matrix,
	const CLookupVector& vector, const CFloatHandle& resultHandle, int resultSize)
{
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR(matrix.Width() == vector.VectorSize());
	ASSERT_EXPR(resultSize >= batchSize * matrix.Height());

	int sseSize;
	int nonSseSize;
	checkSse(matrix.Width(), sseSize, nonSseSize);

	int height4 = matrix.Height() / 4;
	int height = matrix.Height() % 4;

	const float* matrixTable = GetRaw(matrix.Table);
	const int* matrixRows = GetRaw(matrix.Rows);
	const float* vectorTable = GetRaw(vector.Table);
	const int* vectors = GetRaw(vector.Vector);
	float* result = GetRaw(resultHandle);

	for(int b = 0; b < batchSize; ++b) {
		const float* vectorData = vectorTable + (*vectors++) * vector.VectorSize();

		for(int j = 0; j < height4; ++j) {
			// Multiply 4 matrix rows
			const float* rows[4];
			rows[0] = matrixTable + (*matrixRows++) * matrix.Width();
			rows[1] = matrixTable + (*matrixRows++) * matrix.Width();
			rows[2] = matrixTable + (*matrixRows++) * matrix.Width();
			rows[3] = matrixTable + (*matrixRows++) * matrix.Width();

			const float* vec = vectorData;
			__m128 prods[4];
			prods[3] = prods[2] = prods[1] = prods[0] = _mm_setzero_ps();

			for(int i = 0; i < sseSize; ++i) {
				__m128 vecVal = LoadSse4(vec);

				prods[0] = _mm_add_ps(prods[0], _mm_mul_ps(vecVal, LoadSse4(rows[0])));
				prods[1] = _mm_add_ps(prods[1], _mm_mul_ps(vecVal, LoadSse4(rows[1])));
				prods[2] = _mm_add_ps(prods[2], _mm_mul_ps(vecVal, LoadSse4(rows[2])));
				prods[3] = _mm_add_ps(prods[3], _mm_mul_ps(vecVal, LoadSse4(rows[3])));

				vec += 4;
				rows[0] += 4;
				rows[1] += 4;
				rows[2] += 4;
				rows[3] += 4;
			}

			if(nonSseSize > 0) {
				__m128 vecVal = LoadSse(vec, nonSseSize);

				prods[0] = _mm_add_ps(prods[0], _mm_mul_ps(vecVal, LoadSse(rows[0], nonSseSize)));
				prods[1] = _mm_add_ps(prods[1], _mm_mul_ps(vecVal, LoadSse(rows[1], nonSseSize)));
				prods[2] = _mm_add_ps(prods[2], _mm_mul_ps(vecVal, LoadSse(rows[2], nonSseSize)));
				prods[3] = _mm_add_ps(prods[3], _mm_mul_ps(vecVal, LoadSse(rows[3], nonSseSize)));
			}

			_MM_TRANSPOSE4_PS(prods[0], prods[1], prods[2], prods[3]);
			__m128 res = _mm_add_ps(_mm_add_ps(prods[0], prods[1]), _mm_add_ps(prods[2], prods[3]));

			StoreSse4(res, result);
			result += 4;
		}

		for(int j = 0; j < height; ++j) {
			// Multiply the rows one by one
			const float* row;
			row = matrixTable + (*matrixRows++) * matrix.Width();

			const float* vec = vectorData;
			__m128 prod;
			prod = _mm_setzero_ps();

			for(int i = 0; i < sseSize; ++i) {
				__m128 vecVal = LoadSse4(vec);

				prod = _mm_add_ps(prod, _mm_mul_ps(vecVal, LoadSse4(row)));

				vec += 4;
				row += 4;
			}

			if(nonSseSize > 0) {
				__m128 vecVal = LoadSse(vec, nonSseSize);

				prod = _mm_add_ps(prod, _mm_mul_ps(vecVal, LoadSse(row, nonSseSize)));
			}

			__m128 res = HorizontalAddSse(prod);

			*result++ = _mm_cvtss_f32(res);
		}
	}
}

} // namespace NeoML

#endif // NEOML_USE_SSE
