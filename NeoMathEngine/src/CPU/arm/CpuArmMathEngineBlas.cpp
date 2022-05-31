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

#ifdef NEOML_USE_NEON

#include <CpuMathEngine.h>
#include <CpuExecutionScope.h>
#include <CpuArm.h>
#include <MemoryHandleInternal.h>
#include <MathEngineCommon.h>
#include <CpuArmMathEngineVectorMathPrivate.h>

namespace NeoML {

static inline float euclidianNoNeon( const float* x, const float* y, const int size )
{
	float result = 0.f;
	for( int i = 0; i < size; ++i ) {
		const float num = x[i] - y[i];
		result += num * num;
	}
	return result;
}

static inline float euclidianNeon( const float* x, const float* y, const int size )
{
	float result = 0;

	int sseSize = size / 4;
	int nonNeonSize = size % 4;

	float32x4_t euclidean = vdupq_n_f32(0);
	for( int i = 0; i < sseSize; i++ ) {
		const float32x4_t neonX = LoadNeon4( x );
		const float32x4_t neonY = LoadNeon4( y );
		const float32x4_t diff = vsubq_f32 ( neonX, neonY );
		const float32x4_t diffSquared = vmulq_f32 ( diff, diff );
		euclidean = vaddq_f32( euclidean, diffSquared );
		x += 4;
		y += 4;
	}

	float32x2_t summ2 = vadd_f32( vget_high_f32( euclidean ), vget_low_f32( euclidean ) );
	result = vget_lane_f32( summ2, 0 ) + vget_lane_f32( summ2, 1 );

	if( nonNeonSize > 0 ) {
		result += euclidianNoNeon( x, y, nonNeonSize );
	}
	return result;
}

void CCpuMathEngine::MatrixRowsToVectorSquaredL2Distance( const CConstFloatHandle& matrixHandle,
	const int matrixHeight, const int matrixWidth, const CConstFloatHandle& vectorHandle,
	const CFloatHandle& resultHandle )
{
	ASSERT_EXPR( matrixHandle.GetMathEngine() == this );
	ASSERT_EXPR( vectorHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* matrix = GetRaw( matrixHandle );
	float* result = GetRaw( resultHandle );

	for( int i = 0; i < matrixHeight; i++ ) {
		const float* rawVector = GetRaw( vectorHandle );
		*result++ = euclidianNeon( matrix, rawVector, matrixWidth );
		matrix += matrixWidth;
	}
}

static inline void findMaxValueWorker(const float32x4_t& curVal, int32x4_t& curIndex,
	float32x4_t& maxVal, int32x4_t& maxIndex)
{
	maxIndex = ConditionIntNeon(vcgtq_f32(curVal, maxVal), curIndex, maxIndex);
	maxVal = vmaxq_f32(curVal, maxVal);
}

void CCpuMathEngine::FindMaxValueInRows(const CConstFloatHandle& matrixHandle, int matrixHeight, int matrixWidth,
	const CFloatHandle& resultHandle, const CIntHandle& columnIndices, int vectorSize)
{
	ASSERT_EXPR( matrixHandle.GetMathEngine() == this );
	ASSERT_EXPR( columnIndices.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( vectorSize >= matrixHeight );
	CCpuExecutionScope scope;

	const float* matrix = GetRaw(matrixHandle);
	float* result = GetRaw(resultHandle);
	int* indices = GetRaw(columnIndices);

	int count = GetCount4(matrixWidth);

	const int32x4_t initIndex = SetRegisterIntNeon(0, 1, 2, 3);
	const int32x4_t indexStep = vdupq_n_s32(4);

	for(int j = 0; j < matrixHeight; ++j) {
		float32x4_t maxVal = vdupq_n_f32(-FLT_MAX);
		int32x4_t curIndex = initIndex;
		int32x4_t maxIndex = curIndex;

		for(int i = 0; i < count; ++i) {
			float32x4_t curVal = LoadNeon4(matrix);
			findMaxValueWorker(curVal, curIndex, maxVal, maxIndex);
			curIndex = vaddq_s32(curIndex, indexStep);
			matrix += 4;
		}

		if(matrixWidth > 0) {
			float32x4_t curVal = LoadNeon(matrix, matrixWidth, -FLT_MAX);
			findMaxValueWorker(curVal, curIndex, maxVal, maxIndex);
			matrix += matrixWidth;
		}

		float32x2_t res;
		int32x2_t resIndex;
		HorizontalMaxWithIndexNeon(maxVal, maxIndex, res, resIndex);

		*result++ = vget_lane_f32(res, 0);
		*indices++ = vget_lane_s32(resIndex, 0);
	}
}

void CCpuMathEngine::FindMaxValueInRows(const CConstFloatHandle& matrixHandle, int matrixHeight, int matrixWidth,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( matrixHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR(vectorSize >= matrixHeight);
	CCpuExecutionScope scope;

	const float* matrix = GetRaw(matrixHandle);
	float* result = GetRaw(resultHandle);

	int count = GetCount4(matrixWidth);

	for(int j = 0; j < matrixHeight; ++j) {
		float32x4_t maxVal = vdupq_n_f32(-FLT_MAX);
		for(int i = 0; i < count; ++i) {
			float32x4_t curVal = LoadNeon4(matrix);
			maxVal = vmaxq_f32(maxVal, curVal);
			matrix += 4;
		}

		if(matrixWidth > 0) {
			float32x4_t curVal = LoadNeon(matrix, matrixWidth, -FLT_MAX);
			maxVal = vmaxq_f32(maxVal, curVal);
			matrix += matrixWidth;
		}

		*result++ = vget_lane_f32(HorizontalMaxNeon(maxVal), 0);
	}
}

void CCpuMathEngine::FindMaxValueInColumns( int batchSize, const CConstFloatHandle& matrixHandle,
	int matrixHeight, int matrixWidth, const CFloatHandle& resultHandle, const CIntHandle& rowIndices, int vectorSize )
{
	ASSERT_EXPR( matrixHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( rowIndices.GetMathEngine() == this );
	CCpuExecutionScope scope;

	if( matrixWidth == 1 ) {
		// In this case FindMaxValueInRows would be more optimal because it uses Neon in a different way
		FindMaxValueInRows( matrixHandle, batchSize, matrixHeight, resultHandle, rowIndices, vectorSize );
		return;
	}

	ASSERT_EXPR( vectorSize >= batchSize * matrixWidth );

	// The pointer moves over the first row of the first matrix in the batch
	const float* firstRow = GetRaw(matrixHandle);
	// The pointer moves over the results calculated for the first matrix
	float* firstMatrixRes = GetRaw( resultHandle );
	int* firstMatrixIndices = GetRaw( rowIndices );

	// Store the number of remaining columns in a separate variable
	// because matrixWidth will be also used to offset the pointers
	int remainingColumns = matrixWidth;
	int count = GetCount4(remainingColumns);

	// Process 4 rows in each matrix on one iteration
	for( int i = 0; i < count; ++i ) {
		const float* data = firstRow;
		float* result = firstMatrixRes;
		int* indices = firstMatrixIndices;
		for( int b = 0; b < batchSize; ++b ) {
			// Copy the first elements from 4 rows of the current matrix
			float32x4_t maxValue = LoadNeon4( data );
			int32x4_t maxIndices = vdupq_n_s32( 0 );
			data += matrixWidth;
			for( int h = 1; h < matrixHeight; ++h ) {
				float32x4_t value = LoadNeon4( data );
				int32x4_t currIndices = vdupq_n_s32( h );
				findMaxValueWorker( value, currIndices, maxValue, maxIndices );
				data += matrixWidth;
			}
			StoreNeon4( maxValue, result );
			StoreIntNeon4( maxIndices, indices );
			result += matrixWidth;
			indices += matrixWidth;
		}
		firstRow += 4;
		firstMatrixRes += 4;
		firstMatrixIndices += 4;
	}

	if( remainingColumns > 0 ) {
		// Process the rest of the columns in a similar way
		const float* data = firstRow;
		float* result = firstMatrixRes;
		int* indices = firstMatrixIndices;
		for( int b = 0; b < batchSize; ++b ) {
			float32x4_t maxValue = LoadNeon( data, remainingColumns );
			int32x4_t maxIndices = vdupq_n_s32( 0 );
			data += matrixWidth;
			for( int h = 1; h < matrixHeight; ++h ) {
				float32x4_t value = LoadNeon( data, remainingColumns );
				int32x4_t currIndices = vdupq_n_s32( h );
				findMaxValueWorker( value, currIndices, maxValue, maxIndices );
				data += matrixWidth;
			}
			StoreNeon( maxValue, result, remainingColumns );
			StoreIntNeon( maxIndices, indices, remainingColumns );
			result += matrixWidth;
			indices += matrixWidth;
		}
	}
}

void CCpuMathEngine::FindMinValueInColumns( const CConstFloatHandle& matrixHandle, int matrixHeight,
	int matrixWidth, const CFloatHandle& resultHandle, const CIntHandle& rowIndicesHandle )
{
	ASSERT_EXPR( matrixHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( rowIndicesHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* matrix = GetRaw( matrixHandle );
	float* result = GetRaw( resultHandle );
	int* rowIndices = GetRaw( rowIndicesHandle );

	// Copy the first row
	dataCopy( result, matrix, matrixWidth );
	vectorFill( rowIndices, 0, matrixWidth );
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

void CCpuMathEngine::MultiplyLookupMatrixByLookupVector(int batchSize, const CLookupMatrix& matrix,
	const CLookupVector& vector, const CFloatHandle& resultHandle, int resultSize)
{
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR(matrix.Width() == vector.VectorSize());
	ASSERT_EXPR(resultSize >= batchSize * matrix.Height());
	CCpuExecutionScope scope;

	int height = matrix.Height();
	int height4 = GetCount4(height);
	int width = matrix.Width();
	int width4 = GetCount4(width);

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
			CMatrixBlock4x4 prod;
			prod.Rows[3] = prod.Rows[2] = prod.Rows[1] = prod.Rows[0] = vdupq_n_f32(0);

			for(int i = 0; i < width4; ++i) {
				float32x4_t vecVal = LoadNeon4(vec);

				prod.Rows[0] = MultiplyAndAddNeon(prod.Rows[0], vecVal, LoadNeon4(rows[0]));
				prod.Rows[1] = MultiplyAndAddNeon(prod.Rows[1], vecVal, LoadNeon4(rows[1]));
				prod.Rows[2] = MultiplyAndAddNeon(prod.Rows[2], vecVal, LoadNeon4(rows[2]));
				prod.Rows[3] = MultiplyAndAddNeon(prod.Rows[3], vecVal, LoadNeon4(rows[3]));

				vec += 4;
				rows[0] += 4;
				rows[1] += 4;
				rows[2] += 4;
				rows[3] += 4;
			}

			if(width > 0) {
				float32x4_t vecVal = LoadNeon(vec, width, 0);

				prod.Rows[0] = MultiplyAndAddNeon(prod.Rows[0], vecVal, LoadNeon(rows[0], width, 0));
				prod.Rows[1] = MultiplyAndAddNeon(prod.Rows[1], vecVal, LoadNeon(rows[1], width, 0));
				prod.Rows[2] = MultiplyAndAddNeon(prod.Rows[2], vecVal, LoadNeon(rows[2], width, 0));
				prod.Rows[3] = MultiplyAndAddNeon(prod.Rows[3], vecVal, LoadNeon(rows[3], width, 0));
			}

			prod.Transpose();
			float32x4_t res = vaddq_f32(vaddq_f32(prod.Rows[0], prod.Rows[1]), vaddq_f32(prod.Rows[2], prod.Rows[3]));

			StoreNeon4(res, result);
			result += 4;
		}

		for(int j = 0; j < height; ++j) {
			// Multiply matrix rows one by one
			const float* row;
			row = matrixTable + (*matrixRows++) * matrix.Width();

			const float* vec = vectorData;
			float32x4_t prod;
			prod = vdupq_n_f32(0);

			for(int i = 0; i < width4; ++i) {
				float32x4_t vecVal = LoadNeon4(vec);

				prod = MultiplyAndAddNeon(prod, vecVal, LoadNeon4(row));

				vec += 4;
				row += 4;
			}

			if(width > 0) {
				float32x4_t vecVal = LoadNeon(vec, width, 0);

				prod = MultiplyAndAddNeon(prod, vecVal, LoadNeon(row, width, 0));
			}

			float32x2_t res = HorizontalAddNeon(prod);

			*result++ = vget_lane_f32(res, 0);
		}
	}
}

void CCpuMathEngine::MultiplySparseMatrixByTransposedMatrix( int firstHeight, int firstWidth, int secondHeight,
	const CSparseMatrixDesc& firstDesc, const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle )
{
	ASSERT_EXPR( firstDesc.Rows.GetMathEngine() == this );
	ASSERT_EXPR( firstDesc.Columns.GetMathEngine() == this );
	ASSERT_EXPR( firstDesc.Values.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const int* firstRows = GetRaw( firstDesc.Rows );
	const int* firstColumns = GetRaw( firstDesc.Columns );
	const float* values = GetRaw( firstDesc.Values );
	const float* second = GetRaw( secondHandle );
	float* res = GetRaw( resultHandle );

	for( int col = 0; col < secondHeight; ++col ) {
		float* result = res;
		for( int row = 0; row < firstHeight; ++row ){
			float resultVal = 0;
			for( int ind = firstRows[row]; ind < firstRows[row + 1]; ++ind ) {
				resultVal += values[ind] * second[firstColumns[ind]];
			}
			result[col] = resultVal;
			result += secondHeight;
		}
		second += firstWidth;
	}
}

void CCpuMathEngine::MultiplyTransposedMatrixBySparseMatrixAndAdd( int firstHeight, int firstWidth, int secondWidth,
	const CConstFloatHandle& firstHandle, const CSparseMatrixDesc& secondDesc, const CFloatHandle& resultHandle )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondDesc.Rows.GetMathEngine() == this );
	ASSERT_EXPR( secondDesc.Columns.GetMathEngine() == this );
	ASSERT_EXPR( secondDesc.Values.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw( firstHandle );
	const int* secondRows = GetRaw( secondDesc.Rows );
	const int* secondColumns = GetRaw( secondDesc.Columns );
	const float* secondValues = GetRaw( secondDesc.Values );
	float* result = GetRaw( resultHandle );

	for( int row = 0; row < firstHeight; ++row ) {
		for( int ind = secondRows[row]; ind < secondRows[row + 1]; ++ind ) {
			for( int col = 0; col < firstWidth; ++col ) {
				result[col * secondWidth + secondColumns[ind]] += first[col] * secondValues[ind];
			}
		}
		first += firstWidth;
	}
}

void CCpuMathEngine::MultiplyDiagMatrixByMatrixAndAdd(int batchSize, const CConstFloatHandle& firstHandle,
	int firstSize, const CConstFloatHandle& secondHandle, int secondWidth, const CFloatHandle& resultHandle)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	const float* second = GetRaw(secondHandle);
	float* resultStart = GetRaw(resultHandle);

	int secondWidth4 = GetCount4(secondWidth);

	for(int b = 0; b < batchSize; ++b) {
		float* result = resultStart;
		float32x4_t firstValBuf = vdupq_n_f32(0);
		for(int j = 0; j < firstSize; ++j) {
			int phase = j % 4;
			if(phase == 0) {
				int loadSize = firstSize - j;
				if(loadSize >= 4) {
					loadSize = 4;
					firstValBuf = LoadNeon4(first);
				} else {
					firstValBuf = LoadNeon(first, loadSize);
				}
				first += loadSize;
			}
			float firstVal = GetLaneNeon(firstValBuf, phase);

			for(int i = 0; i < secondWidth4; ++i) {
				float32x4_t secondVal = LoadNeon4(second);
				float32x4_t resultVal = LoadNeon4(result);
				resultVal = vmlaq_n_f32(resultVal, secondVal, firstVal);
				StoreNeon4(resultVal, result);

				second += 4;
				result += 4;
			}

			if(secondWidth > 0) {
				float32x4_t secondVal = LoadNeon(second, secondWidth);
				float32x4_t resultVal = LoadNeon(result, secondWidth);
				resultVal = vmlaq_n_f32(resultVal, secondVal, firstVal);
				StoreNeon(resultVal, result, secondWidth);

				second += secondWidth;
				result += secondWidth;
			}
		}
	}
}

void CCpuMathEngine::SingularValueDecomposition( const CFloatHandle&, int, int, const CFloatHandle&, const CFloatHandle&,
	const CFloatHandle&, const CFloatHandle&, bool, bool )
{
	ASSERT_EXPR( false );
}

void CCpuMathEngine::MultiplyTransposedMatrixBySparseMatrix( int, int, int, const CConstFloatHandle&, const CSparseMatrixDesc&, const CFloatHandle&, bool )
{
    ASSERT_EXPR( false );
}

void CCpuMathEngine::MultiplySparseMatrixByMatrix( int, int, int, const CSparseMatrixDesc&, const CConstFloatHandle&, const CFloatHandle& )
{
    ASSERT_EXPR( false );
}

void CCpuMathEngine::MultiplyTransposedSparseMatrixByMatrix( int, int, int, const CSparseMatrixDesc&, const CConstFloatHandle&, const CFloatHandle& )
{
    ASSERT_EXPR( false );
}

void CCpuMathEngine::QRFactorization( int, int, const CFloatHandle&, const CFloatHandle*, const CFloatHandle*, bool, bool, bool )
{
    ASSERT_EXPR( false );
}

} // namespace NeoML

#endif // NEOML_USE_NEON
