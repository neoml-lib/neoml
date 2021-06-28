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
#include <CpuRandom.h>
#include <CpuMathEnginePrivate.h>

namespace NeoML {

void CCpuMathEngine::VectorFill(const CFloatHandle& result, int vectorSize, const CConstFloatHandle& value)
{
	ASSERT_EXPR( result.GetMathEngine() == this );
	ASSERT_EXPR( value.GetMathEngine() == this );

	VectorFill(result, *GetRaw(value), vectorSize);
}

void CCpuMathEngine::VectorFill(const CIntHandle& result, int vectorSize, const CConstIntHandle& value)
{
	ASSERT_EXPR( result.GetMathEngine() == this );
	ASSERT_EXPR( value.GetMathEngine() == this );

	VectorFill(result, *GetRaw(value), vectorSize);
}

void CCpuMathEngine::VectorCopy(const CFloatHandle& firstHandle, const CConstFloatHandle& secondHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );

	vectorCopy( GetRaw( firstHandle ), GetRaw( secondHandle ), vectorSize );
}

void CCpuMathEngine::VectorAdd(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

	const int curThreadCount = IsOmpRelevant( vectorSize, vectorSize ) ? threadCount : 1;

	NEOML_OMP_NUM_THREADS( curThreadCount )
	{
		int index;
		int count;
		if( OmpGetTaskIndexAndCount( vectorSize, 16, index, count ) ) {
			NeoML::vectorAdd( GetRaw(firstHandle + index), GetRaw(secondHandle + index), GetRaw(resultHandle + index), count );
		}
	}
}

void CCpuMathEngine::VectorSum(const CConstFloatHandle& firstHandle, int vectorSize, const CFloatHandle& resultHandle)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

	*GetRaw(resultHandle) = 0;
	VectorSumAdd(firstHandle, vectorSize, resultHandle);
}

void CCpuMathEngine::VectorNegSum(const CConstFloatHandle& firstHandle, int vectorSize, const CFloatHandle& resultHandle)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

	VectorSum(firstHandle, vectorSize, resultHandle);
	*GetRaw(resultHandle) = -*GetRaw(resultHandle);
}

void CCpuMathEngine::VectorSumAlongDimension( const CConstFloatHandle& firstHandle, int precedingDimension, int dimension,
	int followingDimension, const CFloatHandle& resultHandle )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

	int firstIndex = 0;
	int resultIndex = 0;

	for( int i = 0; i < followingDimension; i++ ) {
		VectorCopy( resultHandle + resultIndex, firstHandle + firstIndex, precedingDimension );
		firstIndex += precedingDimension;
		for( int j = 1; j < dimension; j++ ) {
			VectorAdd(  firstHandle + firstIndex, resultHandle + resultIndex, resultHandle + resultIndex, precedingDimension );
			firstIndex += precedingDimension;
		}
		resultIndex += precedingDimension;
	}
}

void CCpuMathEngine::VectorFillBernoulli( const CFloatHandle& result, float p, int vectorSize, float value, int seed )
{
	ASSERT_EXPR( result.GetMathEngine() == this );
	
	float* const resultPtr = GetRaw( result );
	const unsigned int threshold = ( unsigned int ) ( ( double ) p * UINT_MAX );

	CCpuRandom random( seed );

	int index = 0;
	for( int i = 0; i < ( vectorSize + 3 ) / 4; ++i ) {
		CIntArray<4> generated = random.Next();
		for( int j = 0; j < 4 && index < vectorSize; ++j ) {
			resultPtr[index++] = ( generated[j] <= threshold ) ? value : 0.f;
		}
	}
}

void CCpuMathEngine::VectorFindMaxValueInSet( const CConstFloatHandle* vectors, int vectorCount,
	const CFloatHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR( vectorCount > 0 );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

	if( vectorCount == 1 ) {
		VectorCopy( resultHandle, vectors[0], vectorSize );
		return;
	}

	VectorEltwiseMax( vectors[0], vectors[1], resultHandle, vectorSize );

	for( int i = 2; i < vectorCount; ++i ) {
		VectorEltwiseMax( vectors[i], resultHandle, resultHandle, vectorSize );
	}
}

void CCpuMathEngine::VectorFindMaxValueInSet( const CConstFloatHandle* vectors, int vectorCount, const CFloatHandle& resultHandle,
	const CIntHandle& indexHandle, int vectorSize )
{
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( indexHandle.GetMathEngine() == this );
	ASSERT_EXPR( vectorCount > 0 );

	VectorFill( indexHandle, 0, vectorSize );
	VectorCopy( resultHandle, vectors[0], vectorSize );

	int* indices = GetRaw( indexHandle );

	float* result = GetRaw( resultHandle );

	for( int j = 1; j < vectorCount; ++j ) {
		ASSERT_EXPR( vectors[j].GetMathEngine() == this );
		const float* vectorData = GetRaw( vectors[j] );
		for( int i = 0; i < vectorSize; ++i ) {
			if( vectorData[i] > result[i] ) {
				result[i] = vectorData[i];
				indices[i] = j;
			}
		}
	}
}

void CCpuMathEngine::VectorSpreadValues( const CConstFloatHandle& sourceHandle, CFloatHandle* vectors, int vectorCount,
	const CConstIntHandle& indexHandle, int vectorSize )
{
	ASSERT_EXPR( sourceHandle.GetMathEngine() == this );
	ASSERT_EXPR( indexHandle.GetMathEngine() == this );

	const int* indices = GetRaw( indexHandle );
	const float* source = GetRaw( sourceHandle );

	for( int i = 0; i < vectorSize; ++i ) {
		int index = *indices++;
		float value = *source++;
		if( 0 <= index && index < vectorCount ) {
			CFloatHandle vector = vectors[index];
			ASSERT_EXPR( vector.GetMathEngine() == this );
			*GetRaw( vector + i ) = value;
		}
	}
}

void CCpuMathEngine::VectorAddValue(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle,
	int vectorSize, const CConstFloatHandle& addition)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( addition.GetMathEngine() == this );

	const float* first = GetRaw( firstHandle );
	float* result = GetRaw( resultHandle );
	float value = *GetRaw( addition );

	vectorAddValue( first, result, vectorSize, value );
}

void CCpuMathEngine::VectorDotProduct(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	int vectorSize, const CFloatHandle& resultHandle)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

	const float* first = GetRaw( firstHandle );
	const float* second = GetRaw( secondHandle );
	float* result = GetRaw( resultHandle );

	vectorDotProduct( first, second, vectorSize, result );
}

void CCpuMathEngine::VectorTopK(const CConstFloatHandle& firstHandle, int firstSize, int k, const CFloatHandle& resultHandle,
	const CIntHandle& indicesHandle)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( firstSize >= 0 );
	ASSERT_EXPR( k > 0 );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( indicesHandle.GetMathEngine() == this );

	const float* first = GetRaw( firstHandle );
	float* result = GetRaw( resultHandle );
	int* indices = GetRaw( indicesHandle );
	int size = 0;

	for( int i = 0; i < firstSize; i++ ) {
		int pos = 0;
		for( pos = 0; pos < size; pos++ ) {
			if( *first > result[pos] ) {
				for( int j = min(size + 1, k) - 1; j >= pos + 1; j-- ) {
					result[j] = result[j - 1];
					indices[j] = indices[j - 1];
				}
				break;
			}
		}
		if( pos < k ) {
			result[pos] = *first;
			indices[pos] = i;
			size = min( size + 1, k );
		}
		first++;
	}
}

void CCpuMathEngine::VectorTopKDiff(const CConstFloatHandle& sourceGradHandle, int sourceGradHeight, int sourceGradWidth,
	const CConstIntHandle& indicesHandle, int k, const CFloatHandle& resultGradHandle)
{
	ASSERT_EXPR( sourceGradHandle.GetMathEngine() == this );
	ASSERT_EXPR( sourceGradHeight > 0 );
	ASSERT_EXPR( sourceGradWidth > 0 );
	ASSERT_EXPR( indicesHandle.GetMathEngine() == this );
	ASSERT_EXPR( k > 0 );
	ASSERT_EXPR( resultGradHandle.GetMathEngine() == this );

	const float* sourceGrad = GetRaw( sourceGradHandle );
	const int* indices = GetRaw( indicesHandle );
	float* resultGrad = GetRaw( resultGradHandle );

	if( sourceGradHeight == 1 ) {
		vectorFill0( resultGrad, k * sourceGradWidth );
		for( int i = 0; i < k; i++ ) {
			const int pos = indices[i];
			resultGrad[pos] = sourceGrad[pos];

			resultGrad += sourceGradWidth;
		}
		return;
	}

	for( int i = 0; i < k; i++ ) {
		const int pos = indices[i] * sourceGradWidth ;
		vectorCopy( resultGrad, sourceGrad + pos, sourceGradWidth );

		resultGrad += sourceGradWidth;
	}
}

void CCpuMathEngine::VectorEltwiseMultiply(const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

	const float* first = GetRaw(firstHandle);
	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);

	NeoML::vectorEltwiseMultiply( first, second, result, vectorSize );
}

void CCpuMathEngine::VectorEltwiseMultiplyAdd( const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

	const float* first = GetRaw(firstHandle);
	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);

	NeoML::vectorEltwiseMultiplyAdd( first ,second, result, vectorSize );
}

void CCpuMathEngine::VectorAbsDiff(const CConstFloatHandle& sourceGradHandle, int gradHeight, int gradWidth,
	const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle)
{
	ASSERT_EXPR( sourceGradHandle.GetMathEngine() == this );
	ASSERT_EXPR( gradHeight > 0 );
	ASSERT_EXPR( gradWidth > 0 );
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

	const float* first = GetRaw(firstHandle);
	const float* grad = GetRaw(sourceGradHandle);
	float* result = GetRaw(resultHandle);

	const int firstSize = gradHeight == 1 ? gradWidth : gradHeight;
	const int gradSize = gradHeight == 1 ? 1 : gradWidth;

	for( int i = 0; i < firstSize; i++ ) {
		if( *first > 0 ) {
			for( int j = 0; j < gradSize; j++ ) {
				*result = *grad;
				result++;
				grad++;
			}
		} else {
			for( int j = 0; j < gradSize; j++ ) {
				*result = -*grad;
				result++;
				grad++;
			}
		}
		first++;
	}
}

void CCpuMathEngine::VectorMax( const CConstFloatHandle& firstHandle, float secondValue, const CFloatHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

	const float* first = GetRaw( firstHandle );
	float* result = GetRaw( resultHandle );

	for( int i = 0; i < vectorSize; ++i ) {
		*result = ( *first >= secondValue ) ? *first : secondValue;
		result++;
		first++;
	}
}

void CCpuMathEngine::VectorMaxDiff( const CConstFloatHandle& firstHandle, float secondValue, const CFloatHandle& gradHandle,
	int gradHeight, int gradWidth )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( gradHandle.GetMathEngine() == this );
	ASSERT_EXPR( gradHeight > 0 );
	ASSERT_EXPR( gradWidth > 0 );

	const float* first = GetRaw( firstHandle );
	float* grad = GetRaw( gradHandle );

	const int firstSize = gradHeight == 1 ? gradWidth : gradHeight;
	const int gradSize =  gradHeight == 1 ? 1 : gradWidth;

	for( int i = 0; i < firstSize; ++i ) {
		if( *first < secondValue ) {
			vectorFill( grad, 0.0f, gradSize );
		}
		grad += gradSize;
		first++;
	}
}

void CCpuMathEngine::VectorLogDiff( const CConstFloatHandle& sourceGradHandle, int sourceGradHeight, int sourceGradWidth,
	const CConstFloatHandle& valueHandle, const CFloatHandle& resultHandle )
{
	ASSERT_EXPR( sourceGradHandle.GetMathEngine() == this );
	ASSERT_EXPR( sourceGradHeight > 0 );
	ASSERT_EXPR( sourceGradWidth > 0 );
	ASSERT_EXPR( valueHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

	const float* sourceGrad = GetRaw(sourceGradHandle);
	const float* value = GetRaw(valueHandle);
	float* result = GetRaw(resultHandle);

	const int valueSize = sourceGradHeight == 1 ? sourceGradWidth : sourceGradHeight;
	const int gradSize = sourceGradHeight == 1 ? 1 : sourceGradWidth;
	for( int i = 0; i < valueSize; ++i ) {
		float div = *value++;
		if( (-FLT_MIN <= div && div < 0) || (0 <= div && div <= FLT_MIN) ) {
			for( int j = 0; j < gradSize; j++ ) {
				*result++ = 0;
				sourceGrad++;
			}
		} else {
			for( int j = 0; j < gradSize; j++ ) {
				*result++ = *sourceGrad++ / div;
			}
		}
	}
}

void CCpuMathEngine::VectorNeg(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );

	const float* first = GetRaw(firstHandle);
	float* result = GetRaw(resultHandle);
	for(int i = 0; i < vectorSize; ++i) {
		*result++ = -*first++;
	}
}

void CCpuMathEngine::VectorMinMaxDiff(const CConstFloatHandle& sourceGradHandle, int gradHeight, int gradWidth,
	const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle,
	const CConstFloatHandle& minHandle, const CConstFloatHandle& maxHandle)
{
	ASSERT_EXPR( sourceGradHandle.GetMathEngine() == this );
	ASSERT_EXPR( gradHeight > 0 );
	ASSERT_EXPR( gradWidth > 0 );
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( minHandle.GetMathEngine() == this );
	ASSERT_EXPR( maxHandle.GetMathEngine() == this );

	const float* first = GetRaw(firstHandle);
	const float* sourceGrad = GetRaw(sourceGradHandle);
	float* result = GetRaw(resultHandle);
	const float minValue = *GetRaw(minHandle);
	const float maxValue = *GetRaw(maxHandle);

	const int firstSize = gradHeight == 1 ? gradWidth : gradHeight;
	const int gradSize = gradHeight == 1 ? 1 : gradWidth;

	for( int i = 0; i < firstSize; ++i ) {
		if( *first < minValue || *first > maxValue ) {
			for( int j = 0; j < gradSize; j++ ) {
				*result = 0;
				result++;
				sourceGrad++;
			}
		} else {
			for( int j = 0; j < gradSize; j++ ) {
				*result = *sourceGrad;
				result++;
				sourceGrad++;
			}
		}
		first++;
	}
}

} // namespace NeoML
