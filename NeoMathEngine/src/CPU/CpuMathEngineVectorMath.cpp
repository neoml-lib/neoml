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
#include <CpuExecutionScope.h>
#include <MemoryHandleInternal.h>
#include <MathEngineCommon.h>
#include <CpuRandom.h>
#include <CpuMathEnginePrivate.h>
#include <cmath>
#include <functional>

namespace NeoML {

// Applies singleThreadFunction on the data of vectorSize by using threadCount omp threads
template<class T>
static void applyOmpVectorFunction( int threadCount, int vectorSize, T& function )
{
	if( threadCount > 1 ) {
		NEOML_OMP_NUM_THREADS( threadCount ) {
			int index, count;
			if( OmpGetTaskIndexAndCount( vectorSize, 16, index, count ) ) {
				function( index, count );
			}
		}
	} else {
		function( 0, vectorSize );
	}
}

// Class which wraps binary vector functor into OMP-friendly interface
template<class TFunctor>
class COmpBinaryVectorFunction {
public:
	using TFirst = typename TFunctor::TFirst;
	using TSecond = typename TFunctor::TSecond;
	using TResult = typename TFunctor::TResult;

	COmpBinaryVectorFunction( const TFirst* first, const TSecond* second, TResult* result,
			const TFunctor& functor = TFunctor(), TFirst firstDefaultValue = 1, TSecond secondDefaultValue = 1 ) :
		function( functor, firstDefaultValue, secondDefaultValue ),
		first( first ),
		second( second ),
		result( result )
	{
	}

	void operator()( int index, int count )
	{
		function( first + index, second + index, result + index, count );
	}

private:
	CBinaryVectorFunction<TFunctor> function;
	const TFirst* const first;
	const TSecond* const second;
	TResult* const result;
};

// Class which wraps ternary vector functor into OMP-friendly interface
template<class TFunctor>
class COmpTernaryVectorFunction {
public:
	using TFirst = typename TFunctor::TFirst;
	using TSecond = typename TFunctor::TSecond;
	using TThird = typename TFunctor::TThird;
	using TResult = typename TFunctor::TResult;

	COmpTernaryVectorFunction( const TFirst* first, const TSecond* second, const TThird* third, TResult* result,
			const TFunctor& functor = TFunctor(), TFirst firstDefaultValue = 1, TSecond secondDefaultValue = 1,
			TThird thirdDefaultValue = 1 ) :
		function( functor, firstDefaultValue, secondDefaultValue, thirdDefaultValue ),
		first( first ),
		second( second ),
		third( third ),
		result( result )
	{
	}

	void operator()( int index, int count )
	{
		function( first + index, second + index, third + index, result + index, count );
	}

private:
	CTernaryVectorFunction<TFunctor> function;
	const TFirst* const first;
	const TSecond* const second;
	const TThird* const third;
	TResult* const result;
};

void CCpuMathEngine::VectorFill(const CFloatHandle& result, int vectorSize, const CConstFloatHandle& value)
{
	ASSERT_EXPR( result.GetMathEngine() == this );
	ASSERT_EXPR( value.GetMathEngine() == this );
	CCpuExecutionScope scope;

	VectorFill(result, *GetRaw(value), vectorSize);
}

void CCpuMathEngine::VectorFill(const CIntHandle& result, int vectorSize, const CConstIntHandle& value)
{
	ASSERT_EXPR( result.GetMathEngine() == this );
	ASSERT_EXPR( value.GetMathEngine() == this );
	CCpuExecutionScope scope;

	VectorFill(result, *GetRaw(value), vectorSize);
}

void CCpuMathEngine::VectorCopy(const CFloatHandle& firstHandle, const CConstFloatHandle& secondHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const int curThreadCount = IsOmpRelevant( vectorSize, vectorSize ) ? threadCount : 1;

	if( curThreadCount > 1 ) {
		NEOML_OMP_NUM_THREADS( curThreadCount ) {
			int index, count;
			if( OmpGetTaskIndexAndCount( vectorSize, 16, index, count ) ) {
				dataCopy( GetRaw( firstHandle + index ), GetRaw( secondHandle + index ), count );
			}
		}
	} else {
		dataCopy( GetRaw( firstHandle ), GetRaw( secondHandle ), vectorSize );
	}
}

void CCpuMathEngine::VectorCopy( const CIntHandle& firstHandle, const CConstIntHandle& secondHandle, int vectorSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const int curThreadCount = IsOmpRelevant( vectorSize, vectorSize ) ? threadCount : 1;

	NEOML_OMP_NUM_THREADS( curThreadCount )
	{
		int index;
		int count;
		if( OmpGetTaskIndexAndCount( vectorSize, 16, index, count ) ) {
			dataCopy( GetRaw( firstHandle + index ), GetRaw( secondHandle + index ), count );
		}
	}
}

template<class T>
void broadcastCopyImpl( T* to, const T* from, const CBlobDesc& toDesc, const CBlobDesc& fromDesc, int additionalWidth )
{
	int curSize = fromDesc.BlobSize() * additionalWidth;
	int copySize = additionalWidth;
	dataCopy( to, from, curSize );

	for( int i = BD_Count - 1; i >= 0; i-- ) {
		if( toDesc.DimSize( i ) != fromDesc.DimSize( i ) ) {
			T* fromPtr = to + curSize - copySize;
			T* toPtr = to + curSize * toDesc.DimSize( i ) - copySize;
			for( int j = 0; j < curSize / copySize; j++ ) {
				if( copySize == 1 ) {
					toPtr -= toDesc.DimSize( i );
					vectorFill( toPtr + 1, *fromPtr, toDesc.DimSize( i ) );
				} else {
					for( int k = 0; k < toDesc.DimSize( i ); k++ ) {
						dataCopy( toPtr, fromPtr, copySize );
						toPtr -= copySize;
					}
				}
				fromPtr -= copySize;
			}
			curSize *= toDesc.DimSize( i );
		}
		copySize *= toDesc.DimSize( i );
	}
}

void CCpuMathEngine::BroadcastCopy(const CIntHandle& toHandle, const CConstIntHandle& fromHandle,
	const CBlobDesc& toDesc, const CBlobDesc& fromDesc, int additionalWidth)
{
	ASSERT_EXPR( toHandle.GetMathEngine() == this );
	ASSERT_EXPR( fromHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;
	for( int i = 0; i < BD_Count; i++ ) {
		ASSERT_EXPR( fromDesc.DimSize( i ) == 1 || fromDesc.DimSize( i ) == toDesc.DimSize( i ) );
	}

	broadcastCopyImpl( GetRaw( toHandle ), GetRaw( fromHandle ), toDesc, fromDesc, additionalWidth );
}

void CCpuMathEngine::BroadcastCopy(const CFloatHandle& toHandle, const CConstFloatHandle& fromHandle,
	const CBlobDesc& toDesc, const CBlobDesc& fromDesc, int additionalWidth)
{
	ASSERT_EXPR( toHandle.GetMathEngine() == this );
	ASSERT_EXPR( fromHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;
	for( int i = 0; i < BD_Count; i++ ) {
		ASSERT_EXPR( fromDesc.DimSize( i ) == 1 || fromDesc.DimSize( i ) == toDesc.DimSize( i ) );
	}

	broadcastCopyImpl( GetRaw( toHandle ), GetRaw( fromHandle ), toDesc, fromDesc, additionalWidth );
}

void CCpuMathEngine::VectorAdd(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const int curThreadCount = IsOmpRelevant( vectorSize, vectorSize ) ? threadCount : 1;

	if( curThreadCount > 1 ) {
		NEOML_OMP_NUM_THREADS( curThreadCount ) {
			int index, count;
			if( OmpGetTaskIndexAndCount( vectorSize, 16, index, count ) ) {
				NeoML::vectorAdd( GetRaw(firstHandle + index), GetRaw(secondHandle + index), GetRaw(resultHandle + index), count );
			}
		}
	} else {
		NeoML::vectorAdd( GetRaw(firstHandle), GetRaw(secondHandle), GetRaw(resultHandle), vectorSize );
	}
}

void CCpuMathEngine::VectorSum(const CConstFloatHandle& firstHandle, int vectorSize, const CFloatHandle& resultHandle)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	*GetRaw(resultHandle) = 0;
	VectorSumAdd(firstHandle, vectorSize, resultHandle);
}

void CCpuMathEngine::VectorSumAlongDimension( const CConstFloatHandle& firstHandle, int precedingDimension, int dimension,
	int followingDimension, const CFloatHandle& resultHandle )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	int firstIndex = 0;
	int resultIndex = 0;

	const float* first = GetRaw( firstHandle );
	float* result = GetRaw( resultHandle );

	for( int i = 0; i < followingDimension; i++ ) {
		dataCopy( result + resultIndex, first + firstIndex, precedingDimension );
		firstIndex += precedingDimension;
		for( int j = 1; j < dimension; j++ ) {
			vectorAdd( first + firstIndex, result + resultIndex, result + resultIndex, precedingDimension );
			firstIndex += precedingDimension;
		}
		resultIndex += precedingDimension;
	}
}

template<class T>
static void vectorCumSumAlongDimensionImpl( const CTypedMemoryHandle<const T>& firstHandle, int precedingDimension,
	int dimension, int followingDimension, const CTypedMemoryHandle<T>& resultHandle, bool reverse )
{
	const T* first = GetRaw( firstHandle );
	T* result = GetRaw( resultHandle );
	const int step = reverse ? -precedingDimension : precedingDimension;
	const int firstElemOffset = reverse ? ( dimension - 1 ) * precedingDimension : 0;

	for( int i = 0; i < followingDimension; i++ ) {
		int index = i * dimension * precedingDimension + firstElemOffset;
		dataCopy( result + index, first + index, precedingDimension );
		index += step;
		for( int j = 1; j < dimension; j++ ) {
			dataCopy( result + index, result + index - step, precedingDimension );
			vectorAdd( first + index, result + index, result + index, precedingDimension );
			index += step;
		}
	}
}

void CCpuMathEngine::VectorCumSumAlongDimension( const CConstFloatHandle& firstHandle, int precedingDimension, int dimension,
	int followingDimension, const CFloatHandle& resultHandle, bool reverse )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;
	vectorCumSumAlongDimensionImpl( firstHandle, precedingDimension, dimension, followingDimension, resultHandle, reverse );
}

void CCpuMathEngine::VectorCumSumAlongDimension( const CConstIntHandle& firstHandle, int precedingDimension, int dimension,
	int followingDimension, const CIntHandle& resultHandle, bool reverse )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;
	vectorCumSumAlongDimensionImpl( firstHandle, precedingDimension, dimension, followingDimension, resultHandle, reverse );
}

void CCpuMathEngine::VectorSumAlongDimensionDiag( const CConstFloatHandle& firstHandle, int precedingDimension, int dimension,
	int followingDimension, const CFloatHandle& resultHandle )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	VectorFill( resultHandle, 0.0, precedingDimension * precedingDimension * dimension
		* followingDimension * followingDimension );

	const int width = precedingDimension * dimension * followingDimension;
	const float* first = GetRaw( firstHandle );
	float* result = GetRaw( resultHandle );

	for( int i = 0; i < followingDimension; i++ ) {
		for( int j = 0; j < precedingDimension; j++ ) {
			float* resultRow = result + j;
			for( int k = 0; k < dimension; k++ ) {
				*resultRow = first[k * precedingDimension + j];
				resultRow += precedingDimension;
			}
			result += width;
		}
		first += dimension * precedingDimension;
		result += dimension * precedingDimension;
	}
}

void CCpuMathEngine::VectorCumSumAlongDimensionDiag( const CConstFloatHandle& firstHandle, int precedingDimension, int dimension,
	int followingDimension, const CFloatHandle& resultHandle )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	VectorFill( resultHandle, 0.0, precedingDimension * precedingDimension * dimension
		* dimension * followingDimension * followingDimension );

	const int width = precedingDimension * dimension * followingDimension;
	const float* first = GetRaw( firstHandle );
	float* result = GetRaw( resultHandle );

	for( int i = 0; i < followingDimension; i++ ) {
		for( int j = 0; j < precedingDimension; j++ ) {
			for( int d = 0; d < dimension; d++ ) {
				float* resultRow = result + j;
				for( int k = 0; k <= d; k++ ) {
					*resultRow = first[k * precedingDimension + j];
					resultRow += precedingDimension;
				}
				result += width;
			}
		}
		first += dimension * precedingDimension;
		result += dimension * precedingDimension;
	}
}

void CCpuMathEngine::VectorFillBernoulli( const CFloatHandle& result, float p, int vectorSize, float value, int seed )
{
	ASSERT_EXPR( result.GetMathEngine() == this );
	CCpuExecutionScope scope;
	
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
	CCpuExecutionScope scope;

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
	CCpuExecutionScope scope;

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
	CCpuExecutionScope scope;

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
	CCpuExecutionScope scope;

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
	CCpuExecutionScope scope;

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
	CCpuExecutionScope scope;

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
	CCpuExecutionScope scope;

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
		dataCopy( resultGrad, sourceGrad + pos, sourceGradWidth );

		resultGrad += sourceGradWidth;
	}
}

void CCpuMathEngine::VectorEltwiseMultiply(const CConstIntHandle& firstHandle,
	const CConstIntHandle& secondHandle, const CIntHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;
	
	const int* first = GetRaw( firstHandle );
	const int* second = GetRaw( secondHandle );
	int* result = GetRaw( resultHandle );

	NeoML::vectorEltwiseMultiply( first, second, result, vectorSize );
}

void CCpuMathEngine::VectorMultiply(const CConstFloatHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& multiplierHandle)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( multiplierHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	float multiplier = *GetRaw(multiplierHandle);

	const int curThreadCount = IsOmpRelevant( vectorSize, vectorSize ) ? threadCount : 1;

	if( curThreadCount > 1 ) {
		NEOML_OMP_NUM_THREADS( curThreadCount ) {
			int index, count;
			if( OmpGetTaskIndexAndCount( vectorSize, 16, index, count ) ) {
				vectorMultiply( GetRaw( firstHandle + index ), GetRaw( resultHandle + index ), multiplier, count );
			}
		}
	} else {
		vectorMultiply( GetRaw( firstHandle ), GetRaw( resultHandle ), multiplier, vectorSize );
	}
}

void CCpuMathEngine::VectorMultiply(const CConstIntHandle& firstHandle,
	const CIntHandle& resultHandle, int vectorSize, const CConstIntHandle& multiplierHandle)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( multiplierHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	int multiplier = *GetRaw(multiplierHandle);

	const int curThreadCount = IsOmpRelevant( vectorSize, vectorSize ) ? threadCount : 1;

	if( curThreadCount > 1 ) {
		NEOML_OMP_NUM_THREADS( curThreadCount ) {
			int index, count;
			if( OmpGetTaskIndexAndCount( vectorSize, 16, index, count ) ) {
				vectorMultiply( GetRaw( firstHandle + index ), GetRaw( resultHandle + index ), multiplier, count );
			}
		}
	} else {
		vectorMultiply( GetRaw( firstHandle ), GetRaw( resultHandle ), multiplier, vectorSize );
	}
}

void CCpuMathEngine::VectorEltwiseMultiply(const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const int curThreadCount = IsOmpRelevant( vectorSize, vectorSize ) ? threadCount : 1;

	if( curThreadCount > 1 ) {
		NEOML_OMP_NUM_THREADS( curThreadCount ) {
			int index, count;
			if( OmpGetTaskIndexAndCount( vectorSize, 16, index, count ) ) {
				NeoML::vectorEltwiseMultiply( GetRaw( firstHandle + index ), GetRaw( secondHandle + index ), GetRaw( resultHandle + index ), count );
			}
		}
	} else {
		NeoML::vectorEltwiseMultiply( GetRaw( firstHandle ), GetRaw( secondHandle ), GetRaw( resultHandle ), vectorSize );
	}
}

void CCpuMathEngine::VectorEltwiseMultiplyAdd( const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const int curThreadCount = IsOmpRelevant( vectorSize, vectorSize ) ? threadCount : 1;

	NEOML_OMP_NUM_THREADS( curThreadCount )
	{
		int index;
		int count;
		if( OmpGetTaskIndexAndCount( vectorSize, 16, index, count ) ) {
			NeoML::vectorEltwiseMultiplyAdd( GetRaw( firstHandle + index ), GetRaw( secondHandle + index ), GetRaw( resultHandle + index ), count );
		}
	}
}

void CCpuMathEngine::VectorAbsDiff(const CConstFloatHandle& sourceGradHandle, int gradHeight, int gradWidth,
	const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle)
{
	ASSERT_EXPR( sourceGradHandle.GetMathEngine() == this );
	ASSERT_EXPR( gradHeight > 0 );
	ASSERT_EXPR( gradWidth > 0 );
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

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
	CCpuExecutionScope scope;

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
	CCpuExecutionScope scope;

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
	CCpuExecutionScope scope;

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
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	float* result = GetRaw(resultHandle);
	for(int i = 0; i < vectorSize; ++i) {
		*result++ = -*first++;
	}
}

void CCpuMathEngine::VectorMinMax(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize,
	const CConstFloatHandle& minHandle, const CConstFloatHandle& maxHandle)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( minHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float minValue = *GetRaw(minHandle);
	const float maxValue = *GetRaw(maxHandle);

	const int curThreadCount = IsOmpRelevant( vectorSize, vectorSize ) ? threadCount : 1;

	if( curThreadCount > 1 ) {
		NEOML_OMP_NUM_THREADS( curThreadCount ) {
			int index, count;
			if( OmpGetTaskIndexAndCount( vectorSize, 16, index, count ) ) {
				vectorMinMax( GetRaw(firstHandle + index), GetRaw(resultHandle + index), minValue, maxValue, count );
			}
		}
	} else {
		vectorMinMax( GetRaw(firstHandle ), GetRaw(resultHandle ), minValue, maxValue, vectorSize );
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
	CCpuExecutionScope scope;

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

template<class TSrc, class TDst>
static void vectorEltwiseLessImpl( const CTypedMemoryHandle<const TSrc>& firstHandle,
	const CTypedMemoryHandle<const TSrc>& secondHandle, const CTypedMemoryHandle<TDst>& resultHandle, int vectorSize )
{
	const TSrc* first = GetRaw( firstHandle );
	const TSrc* second = GetRaw( secondHandle );
	TDst* result = GetRaw( resultHandle );

	for( int i = 0; i < vectorSize; ++i ) {
		*result++ = static_cast<TDst>( *first++ < *second++ ? 1 : 0 );
	}
}

void CCpuMathEngine::VectorEltwiseLess( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	vectorEltwiseLessImpl( firstHandle, secondHandle, resultHandle, vectorSize );
}

void CCpuMathEngine::VectorEltwiseLess( const CConstFloatHandle& firstHandle, float second,
	const CFloatHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw( firstHandle );
	float* result = GetRaw( resultHandle );

	for( int i = 0; i < vectorSize; ++i ) {
		*result++ = *first++ < second ? 1.f : 0.f;
	}
}

void CCpuMathEngine::VectorEltwiseLess( float first, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* second = GetRaw( secondHandle );
	float* result = GetRaw( resultHandle );

	for( int i = 0; i < vectorSize; ++i ) {
		*result++ = first < *second++ ? 1.f : 0.f;
	}
}

void CCpuMathEngine::VectorEltwiseLess( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CIntHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	vectorEltwiseLessImpl( firstHandle, secondHandle, resultHandle, vectorSize );
}

void CCpuMathEngine::VectorEltwiseLess( const CConstIntHandle& firstHandle, const CConstIntHandle& secondHandle,
	const CIntHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;
	vectorEltwiseLessImpl( firstHandle, secondHandle, resultHandle, vectorSize );
}

void CCpuMathEngine::VectorEltwiseEqual( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CIntHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;
	const int curThreadCount = IsOmpRelevant( vectorSize, vectorSize ) ? threadCount : 1;
	COmpBinaryVectorFunction<CEqualFunctor<float>> ompFunction( GetRaw( firstHandle ),
		GetRaw( secondHandle ), GetRaw( resultHandle ) );
	applyOmpVectorFunction( curThreadCount, vectorSize, ompFunction );
}

void CCpuMathEngine::VectorEltwiseEqual( const CConstIntHandle& firstHandle, const CConstIntHandle& secondHandle,
	const CIntHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;
	const int curThreadCount = IsOmpRelevant( vectorSize, vectorSize ) ? threadCount : 1;
	COmpBinaryVectorFunction<CEqualFunctor<int>> ompFunction( GetRaw( firstHandle ),
		GetRaw( secondHandle ), GetRaw( resultHandle ) );
	applyOmpVectorFunction( curThreadCount, vectorSize, ompFunction );
}

void CCpuMathEngine::VectorEltwiseWhere( const CConstIntHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CConstFloatHandle& thirdHandle, const CFloatHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( thirdHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;
	const int curThreadCount = IsOmpRelevant( vectorSize, vectorSize ) ? threadCount : 1;
	COmpTernaryVectorFunction<CWhereFunctor<float>> ompFunction( GetRaw( firstHandle ),
		GetRaw( secondHandle ), GetRaw( thirdHandle ), GetRaw( resultHandle ) );
	applyOmpVectorFunction( curThreadCount, vectorSize, ompFunction );
}

void CCpuMathEngine::VectorEltwiseWhere( const CConstIntHandle& firstHandle, const CConstIntHandle& secondHandle,
	const CConstIntHandle& thirdHandle, const CIntHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( thirdHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;
	const int curThreadCount = IsOmpRelevant( vectorSize, vectorSize ) ? threadCount : 1;
	COmpTernaryVectorFunction<CWhereFunctor<int>> ompFunction( GetRaw( firstHandle ),
		GetRaw( secondHandle ), GetRaw( thirdHandle ), GetRaw( resultHandle ) );
	applyOmpVectorFunction( curThreadCount, vectorSize, ompFunction );
}

void CCpuMathEngine::VectorEltwiseDivide(const CConstIntHandle& firstHandle,
	const CConstIntHandle& secondHandle, const CIntHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const int* first = GetRaw(firstHandle);
	const int* second = GetRaw(secondHandle);
	int* result = GetRaw(resultHandle);
	for(int i = 0; i < vectorSize; ++i) {
		*result++ = ( *first++ ) / ( *second++ );
	}
}

void CCpuMathEngine::VectorErf( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw( firstHandle );
	float* result = GetRaw( resultHandle );
	for( int i = 0; i < vectorSize; ++i ) {
		*result++ = std::erff( *first++ );
	}
}

} // namespace NeoML
