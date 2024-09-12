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

#include <CpuMathEngine.h>
#include <CpuExecutionScope.h>
#include <MemoryHandleInternal.h>
#include <MathEngineCommon.h>
#include <CpuRandom.h>
#include <CpuMathEnginePrivate.h>
#include <cmath>
#include <functional>

namespace NeoML {

void CCpuMathEngine::VectorFill( const CFloatHandle& result, int vectorSize, const CConstFloatHandle& value )
{
	ASSERT_EXPR( result.GetMathEngine() == this );
	ASSERT_EXPR( value.GetMathEngine() == this );
	CCpuExecutionScope scope;

	VectorFill( result, *GetRaw( value ), vectorSize );
}

void CCpuMathEngine::VectorFill( const CIntHandle& result, int vectorSize, const CConstIntHandle& value )
{
	ASSERT_EXPR( result.GetMathEngine() == this );
	ASSERT_EXPR( value.GetMathEngine() == this );
	CCpuExecutionScope scope;

	VectorFill( result, *GetRaw( value ), vectorSize );
}

void CCpuMathEngine::VectorCopy( const CFloatHandle& firstHandle, const CConstFloatHandle& secondHandle, int vectorSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	dataCopy( GetRaw( firstHandle ), GetRaw( secondHandle ), vectorSize );
}

void CCpuMathEngine::VectorCopy( const CIntHandle& firstHandle, const CConstIntHandle& secondHandle, int vectorSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	dataCopy( GetRaw( firstHandle ), GetRaw( secondHandle ), vectorSize );
}

//------------------------------------------------------------------------------------------------------------

template<class T>
void broadcastCopyImpl( T* to, const T* from, const CBlobDesc& toDesc, const CBlobDesc& fromDesc, int additionalWidth )
{
	int curSize = fromDesc.BlobSize() * additionalWidth;
	int copySize = additionalWidth;
	dataCopy( to, from, curSize );

	for( int i = BD_Count - 1; i >= 0; i-- ) {
		if( toDesc.DimSize( i ) != fromDesc.DimSize( i ) ) {
			const T* fromPtr = to + curSize - copySize;
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

void CCpuMathEngine::BroadcastCopy( const CIntHandle& toHandle, const CConstIntHandle& fromHandle,
	const CBlobDesc& toDesc, const CBlobDesc& fromDesc, int additionalWidth )
{
	ASSERT_EXPR( toHandle.GetMathEngine() == this );
	ASSERT_EXPR( fromHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;
	for( int i = 0; i < BD_Count; i++ ) {
		ASSERT_EXPR( fromDesc.DimSize( i ) == 1 || fromDesc.DimSize( i ) == toDesc.DimSize( i ) );
	}

	broadcastCopyImpl( GetRaw( toHandle ), GetRaw( fromHandle ), toDesc, fromDesc, additionalWidth );
}

void CCpuMathEngine::BroadcastCopy( const CFloatHandle& toHandle, const CConstFloatHandle& fromHandle,
	const CBlobDesc& toDesc, const CBlobDesc& fromDesc, int additionalWidth )
{
	ASSERT_EXPR( toHandle.GetMathEngine() == this );
	ASSERT_EXPR( fromHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;
	for( int i = 0; i < BD_Count; i++ ) {
		ASSERT_EXPR( fromDesc.DimSize( i ) == 1 || fromDesc.DimSize( i ) == toDesc.DimSize( i ) );
	}

	broadcastCopyImpl( GetRaw( toHandle ), GetRaw( fromHandle ), toDesc, fromDesc, additionalWidth );
}

//------------------------------------------------------------------------------------------------------------

void CCpuMathEngine::VectorAdd( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	NeoML::vectorAdd( GetRaw( firstHandle ), GetRaw( secondHandle ), GetRaw( resultHandle ), vectorSize );
}

void CCpuMathEngine::VectorSum( const CConstFloatHandle& firstHandle, int vectorSize, const CFloatHandle& resultHandle )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	*GetRaw( resultHandle ) = 0;
	VectorSumAdd( firstHandle, vectorSize, resultHandle );
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
	const unsigned int threshold = (unsigned int)( (double)p * UINT_MAX );

	CCpuRandom random( seed );
	CCpuRandom::CCounter generated{};

	int index = 0;
	for( int i = 0; i < ( vectorSize + 3 ) / 4; ++i ) {
		random.Next( generated );
		for( int j = 0; j < 4 && index < vectorSize; ++j ) {
			resultPtr[index++] = ( generated.Data[j] <= threshold ) ? value : 0.f;
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

void CCpuMathEngine::VectorAddValue( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle,
	int vectorSize, const CConstFloatHandle& addition )
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

void CCpuMathEngine::VectorDotProduct( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	int vectorSize, const CFloatHandle& resultHandle )
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

void CCpuMathEngine::VectorTopK( const CConstFloatHandle& firstHandle, int firstSize, int k, const CFloatHandle& resultHandle,
	const CIntHandle& indicesHandle )
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
				for( int j = std::min( size + 1, k ) - 1; j >= pos + 1; j-- ) {
					result[j] = result[j - 1];
					indices[j] = indices[j - 1];
				}
				break;
			}
		}
		if( pos < k ) {
			result[pos] = *first;
			indices[pos] = i;
			size = std::min( size + 1, k );
		}
		first++;
	}
}

void CCpuMathEngine::VectorTopKDiff( const CConstFloatHandle& sourceGradHandle, int sourceGradHeight, int sourceGradWidth,
	const CConstIntHandle& indicesHandle, int k, const CFloatHandle& resultGradHandle )
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
		const int pos = indices[i] * sourceGradWidth;
		dataCopy( resultGrad, sourceGrad + pos, sourceGradWidth );

		resultGrad += sourceGradWidth;
	}
}

void CCpuMathEngine::VectorEltwiseMultiply( const CConstIntHandle& firstHandle,
	const CConstIntHandle& secondHandle, const CIntHandle& resultHandle, int vectorSize )
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

void CCpuMathEngine::VectorMultiply( const CConstFloatHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& multiplierHandle )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( multiplierHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float multiplier = *GetRaw( multiplierHandle );

	vectorMultiply( GetRaw( firstHandle ), GetRaw( resultHandle ), vectorSize, multiplier );
}

void CCpuMathEngine::VectorMultiply( const CConstIntHandle& firstHandle,
	const CIntHandle& resultHandle, int vectorSize, const CConstIntHandle& multiplierHandle )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( multiplierHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const int multiplier = *GetRaw( multiplierHandle );

	vectorMultiply( GetRaw( firstHandle ), GetRaw( resultHandle ), vectorSize, multiplier );
}

void CCpuMathEngine::VectorEltwiseMultiply( const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;
	
	NeoML::vectorEltwiseMultiply( GetRaw( firstHandle ), GetRaw( secondHandle ), GetRaw( resultHandle ), vectorSize );
}

void CCpuMathEngine::VectorEltwiseMultiplyAdd( const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	NeoML::vectorEltwiseMultiplyAdd( GetRaw( firstHandle ), GetRaw( secondHandle ), GetRaw( resultHandle ), vectorSize );
}

void CCpuMathEngine::VectorAbsDiff( const CConstFloatHandle& sourceGradHandle, int gradHeight, int gradWidth,
	const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle )
{
	ASSERT_EXPR( sourceGradHandle.GetMathEngine() == this );
	ASSERT_EXPR( gradHeight > 0 );
	ASSERT_EXPR( gradWidth > 0 );
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw( firstHandle );
	const float* grad = GetRaw( sourceGradHandle );
	float* result = GetRaw( resultHandle );

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
	const int gradSize = gradHeight == 1 ? 1 : gradWidth;

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

	const float* sourceGrad = GetRaw( sourceGradHandle );
	const float* value = GetRaw( valueHandle );
	float* result = GetRaw( resultHandle );

	const int valueSize = sourceGradHeight == 1 ? sourceGradWidth : sourceGradHeight;
	const int gradSize = sourceGradHeight == 1 ? 1 : sourceGradWidth;
	for( int i = 0; i < valueSize; ++i ) {
		float div = *value++;
		if( ( -FLT_MIN <= div && div < 0 ) || ( 0 <= div && div <= FLT_MIN ) ) {
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

void CCpuMathEngine::VectorNeg( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw( firstHandle );
	float* result = GetRaw( resultHandle );
	for( int i = 0; i < vectorSize; ++i ) {
		*result++ = -*first++;
	}
}

void CCpuMathEngine::VectorMinMax( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize,
	const CConstFloatHandle& minHandle, const CConstFloatHandle& maxHandle )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( minHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float minValue = *GetRaw( minHandle );
	const float maxValue = *GetRaw( maxHandle );
	
	vectorMinMax( GetRaw( firstHandle ), GetRaw( resultHandle ), minValue, maxValue, vectorSize );
}

void CCpuMathEngine::VectorMinMaxDiff( const CConstFloatHandle& sourceGradHandle, int gradHeight, int gradWidth,
	const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle,
	const CConstFloatHandle& minHandle, const CConstFloatHandle& maxHandle )
{
	ASSERT_EXPR( sourceGradHandle.GetMathEngine() == this );
	ASSERT_EXPR( gradHeight > 0 );
	ASSERT_EXPR( gradWidth > 0 );
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( minHandle.GetMathEngine() == this );
	ASSERT_EXPR( maxHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw( firstHandle );
	const float* sourceGrad = GetRaw( sourceGradHandle );
	float* result = GetRaw( resultHandle );
	const float minValue = *GetRaw( minHandle );
	const float maxValue = *GetRaw( maxHandle );

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

	CBinaryVectorFunction<CEqualFunctor<float>>{}(
		GetRaw( firstHandle ), GetRaw( secondHandle ), GetRaw( resultHandle ), vectorSize );
}

void CCpuMathEngine::VectorEltwiseEqual( const CConstIntHandle& firstHandle, const CConstIntHandle& secondHandle,
	const CIntHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	CBinaryVectorFunction<CEqualFunctor<int>>{}(
		GetRaw( firstHandle ), GetRaw( secondHandle ), GetRaw( resultHandle ), vectorSize );
}

void CCpuMathEngine::VectorEltwiseWhere( const CConstIntHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CConstFloatHandle& thirdHandle, const CFloatHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( thirdHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	CTernaryVectorFunction<CWhereFunctor<float>>{}(
		GetRaw( firstHandle ), GetRaw( secondHandle ), GetRaw( thirdHandle ), GetRaw( resultHandle ), vectorSize );
}

void CCpuMathEngine::VectorEltwiseWhere( const CConstIntHandle& firstHandle, const CConstIntHandle& secondHandle,
	const CConstIntHandle& thirdHandle, const CIntHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( thirdHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;
	
	CTernaryVectorFunction<CWhereFunctor<int>>{}(
		GetRaw( firstHandle ), GetRaw( secondHandle ), GetRaw( thirdHandle ), GetRaw( resultHandle ), vectorSize );
}

void CCpuMathEngine::VectorEltwiseDivide( const CConstIntHandle& firstHandle,
	const CConstIntHandle& secondHandle, const CIntHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const int* first = GetRaw( firstHandle );
	const int* second = GetRaw( secondHandle );
	int* result = GetRaw( resultHandle );
	for( int i = 0; i < vectorSize; ++i ) {
		*result++ = ( *first++ ) / ( *second++ );
	}
}

void CCpuMathEngine::VectorHSwish( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle,
	int vectorSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	vectorHSwish( GetRaw( firstHandle ), GetRaw( resultHandle ), vectorSize );
}

void CCpuMathEngine::VectorHardSigmoid( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize,
	const CConstFloatHandle& slopeHandle, const CConstFloatHandle& biasHandle )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	float* result = GetRaw(resultHandle);

	const float slope = *GetRaw( slopeHandle );
	const float bias = *GetRaw( biasHandle );

	ASSERT_EXPR( slope != 0.f );

	vectorHardSigmoid( first, result, slope, bias, vectorSize );
}

void CCpuMathEngine::VectorLeakyReLU( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle,
	int vectorSize, const CConstFloatHandle& alphaHandle )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( alphaHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;
	vectorLeakyReLU( GetRaw( firstHandle ), GetRaw( resultHandle ), *GetRaw( alphaHandle ), vectorSize );
}

void CCpuMathEngine::VectorELU( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle,
	int vectorSize, const CConstFloatHandle& alphaHandle )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( alphaHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;
	vectorELU( GetRaw( firstHandle ), GetRaw( resultHandle ), *GetRaw( alphaHandle ), vectorSize );
}

void CCpuMathEngine::VectorHardTanh(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;
	vectorMinMax( GetRaw( firstHandle ), GetRaw( resultHandle ), -1.f, 1.f, vectorSize );
}

} // namespace NeoML
