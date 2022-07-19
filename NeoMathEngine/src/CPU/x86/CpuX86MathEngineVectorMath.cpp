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
#include <CpuExecutionScope.h>
#include <CpuX86.h>
#include <float.h>
#include <MemoryHandleInternal.h>
#include <MathEngineCommon.h>
#include <CpuX86MathEngineVectorMathPrivate.h>

namespace NeoML {

void CCpuMathEngine::VectorFill( const CFloatHandle& result, float value, int vectorSize )
{
	ASSERT_EXPR( result.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const int curThreadCount = IsOmpRelevant( vectorSize, vectorSize ) ? threadCount : 1;

	if( curThreadCount > 1 ) {
		NEOML_OMP_NUM_THREADS( curThreadCount ) {
			int index, count;
			if( OmpGetTaskIndexAndCount( vectorSize, 16, index, count ) ) {
				vectorFill( GetRaw( result + index ), value, count );
			}
		}
	} else {
		vectorFill( GetRaw( result ), value, vectorSize );
	}
}

void CCpuMathEngine::VectorFill( const CIntHandle& resultHandle, int value, int vectorSize )
{
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const int curThreadCount = IsOmpRelevant( vectorSize, vectorSize ) ? threadCount : 1;

	if( curThreadCount > 1 ) {
		NEOML_OMP_NUM_THREADS( curThreadCount ) {
			int index, count;
			if( OmpGetTaskIndexAndCount( vectorSize, 16, index, count ) ) {
				vectorFill( GetRaw( resultHandle + index ), value, count );
			}
		}
	} else {
		vectorFill( GetRaw( resultHandle ), value, vectorSize );
	}
}

void CCpuMathEngine::VectorConvert( const CConstFloatHandle& from, const CIntHandle& to, int vectorSize )
{
	ASSERT_EXPR( from.GetMathEngine() == this );
	ASSERT_EXPR( to.GetMathEngine() == this );
	ASSERT_EXPR( vectorSize >= 0 );
	CCpuExecutionScope scope;

	const float* fromPtr = GetRaw( from );
	int* toPtr = GetRaw( to );

	int sseSize;
	int nonSseSize;
	checkSse( vectorSize, sseSize, nonSseSize );

	for( int i = 0; i < sseSize; ++i ) {
		StoreIntSse4( _mm_cvttps_epi32( LoadSse4( fromPtr ) ), toPtr );
		toPtr += 4;
		fromPtr += 4;
	}

	if( nonSseSize > 0 ) {
		StoreIntSse( _mm_cvttps_epi32( LoadSse( fromPtr, nonSseSize ) ), toPtr, nonSseSize );
	}
}

void CCpuMathEngine::VectorConvert( const CConstIntHandle& from, const CFloatHandle& to, int vectorSize )
{
	ASSERT_EXPR( from.GetMathEngine() == this );
	ASSERT_EXPR( to.GetMathEngine() == this );
	ASSERT_EXPR( vectorSize >= 0 );
	CCpuExecutionScope scope;

	const int* fromPtr = GetRaw( from );
	float* toPtr = GetRaw( to );

	int sseSize;
	int nonSseSize;
	checkSse( vectorSize, sseSize, nonSseSize );

	for( int i = 0; i < sseSize; ++i ) {
		StoreSse4( _mm_cvtepi32_ps( LoadIntSse4( fromPtr ) ), toPtr );
		toPtr += 4;
		fromPtr += 4;
	}

	if( nonSseSize > 0 ) {
		StoreSse( _mm_cvtepi32_ps( LoadIntSse( fromPtr, nonSseSize ) ), toPtr, nonSseSize );
	}
}

void CCpuMathEngine::FilterSmallValues( const CFloatHandle& data, int dataSize, float threshold )
{
	ASSERT_EXPR( data.GetMathEngine() == this );
	ASSERT_EXPR( dataSize >= 0 );
	ASSERT_EXPR( threshold > 0.f );
	CCpuExecutionScope scope;

	float* dataPtr = GetRaw( data );

	int sseSize;
	int nonSseSize;
	checkSse( dataSize, sseSize, nonSseSize );

	if( sseSize > 0 ) {
		const __m128 thresholdSse = _mm_set1_ps( threshold );
		const __m128 negThresholdSse = _mm_set1_ps( -threshold );
		for( int i = 0; i < sseSize; ++i ) {
			__m128 input = _mm_loadu_ps( dataPtr );
			// f(x) = x if (x >= threshold OR x <= -threshold) else 0.
			_mm_storeu_ps( dataPtr, _mm_and_ps(
				_mm_or_ps( _mm_cmpge_ps( input, thresholdSse ), _mm_cmple_ps( input, negThresholdSse ) ),
				input ) );
			dataPtr += 4;
		}
	}

	for( int i = 0; i < nonSseSize; ++i ) {
		*dataPtr = ( *dataPtr < threshold && *dataPtr > -threshold ) ? 0.f : *dataPtr;
		*dataPtr++;
	}
}

void CCpuMathEngine::VectorSumAdd( const CConstFloatHandle& firstHandle, int vectorSize, const CFloatHandle& resultHandle )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw( firstHandle );

	int sseSize;
	int nonSseSize;
	checkSse( vectorSize, sseSize, nonSseSize );

	float* result = GetRaw( resultHandle );

	if( sseSize > 0 ) {
		__m128 sum = _mm_loadu_ps( first );
		--sseSize;
		first += 4;
		for( int i = 0; i < sseSize; ++i ) {
			sum = _mm_add_ps( sum, _mm_loadu_ps( first ) );
			first += 4;
		}
		__m128 tmp = _mm_shuffle_ps( sum, sum, _MM_SHUFFLE( 0, 3, 2, 1 ) );
		sum = _mm_add_ps( sum, tmp );
		tmp = _mm_shuffle_ps( sum, sum, _MM_SHUFFLE( 1, 0, 3, 2 ) );
		sum = _mm_add_ss( sum, tmp );

		*result += _mm_cvtss_f32( sum );
	}

	for( int i = 0; i < nonSseSize; ++i ) {
		*result += *first++;
	}
}

void CCpuMathEngine::VectorEqual( const CConstIntHandle& firstHandle, const CConstIntHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const int* first = GetRaw( firstHandle );
	const int* second = GetRaw( secondHandle );
	float* result = GetRaw( resultHandle );

	int sseSize;
	int nonSseSize;
	checkSse( vectorSize, sseSize, nonSseSize );

	if( sseSize > 0 ) {
		const __m128 oneSse = _mm_set1_ps( 1. );
		for( int i = 0; i < sseSize; ++i ) {
			const __m128i intMask = _mm_cmpeq_epi32( LoadIntSse4( first ), LoadIntSse4( second ) );
			const __m128 mask = _mm_castsi128_ps( intMask );
			_mm_storeu_ps( result, _mm_and_ps( oneSse, mask ) );
			first += 4;
			second += 4;
			result += 4;
		}
	}

	for( int i = 0; i < nonSseSize; ++i ) {
		*result++ = ( *first++ == *second++ ) ? 1.0f : 0.0f;
	}
}

void CCpuMathEngine::VectorEqualValue( const CConstIntHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstIntHandle& valueHandle )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( valueHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const int* first = GetRaw( firstHandle );
	float* result = GetRaw( resultHandle );
	const int value = *GetRaw( valueHandle );

	int sseSize;
	int nonSseSize;
	checkSse( vectorSize, sseSize, nonSseSize );

	if( sseSize > 0 ) {
		const __m128 oneSse = _mm_set1_ps( 1. );
		const __m128i valueSse = _mm_set1_epi32( value );
		for( int i = 0; i < sseSize; ++i ) {
			const __m128i intMask = _mm_cmpeq_epi32( LoadIntSse4( first ), valueSse );
			const __m128 mask = _mm_castsi128_ps( intMask );
			_mm_storeu_ps( result, _mm_and_ps( oneSse, mask ) );
			first += 4;
			result += 4;
		}
	}

	for( int i = 0; i < nonSseSize; ++i ) {
		*result++ = ( *first++ == value ) ? 1.0f : 0.0f;
	}
}

void CCpuMathEngine::VectorELU( const CConstFloatHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& alphaHandle )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( alphaHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw( firstHandle );
	float* result = GetRaw( resultHandle );
	const float alpha = *GetRaw( alphaHandle );

	for( int i = 0; i < vectorSize; ++i ) {
		*result = *first >= 0 ? *first : alpha * ( ExponentFunc( *first ) - 1.f );
		++result;
		++first;
	}
}

void CCpuMathEngine::VectorELUDiff( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& alphaHandle )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( alphaHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	VectorExp( firstHandle, resultHandle, vectorSize );

	const float* second = GetRaw( secondHandle );
	float* result = GetRaw( resultHandle );
	const float alpha = *GetRaw( alphaHandle );

	int sseSize;
	int nonSseSize;
	checkSse( vectorSize, sseSize, nonSseSize );

	if( sseSize > 0 ) {
		const __m128 oneSse = _mm_set1_ps( 1. );
		const __m128 alphaSse = _mm_set_ps1( alpha );
		for( int i = 0; i < sseSize; ++i ) {
			const __m128 expSse = _mm_loadu_ps( result );
			const __m128 nonNegSse = _mm_cmpge_ps( expSse, oneSse );
			_mm_storeu_ps( result, _mm_mul_ps( _mm_loadu_ps( second ), _mm_add_ps(
				// *first >= 0  -->  *second * 1
				_mm_and_ps( nonNegSse, oneSse ),
				// *first < 0  -->  *second * exp( *first ) * alpha
				_mm_andnot_ps( nonNegSse, _mm_mul_ps( expSse, alphaSse ) )
			) ) );
			second += 4;
			result += 4;
		}
	}

	for( int i = 0; i < nonSseSize; ++i ) {
		*result = *result >= 1 ? *second : *second * *result * alpha;
		++second;
		++result;
	}
}

void CCpuMathEngine::VectorELUDiffOp( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& alphaHandle )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( alphaHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw( firstHandle );
	const float* second = GetRaw( secondHandle );
	float* result = GetRaw( resultHandle );
	const float alpha = *GetRaw( alphaHandle );

	int sseSize;
	int nonSseSize;
	checkSse( vectorSize, sseSize, nonSseSize );

	if( sseSize > 0 ) {
		const __m128 zeroSse = _mm_setzero_ps();
		const __m128 oneSse = _mm_set1_ps( 1. );
		const __m128 alphaSse = _mm_set_ps1( alpha );
		for( int i = 0; i < sseSize; ++i ) {
			const __m128 firstSse = _mm_loadu_ps( first );
			const __m128 nonNegSse = _mm_cmpge_ps( firstSse, zeroSse );
			_mm_storeu_ps( result, _mm_mul_ps( _mm_loadu_ps( second ), _mm_add_ps(
				// *first >= 0  -->  *second * 1
				_mm_and_ps( nonNegSse, oneSse ),
				// *first < 0  -->  *second * ( first + alpha )
				_mm_andnot_ps( nonNegSse, _mm_add_ps( firstSse, alphaSse ) )
			) ) );
			first += 4;
			second += 4;
			result += 4;
		}
	}

	for( int i = 0; i < nonSseSize; ++i ) {
		*result = *first >= 0 ? *second : *second * ( *first + alpha );
		++first;
		++second;
		++result;
	}
}

void CCpuMathEngine::VectorReLU( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle,
	int vectorSize, const CConstFloatHandle& upperThresholdHandle )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( upperThresholdHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw( firstHandle );
	float* result = GetRaw( resultHandle );
	float threshold = *GetRaw( upperThresholdHandle );

	const int curThreadCount = IsOmpRelevant( vectorSize, vectorSize ) ? threadCount : 1;

	if( curThreadCount > 1 ) {
		NEOML_OMP_NUM_THREADS( curThreadCount ) {
			int index, count;
			if( OmpGetTaskIndexAndCount( vectorSize, 16, index, count ) ) {
				if( threshold > 0 ) {
					vectorReLU( first + index, result + index, count, threshold );
				} else {
					vectorReLU( first + index, result + index, count );
				}
			}
		}
	} else {
		if( threshold > 0 ) {
			vectorReLU( first, result, vectorSize, threshold );
		} else {
			vectorReLU( first, result, vectorSize );
		}
	}
}

void CCpuMathEngine::VectorReLUDiff( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& upperThresholdHandle )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( upperThresholdHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw( firstHandle );
	const float* second = GetRaw( secondHandle );
	float* result = GetRaw( resultHandle );
	float threshold = *GetRaw( upperThresholdHandle );

	int sseSize;
	int nonSseSize;
	checkSse( vectorSize, sseSize, nonSseSize );

	if( sseSize > 0 ) {
		const __m128 zeroSse = _mm_setzero_ps();
		if( threshold > 0 ) {
			const __m128 thresholdSse = _mm_set_ps1( threshold );
			for( int i = 0; i < sseSize; ++i ) {
				__m128 val = _mm_loadu_ps( first );
				_mm_storeu_ps( result,
					_mm_and_ps( _mm_and_ps( _mm_cmpgt_ps( val, zeroSse ), _mm_cmplt_ps( val, thresholdSse ) ),
						_mm_loadu_ps( second ) ) );
				first += 4;
				second += 4;
				result += 4;
			}
		} else {
			for( int i = 0; i < sseSize; ++i ) {
				_mm_storeu_ps( result, _mm_and_ps( _mm_cmpgt_ps( _mm_loadu_ps( first ), zeroSse ), _mm_loadu_ps( second ) ) );
				first += 4;
				second += 4;
				result += 4;
			}
		}
	}

	if( threshold > 0 ) {
		for( int i = 0; i < nonSseSize; ++i ) {
			*result++ = ( *first > 0 && *first < threshold ) ? *second : 0;
			++first;
			++second;
		}
	} else {
		for( int i = 0; i < nonSseSize; ++i ) {
			*result++ = *first++ > 0 ? *second : 0;
			++second;
		}
	}
}

void CCpuMathEngine::VectorLeakyReLU( const CConstFloatHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& alpha )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( alpha.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw( firstHandle );
	float* result = GetRaw( resultHandle );
	const float coeff = *GetRaw( alpha );

	int sseSize;
	int nonSseSize;
	checkSse( vectorSize, sseSize, nonSseSize );

	if( sseSize > 0 ) {
		const __m128 zeroSse = _mm_setzero_ps();
		const __m128 coeffSse = _mm_set1_ps( coeff );
		for( int i = 0; i < sseSize; ++i ) {
			__m128 input = _mm_loadu_ps( first );
			// result = x_pos + x_neg * alpha
			_mm_storeu_ps( result, _mm_add_ps( _mm_max_ps( input, zeroSse ),
				_mm_mul_ps( _mm_min_ps( input, zeroSse ), coeffSse ) ) );
			first += 4;
			result += 4;
		}
	}

	for( int i = 0; i < nonSseSize; ++i ) {
		*result++ = *first >= 0.f ? *first++ : coeff * *first++;
	}
}

void CCpuMathEngine::VectorLeakyReLUDiff( const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle,
	int vectorSize, const CConstFloatHandle& alpha )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( alpha.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw( firstHandle );
	const float* second = GetRaw( secondHandle );
	const float coeff = *GetRaw( alpha );
	float* result = GetRaw( resultHandle );

	int sseSize;
	int nonSseSize;
	checkSse( vectorSize, sseSize, nonSseSize );

	if( sseSize > 0 ) {
		const __m128 zeroSse = _mm_setzero_ps();
		const __m128 coeffSse = _mm_set1_ps( coeff );
		for( int i = 0; i < sseSize; ++i ) {
			__m128 input = _mm_loadu_ps( first );
			__m128 diff = _mm_loadu_ps( second );
			_mm_storeu_ps( result, _mm_add_ps( _mm_and_ps( _mm_cmpgt_ps( input, zeroSse ), diff ),
				_mm_mul_ps( _mm_and_ps( _mm_cmplt_ps( input, zeroSse ), diff ), coeffSse ) ) );
			first += 4;
			second += 4;
			result += 4;
		}
	}

	for( int i = 0; i < nonSseSize; ++i ) {
		*result++ = *first++ > 0 ? *second : *second * coeff;
		++second;
	}
}

void CCpuMathEngine::VectorHSwish( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw( firstHandle );
	float* result = GetRaw( resultHandle );

	int sseSize;
	int nonSseSize;
	checkSse( vectorSize, sseSize, nonSseSize );

	if( sseSize > 0 ) {
		const __m128 minusThreeSse = _mm_set1_ps( -3.f );
		const __m128 threeSse = _mm_set1_ps( 3.f );
		const __m128 oneSixthSse = _mm_set1_ps( 1.f / 6.f );
		for( int i = 0; i < sseSize; ++i ) {
			__m128 input = _mm_loadu_ps( first );
			__m128 middlePart = _mm_cmplt_ps( minusThreeSse, input );
			middlePart = _mm_and_ps( middlePart, _mm_cmplt_ps( input, threeSse ) ); // mask for (-3; 3)
			middlePart = _mm_and_ps( middlePart, _mm_mul_ps( _mm_mul_ps( input, oneSixthSse ), _mm_add_ps( input, threeSse ) ) );
			__m128 rightPart = _mm_cmpge_ps( input, threeSse );
			rightPart = _mm_and_ps( rightPart, input );
			_mm_storeu_ps( result, _mm_add_ps( middlePart, rightPart ) );

			first += 4;
			result += 4;
		}
	}

	for( int i = 0; i < nonSseSize; ++i ) {
		if( *first <= -3.f ) {
			*result = 0.f;
		} else if( *first >= 3.f ) {
			*result = *first;
		} else {
			*result = *first * ( *first + 3 ) / 6.f;
		}
		++result;
		++first;
	}

}

void CCpuMathEngine::VectorHSwishDiff( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle, 
	const CFloatHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw( firstHandle );
	const float* second = GetRaw( secondHandle );
	float* result = GetRaw( resultHandle );

	int sseSize;
	int nonSseSize;
	checkSse( vectorSize, sseSize, nonSseSize );

	if( sseSize > 0 ) {
		const __m128 oneSse = _mm_set1_ps( 1.f );
		const __m128 minusThreeSse = _mm_set1_ps( -3.f );
		const __m128 threeSse = _mm_set1_ps( 3.f );
		const __m128 oneThirdSse = _mm_set1_ps( 1.f / 3.f );
		const __m128 halfSse = _mm_set1_ps( 0.5f );
		for( int i = 0; i < sseSize; ++i ) {
			__m128 input = _mm_loadu_ps( first );
			__m128 middlePart = _mm_cmplt_ps( minusThreeSse, input );
			middlePart = _mm_and_ps( middlePart, _mm_cmplt_ps( input, threeSse ) ); // mask for (-3; 3)
			middlePart = _mm_and_ps( middlePart, _mm_add_ps( _mm_mul_ps( input, oneThirdSse ), halfSse ) );
			__m128 rightPart = _mm_cmpge_ps( input, threeSse );
			rightPart = _mm_and_ps( rightPart, oneSse );
			_mm_storeu_ps( result, _mm_mul_ps( _mm_loadu_ps( second ), _mm_add_ps( middlePart, rightPart ) ) );

			first += 4;
			result += 4;
			second += 4;
		}
	}

	for( int i = 0; i < nonSseSize; ++i ) {
		if( *first <= -3.f ) {
			*result = 0.f;
		} else if( *first >= 3.f ) {
			*result = *second;
		} else {
			*result = *second * ( 1.f / 3.f * *first + 0.5f );
		}
		++result;
		++first;
		++second;
	}
}

void CCpuMathEngine::VectorEltwiseMax(const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);

	vectorEltwiseMax( first, second, result, vectorSize );
}

void CCpuMathEngine::VectorEltwiseMin(const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);

	int sseSize;
	int nonSseSize;
	checkSse(vectorSize, sseSize, nonSseSize);

	for(int i = 0; i < sseSize; ++i) {
		_mm_storeu_ps(result, _mm_min_ps(_mm_loadu_ps(first), _mm_loadu_ps(second)));
		first += 4;
		second += 4;
		result += 4;
	}

	for(int i = 0; i < nonSseSize; ++i) {
		*result++ = min(*first, *second);
		first++;
		second++;
	}
}

void CCpuMathEngine::VectorAbs(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	float* result = GetRaw(resultHandle);

	int sseSize;
	int nonSseSize;
	checkSse(vectorSize, sseSize, nonSseSize);

	if(sseSize > 0) {
		const __m128 zeroSse = _mm_setzero_ps();
		for(int i = 0; i < sseSize; ++i) {
			__m128 value = _mm_loadu_ps(first);
			__m128 negValue = _mm_sub_ps(zeroSse, value);
			_mm_storeu_ps(result, _mm_max_ps(value, negValue));
			first += 4;
			result += 4;
		}
	}

	for(int i = 0; i < nonSseSize; ++i) {
		*result++ = abs(*first++);
	}
}

void CCpuMathEngine::VectorAbsDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);

	int sseSize;
	int nonSseSize;
	checkSse(vectorSize, sseSize, nonSseSize);

	if(sseSize > 0) {
		const __m128 zeroSse = _mm_setzero_ps();
		for(int i = 0; i < sseSize; ++i) {
			__m128 mask = _mm_cmpgt_ps(_mm_loadu_ps(first), zeroSse);
			__m128 value = _mm_loadu_ps(second);
			_mm_storeu_ps(result, _mm_sub_ps(_mm_and_ps(mask, value), _mm_andnot_ps(mask, value)));
			first += 4;
			second += 4;
			result += 4;
		}
	}

	for(int i = 0; i < nonSseSize; ++i) {
		*result++ = *first++ > 0 ? *second : -*second;
		++second;
	}
}

void CCpuMathEngine::VectorHinge(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	float* result = GetRaw(resultHandle);

	int sseSize;
	int nonSseSize;
	checkSse(vectorSize, sseSize, nonSseSize);

	if(sseSize > 0) {
		const __m128 zeroSse = _mm_setzero_ps();
		const __m128 oneSse = _mm_set_ps1(1);
		for(int i = 0; i < sseSize; ++i) {
			_mm_storeu_ps(result, _mm_max_ps(zeroSse, _mm_sub_ps(oneSse, _mm_loadu_ps(first))));
			first += 4;
			result += 4;
		}
	}

	for(int i = 0; i < nonSseSize; ++i) {
		*result = max(0.f, 1 - *first);
		result++;
		first++;
	}
}

void CCpuMathEngine::VectorHingeDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);

	int sseSize;
	int nonSseSize;
	checkSse(vectorSize, sseSize, nonSseSize);

	if(sseSize > 0) {
		__m128 oneSse = _mm_set_ps1(1);
		__m128 negOneSse = _mm_set_ps1(-1);
		for(int i = 0; i < sseSize; ++i) {
			__m128 temp = _mm_cmplt_ps(_mm_loadu_ps(first), oneSse); // *first < 1
			temp = _mm_and_ps(temp, negOneSse); // *first < 1 ? -1 : 0
			temp = _mm_mul_ps(temp, _mm_loadu_ps(second)); // *first < 1 ? -*second : 0
			_mm_storeu_ps(result, temp);
			first += 4;
			second += 4;
			result += 4;
		}
	}

	for(int i = 0; i < nonSseSize; ++i) {
		*result++ = *first++ < 1 ? -(*second) : 0;
		++second;
	}
}

void CCpuMathEngine::VectorSquaredHinge(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	float* result = GetRaw(resultHandle);

	int sseSize;
	int nonSseSize;
	checkSse(vectorSize, sseSize, nonSseSize);

	if(sseSize > 0) {
		const __m128 zeroSse = _mm_setzero_ps();
		const __m128 oneSse = _mm_set_ps1(1);
		const __m128 twoSse = _mm_set_ps1(2);
		const __m128 neg4Sse = _mm_set_ps1(-4);
		for(int i = 0; i < sseSize; ++i) {
			__m128 value = _mm_loadu_ps(first);
			__m128 tmp = _mm_sub_ps(oneSse, value); // 1 - *first
			__m128 res1 = _mm_cmpgt_ps(tmp, twoSse); // 1 - *first > 2 <=> *first < -1
			__m128 res2 = _mm_andnot_ps(res1, tmp); // 1 - *first while !(*first < -1)
			// *first < -1 branch
			res1 = _mm_and_ps(res1, neg4Sse);	// -4
			res1 = _mm_mul_ps(res1, value);	// -4 * *first
			// !(*first < -1) branch
			res2 = _mm_max_ps(zeroSse, res2);	// max(0.f, 1 - *first)
			res2 = _mm_mul_ps(res2, res2);	// max(0.f, 1 - *first) * max(0.f, 1 - *first)
			_mm_storeu_ps(result, _mm_add_ps(res1, res2));

			first += 4;
			result += 4;
		}
	}

	for(int i = 0; i < nonSseSize; ++i) {
		if(*first < -1) {
			*result++ = -4 * *first;
		} else {
			float tmp = max(0.f, 1 - *first);
			*result++ = tmp * tmp;
		}
		++first;
	}
}

void CCpuMathEngine::VectorHuber(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	float* result = GetRaw(resultHandle);

	int sseSize;
	int nonSseSize;
	checkSse(vectorSize, sseSize, nonSseSize);

	if(sseSize > 0) {
		const __m128 oneSse = _mm_set_ps1(1.f);
		const __m128 negOneSse = _mm_set_ps1(-1.f);
		const __m128 halfSse = _mm_set_ps1(0.5f);
		const __m128 negHalfSse = _mm_set_ps1(-0.5f);
		for(int i = 0; i < sseSize; ++i) {
			__m128 value = _mm_loadu_ps(first);
			__m128 cmpOne = _mm_cmpgt_ps(value, oneSse);
			__m128 cmpNegOne = _mm_cmplt_ps(value, negOneSse);
			// *first > 1
			__m128 resRight = _mm_and_ps(cmpOne, _mm_add_ps(negHalfSse, value));
			// *first < -1
			__m128 resLeft = _mm_and_ps(cmpNegOne, _mm_sub_ps(negHalfSse, value));
			// else
			value = _mm_mul_ps(halfSse, _mm_mul_ps(value, value));
			__m128 resMain = _mm_andnot_ps(_mm_or_ps(cmpOne, cmpNegOne), value);

			_mm_storeu_ps(result, _mm_add_ps(_mm_add_ps(resLeft, resRight), resMain));

			first += 4;
			result += 4;
		}
	}

	for(int i = 0; i < nonSseSize; ++i) {
		if(*first > 1) {
			*result++ = (*first - 0.5f);
		} else if(*first < -1) {
			*result++ = (-*first - 0.5f);
		} else {
			*result++ = *first * *first / 2;
		}
		++first;
	}
}

void CCpuMathEngine::VectorHuberDerivative(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	float* result = GetRaw(resultHandle);

	int sseSize;
	int nonSseSize;
	checkSse(vectorSize, sseSize, nonSseSize);

	if(sseSize > 0) {
		const __m128 oneSse = _mm_set_ps1(1.f);
		const __m128 negOneSse = _mm_set_ps1(-1.f);
		for(int i = 0; i < sseSize; ++i) {
			__m128 resSse = _mm_max_ps(_mm_min_ps(_mm_loadu_ps(first), oneSse), negOneSse);
			_mm_storeu_ps(result, resSse);

			first += 4;
			result += 4;
		}
	}

	for(int i = 0; i < nonSseSize; ++i) {
		if(*first > 1) {
			*result++ = 1;
		} else if(*first < -1) {
			*result++ = -1;
		} else {
			*result++ = *first;
		}
		++first;
	}
}

void CCpuMathEngine::VectorHardTanh(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	float* result = GetRaw(resultHandle);

	int sseSize;
	int nonSseSize;
	checkSse(vectorSize, sseSize, nonSseSize);

	if(sseSize > 0) {
		const __m128 oneSse = _mm_set_ps1(1.f);
		const __m128 negOneSse = _mm_set_ps1(-1.f);
		for(int i = 0; i < sseSize; ++i) {
			__m128 value = _mm_loadu_ps(first);
			value = _mm_max_ps(value, negOneSse);
			value = _mm_min_ps(value, oneSse);
			_mm_storeu_ps(result, value);

			first += 4;
			result += 4;
		}
	}

	for(int i = 0; i < nonSseSize; ++i) {
		if(*first > 1) {
			*result++ = 1;
		} else if(*first < -1) {
			*result++ = -1;
		} else {
			*result++ = *first;
		}
		++first;
	}
}

void CCpuMathEngine::VectorHardTanhDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);

	int sseSize;
	int nonSseSize;
	checkSse(vectorSize, sseSize, nonSseSize);

	if(sseSize > 0) {
		const __m128 oneSse = _mm_set_ps1(1.f);
		const __m128 negOneSse = _mm_set_ps1(-1.f);
		for(int i = 0; i < sseSize; ++i) {
			__m128 value = _mm_loadu_ps(first);
			__m128 mask = _mm_cmplt_ps(negOneSse, value);
			mask = _mm_and_ps(mask, _mm_cmplt_ps(value, oneSse));
			_mm_storeu_ps(result, _mm_and_ps(mask, _mm_loadu_ps(second)));

			first += 4;
			second += 4;
			result += 4;
		}
	}

	for(int i = 0; i < nonSseSize; ++i) {
		if(*first > 1 || *first < -1) {
			*result++ = 0;
		} else {
			*result++ = *second;
		}
		++first;
		++second;
	}
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

	int sseSize;
	int nonSseSize;
	checkSse( vectorSize, sseSize, nonSseSize );

	if( sseSize > 0 ) {
		const __m128 oneSse = _mm_set_ps1( 1.f );
		const __m128 zeroSse = _mm_set_ps1( 0.f );
		const __m128 slopeSse = _mm_set_ps1( slope );
		const __m128 biasSse = _mm_set_ps1( bias );
		for( int i = 0; i < sseSize; ++i ) {
			__m128 value = _mm_loadu_ps( first );
			value = _mm_mul_ps( value, slopeSse );
			value = _mm_add_ps( value, biasSse );
			_mm_storeu_ps(result, _mm_min_ps( _mm_max_ps( value, zeroSse ), oneSse ) );

			first += 4;
			result += 4;
		}
	}

	const float maxXValue = ( 1 - bias ) / slope;
	const float minXValue = -bias / slope;

	for(int i = 0; i < nonSseSize; ++i) {
		if(*first >= maxXValue ) {
			*result++ = 1;
		} else if(*first <= minXValue ) {
			*result++ = 0;
		} else {
			*result++ = *first * slope + bias;
		}
		++first;
	}
}

void CCpuMathEngine::VectorHardSigmoidDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& slopeHandle, const CConstFloatHandle& biasHandle )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);

	const float slope = *GetRaw( slopeHandle );
	const float bias = *GetRaw( biasHandle );

	ASSERT_EXPR( slope != 0.f );

	const float maxXValue = ( 1.f - bias ) / slope;
	const float minXValue = -bias / slope;

	int sseSize;
	int nonSseSize;
	checkSse(vectorSize, sseSize, nonSseSize);

	if(sseSize > 0) {
		const __m128 minValueSse = _mm_set_ps1(minXValue);
		const __m128 maxValueSse = _mm_set_ps1(maxXValue);
		const __m128 slopeSse = _mm_set_ps1(slope);
		for(int i = 0; i < sseSize; ++i) {
			__m128 value = _mm_loadu_ps(first);
			__m128 mask = _mm_cmplt_ps(minValueSse, value);
			mask = _mm_and_ps(mask, _mm_cmplt_ps(value, maxValueSse));
			_mm_storeu_ps(result, _mm_and_ps(mask, _mm_mul_ps(_mm_loadu_ps(second), slopeSse)));

			first += 4;
			second += 4;
			result += 4;
		}
	}

	for(int i = 0; i < nonSseSize; ++i) {
		if(*first >= maxXValue || *first <= minXValue) {
			*result = 0;
		} else {
			*result = *second * slope;
		}
		result++;
		first++;
		second++;
	}
}

void CCpuMathEngine::VectorHardSigmoidDiffOp(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& slopeHandle, const CConstFloatHandle& /*biasHandle*/ )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw( firstHandle );
	const float* second = GetRaw( secondHandle );
	float* result = GetRaw( resultHandle );

	const float slope = *GetRaw( slopeHandle );
	ASSERT_EXPR( slope != 0.f );

	int sseSize;
	int nonSseSize;
	checkSse( vectorSize, sseSize, nonSseSize );

	if( sseSize > 0 ) {
		const __m128 zeroSse = _mm_setzero_ps();
		const __m128 oneSse = _mm_set_ps1(1.f);
		const __m128 slopeSse = _mm_set_ps1(slope);
		for( int i = 0; i < sseSize; ++i ) {
			__m128 value = _mm_loadu_ps( first );
			__m128 mask = _mm_cmplt_ps( zeroSse, value );
			mask = _mm_and_ps( mask, _mm_cmplt_ps( value, oneSse ) );
			_mm_storeu_ps( result, _mm_and_ps( mask, _mm_mul_ps( _mm_loadu_ps( second ), slopeSse ) ) );

			first += 4;
			second += 4;
			result += 4;
		}
	}

	for( int i = 0; i < nonSseSize; ++i ) {
		if( *first >= 1.f || *first <= 0.f ) {
			*result = 0;
		} else {
			*result = *second * slope;
		}
		result++;
		first++;
		second++;
	}
}

void CCpuMathEngine::VectorSquaredHingeDiff(const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);

	int sseSize;
	int nonSseSize;
	checkSse(vectorSize, sseSize, nonSseSize);

	if(sseSize > 0) {
		const __m128 zeroSse = _mm_setzero_ps();
		const __m128 oneSse = _mm_set_ps1(1);
		const __m128 twoSse = _mm_set_ps1(2);
		const __m128 neg2Sse = _mm_set_ps1(-2);
		const __m128 neg4Sse = _mm_set_ps1(-4);
		for(int i = 0; i < sseSize; ++i) {
			__m128 value = _mm_loadu_ps(first);
			__m128 tmp = _mm_sub_ps(oneSse, value); // 1 - *first
			__m128 res1 = _mm_cmpgt_ps(tmp, twoSse); // 1 - *first > 2 <=> *first < -1
			__m128 res2 = _mm_andnot_ps(res1, tmp); // 1 - *first while !(*first < -1)
			// *first < -1 branch
			res1 = _mm_and_ps(res1, neg4Sse);	// -4
			__m128 value2 = _mm_loadu_ps(second);
			res1 = _mm_mul_ps(res1, value2);	// -4 * *second
			// !(*first < -1) branch
			res2 = _mm_max_ps(zeroSse, res2);	// max(0.f, 1 - *first)
			res2 = _mm_mul_ps(res2, neg2Sse);	// -2 * max(0.f, 1 - *first)
			res2 = _mm_mul_ps(res2, value2);	// -2 * max(0.f, 1 - *first) * *second

			_mm_storeu_ps(result, _mm_add_ps(res1, res2));

			first += 4;
			second += 4;
			result += 4;
		}
	}

	for(int i = 0; i < nonSseSize; ++i) {
		if(*first < -1) {
			*result++ = -4 * *second;
		} else {
			float hinge = max(0.f, 1 - *first);
			*result++ = -2 * hinge * *second;
		}
		++first;
		++second;
	}
}

void CCpuMathEngine::VectorBernulliKLDerivative(const CConstFloatHandle& estimationHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& targetHandle)
{
	ASSERT_EXPR( estimationHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	ASSERT_EXPR( targetHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* estimation = GetRaw(estimationHandle);
	float* result = GetRaw(resultHandle);
	const float MaxKLDerivative = 10;

	int sseSize;
	int nonSseSize;
	checkSse(vectorSize, sseSize, nonSseSize);

	float target = *GetRaw(targetHandle);

	if(sseSize > 0) {
		const __m128 negTargetSse = _mm_set_ps1(-target);
		const __m128 oneSse = _mm_set_ps1(1);
		const __m128 maxSse = _mm_set_ps1(MaxKLDerivative);
		const __m128 minSse = _mm_set_ps1(-MaxKLDerivative);
		for(int i = 0; i < sseSize; ++i) {
			__m128 estimationSse = _mm_loadu_ps(estimation);
			__m128 derivative = _mm_div_ps(negTargetSse, estimationSse);
			__m128 tmp = _mm_div_ps(_mm_add_ps(oneSse, negTargetSse), _mm_sub_ps(oneSse, estimationSse));
			derivative = _mm_add_ps(derivative, tmp);
			derivative = _mm_min_ps(derivative, maxSse);
			derivative = _mm_max_ps(derivative, minSse);

			_mm_storeu_ps(result, derivative);

			estimation += 4;
			result += 4;
		}
	}

	for(int i = 0; i < nonSseSize; ++i) {
		float value = - target / *estimation + (1 - target) / (1 - *estimation);
		*result++ = min(MaxKLDerivative, max(-MaxKLDerivative, value));
		++estimation;
	}
}

void CCpuMathEngine::VectorAdd( const CConstIntHandle& firstHandle,
	const CConstIntHandle& secondHandle, const CIntHandle& resultHandle, int vectorSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const int* first = GetRaw( firstHandle );
	const int* second = GetRaw( secondHandle );
	int* result = GetRaw( resultHandle );

	int sseSize;
	int nonSseSize;
	checkSse( vectorSize, sseSize, nonSseSize );

	for( int i = 0; i < sseSize; ++i ) {
		_mm_storeu_si128( ( __m128i* )result,
			_mm_add_epi32( _mm_loadu_si128( ( __m128i* )first ), _mm_loadu_si128( ( __m128i* )second ) ) );
		first += 4;
		second += 4;
		result += 4;
	}

	for( int i = 0; i < nonSseSize; ++i ) {
		result[i] = first[i] + second[i];
	}
}

void CCpuMathEngine::VectorAddValue( const CConstIntHandle& firstHandle,
	const CIntHandle& resultHandle, int vectorSize, const CConstIntHandle& additionHandle )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( additionHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const int* first = GetRaw( firstHandle );
	int* result = GetRaw( resultHandle );
	int addition = *GetRaw( additionHandle );

	int sseSize;
	int nonSseSize;
	checkSse( vectorSize, sseSize, nonSseSize );

	if( sseSize > 0 ) {
		const __m128i addSse = _mm_set1_epi32( addition );
		for( int i = 0; i < sseSize; ++i ) {
			_mm_storeu_si128( ( __m128i* )result, _mm_add_epi32( _mm_loadu_si128( ( __m128i* )first ), addSse ) );
			first += 4;
			result += 4;
		}
	}

	for( int i = 0; i < nonSseSize; ++i ) {
		result[i] = first[i] + addition;
	}
}

void CCpuMathEngine::VectorSub(const CConstIntHandle& firstHandle,
	const CConstIntHandle& secondHandle, const CIntHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const int* first = GetRaw(firstHandle);
	const int* second = GetRaw(secondHandle);
	int* result = GetRaw(resultHandle);

	int sseSize;
	int nonSseSize;
	checkSse(vectorSize, sseSize, nonSseSize);

	for(int i = 0; i < sseSize; ++i) {
		StoreIntSse4(_mm_sub_epi32(LoadIntSse4(first), LoadIntSse4(second)), result);
		first += 4;
		second += 4;
		result += 4;
	}

	for(int i = 0; i < nonSseSize; ++i) {
		result[i] = first[i] - second[i];
	}
}

void CCpuMathEngine::VectorSub(const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);

	int sseSize;
	int nonSseSize;
	checkSse(vectorSize, sseSize, nonSseSize);

	for(int i = 0; i < sseSize; ++i) {
		_mm_storeu_ps(result, _mm_sub_ps(_mm_loadu_ps(first), _mm_loadu_ps(second)));
		first += 4;
		second += 4;
		result += 4;
	}

	for(int i = 0; i < nonSseSize; ++i) {
		result[i] = first[i] - second[i];
	}
}

void CCpuMathEngine::VectorSub(const CConstFloatHandle& firstHandle, float second, const CFloatHandle& resultHandle,
	int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	float* result = GetRaw(resultHandle);

	int sseSize;
	int nonSseSize;
	checkSse(vectorSize, sseSize, nonSseSize);

	__m128 secondSse = _mm_set_ps1(second);
	for(int i = 0; i < sseSize; ++i) {
		_mm_storeu_ps(result, _mm_sub_ps(_mm_loadu_ps(first), secondSse));
		first += 4;
		result += 4;
	}

	for(int i = 0; i < nonSseSize; ++i) {
		result[i] = first[i] - second;
	}
}

void CCpuMathEngine::VectorSub(float first, const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle,
	int vectorSize)
{
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);

	int sseSize;
	int nonSseSize;
	checkSse(vectorSize, sseSize, nonSseSize);

	__m128 firstSse = _mm_set_ps1(first);
	for(int i = 0; i < sseSize; ++i) {
		_mm_storeu_ps(result, _mm_sub_ps(firstSse, _mm_loadu_ps(second)));
		second += 4;
		result += 4;
	}

	for(int i = 0; i < nonSseSize; ++i) {
		result[i] = first - second[i];
	}
}

void CCpuMathEngine::VectorMultiplyAndSub(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& multHandle)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	CFloatHandleStackVar mult( mathEngine(), 1 );
	mult.SetValue( -*GetRaw(multHandle) );
	VectorMultiplyAndAdd(firstHandle, secondHandle, resultHandle, vectorSize, mult);
}

void CCpuMathEngine::VectorNegMultiply(const CConstFloatHandle& firstHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& multiplierHandle)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( multiplierHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	CFloatHandleStackVar mult( mathEngine(), 1 );
	mult.SetValue( -*GetRaw(multiplierHandle) );
	VectorMultiply(firstHandle, resultHandle, vectorSize, mult);
}

void CCpuMathEngine::VectorEltwiseNegMultiply(const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);

	int sseSize;
	int nonSseSize;
	checkSse(vectorSize, sseSize, nonSseSize);

	if(sseSize > 0) {
		__m128 zero = _mm_setzero_ps();
		for(int i = 0; i < sseSize; ++i) {
			_mm_storeu_ps(result, _mm_sub_ps(zero, _mm_mul_ps(_mm_loadu_ps(first), _mm_loadu_ps(second))));
			first += 4;
			second += 4;
			result += 4;
		}
	}

	for(int i = 0; i < nonSseSize; ++i) {
		*result++ = -*first++ * *second++;
	}
}

void CCpuMathEngine::VectorEltwiseDivide(const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);

	int sseSize;
	int nonSseSize;
	checkSse(vectorSize, sseSize, nonSseSize);

	for(int i = 0; i < sseSize; ++i) {
		_mm_storeu_ps(result, _mm_div_ps(_mm_loadu_ps(first), _mm_loadu_ps(second)));
		first += 4;
		second += 4;
		result += 4;
	}

	for(int i = 0; i < nonSseSize; ++i) {
		*result++ = *first++ / *second++;
	}
}

void CCpuMathEngine::VectorEltwisePower(const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);

	for(int i = 0; i < vectorSize; ++i) {
		*result++ = (*second == 1) ? *first : powf(*first, *second);
		++first;
		++second;
	}
}

void CCpuMathEngine::VectorSqrt(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	float* result = GetRaw(resultHandle);

	int sseSize;
	int nonSseSize;
	checkSse(vectorSize, sseSize, nonSseSize);

	for(int i = 0; i < sseSize; ++i) {
		_mm_storeu_ps(result, _mm_sqrt_ps(_mm_loadu_ps(first)));
		first += 4;
		result += 4;
	}

	for(int i = 0; i < nonSseSize; ++i) {
		*result++ = sqrtf(*first++);
	}
}

// result = 1 / first
void CCpuMathEngine::VectorInv(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	float* result = GetRaw(resultHandle);

	int sseSize;
	int nonSseSize;
	checkSse(vectorSize, sseSize, nonSseSize);

	if(sseSize > 0) {
		const __m128 oneSse = _mm_set_ps1(1);
		const __m128 zeroSse = _mm_setzero_ps();
		const __m128 fltMaxSse = _mm_set_ps1(FLT_MAX);
		const __m128 fltNegMaxSse = _mm_set_ps1(-FLT_MAX);
		const __m128 fltMinSse = _mm_set_ps1(FLT_MIN);
		const __m128 fltNegMinSse = _mm_set_ps1(-FLT_MIN);
		for(int i = 0; i < sseSize; ++i) {
			__m128 value = _mm_loadu_ps(first);
			__m128 cmpSmallInt = _mm_and_ps(_mm_cmple_ps(fltNegMinSse, value), _mm_cmple_ps(value, fltMinSse));
			__m128 cmpNeg = _mm_cmplt_ps(value, zeroSse);
			__m128 cmpSmallNeg = _mm_and_ps(cmpNeg, cmpSmallInt);
			__m128 cmpSmallPos = _mm_andnot_ps(cmpNeg, cmpSmallInt);
			__m128 res = _mm_or_ps(_mm_or_ps(
				_mm_andnot_ps(cmpSmallInt,
					_mm_div_ps(oneSse, _mm_add_ps(cmpSmallInt, value))), // Add 1 to the zero elements before dividing
				_mm_and_ps(cmpSmallNeg, fltNegMaxSse)), _mm_and_ps(cmpSmallPos, fltMaxSse));
			_mm_storeu_ps(result, res);
			first += 4;
			result += 4;
		}
	}

	for(int i = 0; i < nonSseSize; ++i) {
		float div = *first++;
		if(-FLT_MIN <= div && div < 0) {
			*result++ = -FLT_MAX;
		} else if(0 <= div && div <= FLT_MIN) {
			*result++ = FLT_MAX;
		} else {
			*result++ = 1.f / div;
		}
	}
}

void CCpuMathEngine::VectorSigmoid(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw( firstHandle );
	float* result = GetRaw( resultHandle );
	const int curThreadCount = IsOmpRelevant( vectorSize, 2 * vectorSize ) ? threadCount : 1;
	if( simdMathEngine != nullptr ) {
		simdMathEngine->Sigmoid( result, first, vectorSize, curThreadCount > 1 );
	} else if( curThreadCount > 1 ) {
		NEOML_OMP_NUM_THREADS( curThreadCount )
		{
			int start;
			int count;
			if( OmpGetTaskIndexAndCount( vectorSize, start, count ) ) {
				vectorSigmoid( first + start, result + start, count );
			}
		}
	} else {
		vectorSigmoid( first, result, vectorSize );
	}
}

void CCpuMathEngine::VectorSigmoidDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	VectorExp(firstHandle, resultHandle, vectorSize);

	int sseSize;
	int nonSseSize;
	checkSse(vectorSize, sseSize, nonSseSize);

	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);

	if(sseSize > 0) {
		const __m128 oneSse = _mm_set_ps1(1);
		for(int i = 0; i < sseSize; ++i) {
			__m128 value = _mm_loadu_ps(result);
			__m128 value1 = _mm_add_ps(value, oneSse);
			value = _mm_div_ps(_mm_mul_ps(value, _mm_loadu_ps(second)), _mm_mul_ps(value1, value1));
			_mm_storeu_ps(result, value);

			second += 4;
			result += 4;
		}
	}

	for(int i = 0; i < nonSseSize; ++i) {
		float result1 = (*result + 1);
		*result = *second * *result / (result1 * result1);
		++second;
		++result;
	}
}

void CCpuMathEngine::VectorSigmoidDiffOp(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	int sseSize;
	int nonSseSize;
	checkSse(vectorSize, sseSize, nonSseSize);

	const float* first = GetRaw(firstHandle);
	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);

	if(sseSize > 0) {
		const __m128 oneSse = _mm_set_ps1(1);
		for(int i = 0; i < sseSize; ++i) {
			__m128 value = _mm_loadu_ps(first);
			__m128 value1 = _mm_sub_ps(oneSse, value);
			value = _mm_mul_ps(_mm_mul_ps(value, value1), _mm_loadu_ps(second));
			_mm_storeu_ps(result, value);

			first += 4;
			second += 4;
			result += 4;
		}
	}

	for(int i = 0; i < nonSseSize; ++i) {
		float value = *first++;
		*result++ = value * (1.f - value) * *second++;
	}
}

void CCpuMathEngine::VectorTanhDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	VectorTanh(firstHandle, resultHandle, vectorSize);

	int sseSize;
	int nonSseSize;
	checkSse(vectorSize, sseSize, nonSseSize);

	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);

	if(sseSize > 0) {
		const __m128 oneSse = _mm_set_ps1(1);
		for(int i = 0; i < sseSize; ++i) {
			__m128 value = _mm_loadu_ps(result);
			value = _mm_mul_ps(_mm_loadu_ps(second), _mm_sub_ps(oneSse, _mm_mul_ps(value, value)));
			_mm_storeu_ps(result, value);

			result += 4;
			second += 4;
		}
	}

	for(int i = 0; i < nonSseSize; ++i) {
		*result = *second * (1.f - *result * *result);
		++result;
		++second;
	}
}

void CCpuMathEngine::VectorTanhDiffOp(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	int sseSize;
	int nonSseSize;
	checkSse(vectorSize, sseSize, nonSseSize);

	const float* first = GetRaw(firstHandle);
	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);

	if(sseSize > 0) {
		const __m128 oneSse = _mm_set_ps1(1);
		for(int i = 0; i < sseSize; ++i) {
			__m128 value = _mm_loadu_ps(first);
			value = _mm_mul_ps(_mm_sub_ps(oneSse, _mm_mul_ps(value, value)), _mm_loadu_ps(second));
			_mm_storeu_ps(result, value);

			first += 4;
			second += 4;
			result += 4;
		}
	}

	for(int i = 0; i < nonSseSize; ++i) {
		float value = *first++;
		*result++ = (1.f - value * value) * *second++;
	}
}

void CCpuMathEngine::VectorPowerDiff(float exponent, const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	VectorPower(exponent - 1, firstHandle, resultHandle, vectorSize);

	int sseSize;
	int nonSseSize;
	checkSse(vectorSize, sseSize, nonSseSize);

	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);

	if(sseSize > 0) {
		const __m128 exponentSse = _mm_set_ps1(exponent);
		for(int i = 0; i < sseSize; ++i) {
			__m128 value = _mm_mul_ps(_mm_loadu_ps(second), _mm_mul_ps(exponentSse, _mm_loadu_ps(result)));
			_mm_storeu_ps(result, value);

			result += 4;
			second += 4;
		}
	}

	for(int i = 0; i < nonSseSize; ++i) {
		*result = *second * exponent * *result;
		++result;
		++second;
	}
}

void CCpuMathEngine::VectorPowerDiffOp(float exponent, const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	float exponentOp = (exponent - 1.f) / exponent;

	VectorPower(exponentOp, firstHandle, resultHandle, vectorSize);

	int sseSize;
	int nonSseSize;
	checkSse(vectorSize, sseSize, nonSseSize);

	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);

	if(sseSize > 0) {
		const __m128 exponentSse = _mm_set_ps1(exponent);
		for(int i = 0; i < sseSize; ++i) {
			__m128 value = _mm_mul_ps(_mm_loadu_ps(second), _mm_mul_ps(exponentSse, _mm_loadu_ps(result)));
			_mm_storeu_ps(result, value);

			result += 4;
			second += 4;
		}
	}

	for(int i = 0; i < nonSseSize; ++i) {
		*result = *second * exponent * *result;
		++result;
		++second;
	}
}

void CCpuMathEngine::VectorL1DiffAdd(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize,
	const CConstFloatHandle& hubertThresholdHandle, const CConstFloatHandle& multHandle)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( secondHandle.GetMathEngine() == this );
	ASSERT_EXPR( hubertThresholdHandle.GetMathEngine() == this );
	ASSERT_EXPR( multHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const float* first = GetRaw(firstHandle);
	const float* second = GetRaw(secondHandle);
	float* result = GetRaw(resultHandle);
	float threshold = *GetRaw(hubertThresholdHandle);
	float mult = *GetRaw(multHandle);

	int sseSize;
	int nonSseSize;
	checkSse(vectorSize, sseSize, nonSseSize);

	if(sseSize > 0) {
		__m128 minSse = _mm_set_ps1(-threshold);
		__m128 maxSse = _mm_set_ps1(threshold);
		__m128 multSse = _mm_set_ps1(mult);

		for(int i = 0; i < sseSize; ++i) {
			__m128 x = _mm_loadu_ps(second);
			x = _mm_min_ps(maxSse, _mm_max_ps(minSse, x));
			__m128 res = _mm_add_ps(_mm_loadu_ps(first), _mm_mul_ps(multSse, x));
			_mm_storeu_ps(result, res);

			first += 4;
			second += 4;
			result += 4;
		}
	}

	for(int i = 0; i < nonSseSize; ++i) {
		float x = *second++;
		if(x < -threshold) {
			x = -threshold;
		} else if(x > threshold) {
			x = threshold;
		}

		*result++ = *first++ + mult * x;
	}
}

void CCpuMathEngine::VectorEltwiseNot( const CConstIntHandle& firstHandle, const CIntHandle& resultHandle,
	int vectorSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const int* first = GetRaw( firstHandle );
	int* result = GetRaw( resultHandle );

	int sseSize;
	int nonSseSize;
	checkSse( vectorSize, sseSize, nonSseSize );

	const __m128i zeros = _mm_set1_epi32( 0 );
	const __m128i ones = _mm_set1_epi32( 1 );

	for( int i = 0; i < sseSize; ++i ) {
		StoreIntSse4( _mm_and_si128( ones, _mm_cmpeq_epi32( LoadIntSse4( first ), zeros ) ), result );
		first += 4;
		result += 4;
	}

	if( nonSseSize > 0 ) {
		StoreIntSse( _mm_and_si128( ones, _mm_cmpeq_epi32( LoadIntSse( first, nonSseSize ), zeros ) ),
			result, nonSseSize );
	}
}

void CCpuMathEngine::VectorEltwiseNotNegative( const CConstIntHandle& firstHandle, const CFloatHandle& resultHandle,
	int vectorSize )
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const int* first = GetRaw( firstHandle );
	float* result = GetRaw( resultHandle );

	for( int i = 0; i < vectorSize; ++i ) {
		*result++ = *first++ >= 0 ? 1.f : 0.f;
	}
}

void CCpuMathEngine::VectorNegLog(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize)
{
	ASSERT_EXPR( firstHandle.GetMathEngine() == this );
	ASSERT_EXPR( resultHandle.GetMathEngine() == this );
	CCpuExecutionScope scope;

	VectorLog(firstHandle, resultHandle, vectorSize);

	float* result = GetRaw(resultHandle);

	int sseSize;
	int nonSseSize;
	checkSse(vectorSize, sseSize, nonSseSize);

	__m128 mask = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
	for(int i = 0; i < sseSize; ++i) {
		__m128 val = LoadSse4(result);
		val = _mm_xor_ps(val, mask);
		StoreSse4(val, result);
		result += 4;
	}

	if(nonSseSize > 0) {
		__m128 val = LoadSse(result, nonSseSize);
		val = _mm_xor_ps(val, mask);
		StoreSse(val, result, nonSseSize);
	}
}

} // namespace NeoML

#endif // NEOML_USE_SSE
