/* Copyright © 2023-2024 ABBYY

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
#include <CPUInfo.h>

#if FINE_ARCHITECTURE( FINE_X64 )

#include <stdint.h>
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include <immintrin.h>

#ifdef __cplusplus
extern "C" {
#endif

void fma_shuffle_tpt( uint64_t loop_cnt );
void fma_only_tpt( uint64_t loop_cnt );

int64_t rdtsc( void )
{
	return __rdtsc();
}

int fma_unit_count( void )
{
	int i;
	uint64_t fma_shuf_tpt_test[3];
	uint64_t fma_shuf_tpt_test_min;
	uint64_t fma_only_tpt_test[3];
	uint64_t fma_only_tpt_test_min;
	uint64_t start = 0;
	int number_of_fma_units_per_core = 2;

	/*********************************************************/
	/* Step 1: Warmup */
	/*********************************************************/

	fma_only_tpt( 100000 );

	/*********************************************************/
	/* Step 2: Execute FMA and Shuffle TPT Test */
	/*********************************************************/
	for( i = 0; i < 3; ++i ) {
		start = rdtsc();
		fma_shuffle_tpt( 1000 );
		fma_shuf_tpt_test[i] = rdtsc() - start;
	}

	/*********************************************************/
	/* Step 3: Execute FMA only TPT Test */
	/*********************************************************/
	for( i = 0; i < 3; ++i ) {
		start = rdtsc();
		fma_only_tpt( 1000 );
		fma_only_tpt_test[i] = rdtsc() - start;
	}

	/*********************************************************/
	/* Step 4: Decide if 1 FMA server or 2 FMA server */
	/*********************************************************/
	fma_shuf_tpt_test_min = fma_shuf_tpt_test[0];
	fma_only_tpt_test_min = fma_only_tpt_test[0];
	for( i = 1; i < 3; ++i ) {
		if( (int)fma_shuf_tpt_test[i] < (int)fma_shuf_tpt_test_min ) {
			fma_shuf_tpt_test_min = fma_shuf_tpt_test[i];
		}
		if( (int)fma_only_tpt_test[i] < (int)fma_only_tpt_test_min ) {
			fma_only_tpt_test_min = fma_only_tpt_test[i];
		}
	}

	if( ( double( fma_shuf_tpt_test_min ) / fma_only_tpt_test_min ) < 1.5 ) {
		number_of_fma_units_per_core = 1;
	}

	printf( " *** x64 AVX512 %d FMA units per core *** \n", number_of_fma_units_per_core );
	return number_of_fma_units_per_core;
}

#ifdef __cplusplus
}
#endif

//-------------------------------------------------------------------------------------------------

int Avx512FmaUnitCount()
{
	return fma_unit_count();
}

#endif // x64
