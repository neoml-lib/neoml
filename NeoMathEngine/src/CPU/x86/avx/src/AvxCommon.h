/* Copyright © 2017-2021 ABBYY Production LLC

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

#pragma once

#include <immintrin.h>

#define PERMUTE2( p1, p0 ) ( ( p0 << 0 ) + ( p1 << 4 ) )
#define PERMUTE4( p3, p2, p1, p0 ) ( ( p0 << 0 ) + ( p1 << 2 ) + ( p2 << 4 ) + ( p3 << 6 ) )
#define PERMUTE8( p7, p6, p5, p4, p3, p2, p1, p0 ) _mm256_set_epi32( p7, p6, p5, p4, p3, p2, p1, p0 )
#define BLEND8( b7, b6, b5, b4, b3, b2, b1, b0 ) ( b0 + ( b1 << 1 ) + ( b2 << 2 ) + ( b3 << 3 ) + \
	( b4 << 4 ) + ( b5 << 5 ) + ( b6 << 6 ) + ( b7 << 7 ) )
#define SHUFFLE4( s3, s2, s1, s0 ) ( ( s3 << 3 ) + ( s2 << 2 ) + ( s1 << 1 ) + ( s0 << 0 ) )

#ifndef _mm256_loadu2_m128
#define _mm256_loadu2_m128( hiAddr, loAddr ) \
  _mm256_insertf128_ps( _mm256_castps128_ps256( _mm_loadu_ps ( loAddr ) ), _mm_loadu_ps( hiAddr ), 1 )
#endif

#ifndef _mm256_storeu2_m128
#define _mm256_storeu2_m128( hiAddr, loAddr, data ) \
  _mm_storeu_ps ( loAddr, _mm256_castps256_ps128( data ) ); \
  _mm_storeu_ps ( hiAddr, _mm256_extractf128_ps( data, 1) )
#endif

