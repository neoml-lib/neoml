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

#include <metal_stdlib>

using namespace metal;

#define FLT_MIN_LOG -87.33654474f
#define FLT_MAX_LOG 88.f

inline float LogSumExpFunc( float f, float s )
{
    if( f >= s ) {
        return f + log( 1 + exp( s - f ) );
    }
    return s + log( 1 + exp( f - s ) );
}

inline float ExponentFunc(float f)
{
    if(f < FLT_MIN_LOG) {
        return 0;
    } else if(f > FLT_MAX_LOG) {
        return FLT_MAX;
    } else {
        return exp( f );
    }
}

inline float LogFunc(float f)
{
    return log( min( max( f, FLT_MIN ), FLT_MAX ) );
}
