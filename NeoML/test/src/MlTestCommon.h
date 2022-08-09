/* Copyright © 2017-2022 ABBYY Production LLC

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

namespace NeoMLTest {

inline CSparseFloatVector generateRandomVector( CRandom& rand, int maxLength = 100,
	float minValue = -100., float maxValue = 100. )
{
	CSparseFloatVector res;
	for( int i = 0; i < maxLength; ++i ) {
		int index = rand.UniformInt( 0, maxLength - 1 ) ;
		res.SetAt( index, static_cast<float>( rand.Uniform( minValue, maxValue ) ) );
	}
	return res;
}

} // namespace NeoMLTest
