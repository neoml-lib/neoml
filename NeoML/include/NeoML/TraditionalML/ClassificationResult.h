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

#pragma once

#include <NeoML/NeoMLDefs.h>
#include <NeoML/TraditionalML/ClassificationProbability.h>

namespace NeoML {

// Classification result
struct NEOML_API CClassificationResult {
	int PreferredClass;									// the number of the class to which the input object is assigned
	CClassificationProbability ExceptionProbability;	// the probability that the input object fits none of the classes
	CArray<CClassificationProbability> Probabilities;	// the array of probabilities for the object to belong to each of the classes

	CClassificationResult() : PreferredClass( 0 ), ExceptionProbability( 0 ) {}
	CClassificationResult( const CClassificationResult& other ) {
		PreferredClass = other.PreferredClass;
		ExceptionProbability = other.ExceptionProbability;
		other.Probabilities.CopyTo( Probabilities );
	}
};

} // namespace NeoML
