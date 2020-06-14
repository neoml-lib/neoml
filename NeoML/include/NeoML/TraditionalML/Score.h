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

#include <NeoML/TraditionalML/Problem.h>
#include <NeoML/TraditionalML/ClassificationResult.h>

namespace NeoML {

typedef double( *TScore ) ( const CArray<CClassificationResult>& classificationResult, const IProblem* problem );

// AccuracyScore implementation. Returns the total normalized weight of the objects that were classified correctly.
NEOML_API double AccuracyScore( const CArray<CClassificationResult>& classificationResult,
	const IProblem* problem );

// F1 measure implementation for binary classification
NEOML_API double F1Score( const CArray<CClassificationResult>& classificationResult,
	const IProblem* problem );

} // namespace NeoML
