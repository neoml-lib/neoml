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
#include <NeoML/TraditionalML/TrainingModel.h>
#include <NeoML/TraditionalML/Score.h>

namespace NeoML {

// Cross-validation result
struct NEOML_API CCrossValidationResult {
	CPtr<const IProblem> Problem; // the data on which cross-validation was performed
	CObjectArray<IModel> Models; // the models trained
	CArray<double> Success; // the classification quality for each of the models on the corresponding training set
	CArray<CClassificationResult> Results; // the vector classification results
	CArray<int> ModelIndex; // the index of the model that classified the given vector
};

// The cross-validation algorithm
class NEOML_API CCrossValidation {
public:
	CCrossValidation( ITrainingModel& trainingModel, const IProblem* problem );

	// Performs cross-validation
	void Execute( int partsCount, TScore score, CCrossValidationResult& results, bool stratified );

private:
	ITrainingModel& trainingModel; // the base training model
	const CPtr<const IProblem> problem; // the input data
};

} // namespace NeoML
