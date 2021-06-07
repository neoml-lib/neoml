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
#include <NeoML/TraditionalML/Model.h>
#include <NeoML/TraditionalML/Problem.h>

namespace NeoML {

// The algorithm used for multi-class classification
enum TMulticlassMode {
	MM_SingleClassifier = 0, // Special case supported in some classifiers (CDesicionTree)
	MM_OneVsAll,
	MM_OneVsOne,
	MM_Count
};

// Classification training interface
class NEOML_API ITrainingModel {
public:
	// Trains a classifier on the input data
	virtual CPtr<IModel> Train( const IProblem& trainingClassificationData ) = 0;

	template<typename TModel>
	CPtr<TModel> TrainModel( const IProblem& trainingClassificationData )
	{
		return CheckCast<TModel>( Train( trainingClassificationData ) );
	}

	virtual ~ITrainingModel();
};

// Regression training interface
class NEOML_API IRegressionTrainingModel {
public:
	virtual CPtr<IRegressionModel> TrainRegression( const IRegressionProblem& problem ) = 0;

	virtual ~IRegressionTrainingModel();
};

} // namespace NeoML
