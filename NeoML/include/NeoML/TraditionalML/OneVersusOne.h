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

#include <NeoML/NeoMLDefs.h>
#include <NeoML/TraditionalML/FloatVector.h>
#include <NeoML/TraditionalML/ClassificationResult.h>
#include <NeoML/TraditionalML/TrainingModel.h>

namespace NeoML {

DECLARE_NEOML_MODEL_NAME( OneVersusOneModelName, "NeoMLOneVersusOneModel" )

// One-versus-one classification model interface
class NEOML_API IOneVersusOneModel : public IModel {
public:
	virtual ~IOneVersusOneModel() = default;

	// Gets the basic IModel for all the binary classifiers
	virtual const CObjectArray<IModel>& GetModels() const = 0;
};

// One versus one classifier training interface
class NEOML_API COneVersusOne : public ITrainingModel {
public:
	explicit COneVersusOne( ITrainingModel& baseBinaryClassifier );

	// Sets a text stream for logging
	void SetLog( CTextStream* newLog ) { log = newLog; }

	// ITrainingModel interface methods
	CPtr<IModel> Train( const IProblem& traningData ) override;

private:
	ITrainingModel& baseClassifier; // the basic binary classifier used
	CTextStream* log; // the logging stream
};

} // namespace NeoML
