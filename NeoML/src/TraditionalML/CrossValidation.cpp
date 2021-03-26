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

#include <NeoML/TraditionalML/CrossValidation.h>
#include <NeoML/TraditionalML/CrossValidationSubProblem.h>
#include <NeoML/TraditionalML/StratifiedCrossValidationSubProblem.h>

namespace NeoML {

CCrossValidation::CCrossValidation( ITrainingModel& _trainingModel, const IProblem* _problem ) :
	trainingModel( _trainingModel ),
	problem( _problem )
{
	NeoAssert( problem != 0 );
}

void CCrossValidation::Execute( int partsCount, TScore score, CCrossValidationResult& result, bool stratified )
{
	NeoAssert( partsCount > 0 );
	NeoAssert( partsCount < problem->GetVectorCount() / 2 );

	result.Problem = problem;
	result.Models.Empty();
	result.Results.Empty();
	result.Results.SetSize( problem->GetVectorCount() );
	result.ModelIndex.Empty();
	result.ModelIndex.SetSize( problem->GetVectorCount() );
	result.Success.Empty();

	for( int i = 0; i < partsCount; i++ ) {
		// Choose the training subset
		CPtr<ISubProblem> trainSubProblem;
		if( stratified ) {
			trainSubProblem = FINE_DEBUG_NEW CStratifiedCrossValidationSubProblem( problem, partsCount, i, false );
		} else {
			trainSubProblem = FINE_DEBUG_NEW CCrossValidationSubProblem( problem, partsCount, i, false );
		}

		// Train the model
		CPtr<IModel> model = trainingModel.Train( *trainSubProblem );
		result.Models.Add( model );

		// Choose the testing subset
		CPtr<ISubProblem> testSubProblem;
		if( stratified ) {
			testSubProblem = FINE_DEBUG_NEW CStratifiedCrossValidationSubProblem( problem, partsCount, i, true );
		} else {
			testSubProblem = FINE_DEBUG_NEW CCrossValidationSubProblem( problem, partsCount, i, true );
		}

		CFloatMatrixDesc testSubProblemMatrix = testSubProblem->GetMatrix();
		
		// Current model classification result to calculate the loss function
		CArray<CClassificationResult> classificationResults;

		for( int j = 0; j < testSubProblem->GetVectorCount(); j++ ) {
			CFloatVectorDesc vector;
			testSubProblemMatrix.GetRow( j, vector );
			model->Classify( vector, result.Results[testSubProblem->GetOriginalIndex( j )] );
			classificationResults.Add( result.Results[testSubProblem->GetOriginalIndex( j )] );

			result.ModelIndex[testSubProblem->GetOriginalIndex( j )] = i;
		}

		result.Success.Add( score( classificationResults, testSubProblem ) );
	}
}

} // namespace NeoML
