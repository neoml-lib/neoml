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

#include <common.h>
#pragma hdrstop

#include <NeoML/TraditionalML/OneVersusOne.h>
#include <OneVersusOneModel.h>

namespace NeoML {

// The data for training binary classification: 0 for the first class, 1 for the second
class COneVersusOneTrainingData : public IProblem {
public:
	COneVersusOneTrainingData( const IProblem& data, int firstClass, int secondClass );

	// IProblem interface methods
	int GetClassCount() const override { return 2; }
	int GetFeatureCount() const override { return desc.Width; }
	bool IsDiscreteFeature( int index ) const override { return baseProblem->IsDiscreteFeature( index ); }
	int GetVectorCount() const override { return desc.Height; }
	int GetClass( int index ) const override;
	CFloatMatrixDesc GetMatrix() const override { return desc; }
	double GetVectorWeight( int index ) const override { return baseProblem->GetVectorWeight( vectorIndices[index] ); }
	int GetDiscretizationValue( int index ) const override { return baseProblem->GetDiscretizationValue( index ); }

private:
	const CPtr<const IProblem> baseProblem; // original data
	const int firstClass; // first class of problem
	const int secondClass; // second class of problem
	CFloatMatrixDesc desc; // matrix descriptor
	CArray<int> rowStart; // row starts in matrix
	CArray<int> rowEnd; // row ends in matrix
	CArray<int> vectorIndices; // indices of vectors in base problem
};

COneVersusOneTrainingData::COneVersusOneTrainingData( const IProblem& data, int _firstClass, int _secondClass ) :
	baseProblem( &data ),
	firstClass( _firstClass ),
	secondClass( _secondClass )
{
	NeoAssert( firstClass != secondClass );
	const CFloatMatrixDesc baseDesc = data.GetMatrix();
	desc.Height = 0;
	desc.Width = baseDesc.Width;
	desc.Columns = baseDesc.Columns; // This works for both sparse and dense cases
	desc.Values = baseDesc.Values;
	for( int vecIndex = 0; vecIndex < baseDesc.Height; ++vecIndex ) {
		const int currClass = baseProblem->GetClass( vecIndex );
		if( currClass == firstClass || currClass == secondClass ) {
			desc.Height++;
			rowStart.Add( baseDesc.PointerB[vecIndex] );
			rowEnd.Add( baseDesc.PointerE[vecIndex] );
			vectorIndices.Add( vecIndex );
		}
	}
	desc.PointerB = rowStart.GetPtr();
	desc.PointerE = rowEnd.GetPtr();
}

int COneVersusOneTrainingData::GetClass( int index ) const
{
	const int baseClass = baseProblem->GetClass( vectorIndices[index] );
	NeoPresume( baseClass == firstClass || baseClass == secondClass );
	return baseClass == firstClass ? 0 : 1;
}

//---------------------------------------------------------------------------------------------------------

COneVersusOne::COneVersusOne( ITrainingModel& _baseClassifier ) :
	baseClassifier( _baseClassifier ),
	log( nullptr )
{
}

CPtr<IModel> COneVersusOne::Train( const IProblem& trainingData )
{
	if( log != nullptr ) {
		*log << "\nOne versus one traning started:\n";
	}

	CObjectArray<IModel> classifiers;
	const int classCount = trainingData.GetClassCount();
	for( int firstClass = 0; firstClass < classCount - 1; ++firstClass ) {
		for( int secondClass = firstClass + 1; secondClass < classCount; ++secondClass ) {
			CPtr<IProblem> subproblem = FINE_DEBUG_NEW COneVersusOneTrainingData( trainingData, firstClass, secondClass );
			classifiers.Add( baseClassifier.Train( *subproblem ) );
		}
	}

	if( log != nullptr ) {
		*log << "\nOne versus one training finished\n";
	}

	return FINE_DEBUG_NEW COneVersusOneModel( classifiers );
}

} // namespace NeoML
