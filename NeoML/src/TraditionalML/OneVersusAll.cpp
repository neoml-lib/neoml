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

#include <NeoML/TraditionalML/OneVersusAll.h>
#include <OneVersusAllModel.h>

namespace NeoML {

// The data for training binary classification: 0 for the target class, 1 for the other classes
class COneVersusAllTrainingData : public IProblem {
public:
	COneVersusAllTrainingData( const IProblem* data, int baseClass );

	// IProblem interface methods
	int GetClassCount() const override { return data->GetClassCount(); }
	int GetFeatureCount() const override { return data->GetFeatureCount(); }
	bool IsDiscreteFeature( int ) const override { return false; }
	int GetVectorCount() const override { return data->GetVectorCount(); }
	int GetClass( int index ) const override { return ( data->GetClass( index) == baseClass ) ? 0 : 1; }
	CSparseFloatMatrixDesc GetMatrix() const override { return data->GetMatrix(); }
	double GetVectorWeight( int index ) const override { return data->GetVectorWeight( index ); }
	int GetDiscretizationValue( int index ) const override { return data->GetDiscretizationValue( index ); }

protected:
	virtual ~COneVersusAllTrainingData() {} // delete prohibited

private:
	const CPtr<const IProblem> data; // the source data
	const int baseClass; // the number of the target class
};

inline COneVersusAllTrainingData::COneVersusAllTrainingData( const IProblem* _data, int _baseClass ) :
	data( _data ),
	baseClass( _baseClass )
{
}

//---------------------------------------------------------------------------------------------------------

COneVersusAll::COneVersusAll( ITrainingModel& _baseBinaryClassifier ) :
	baseBinaryClassifier( _baseBinaryClassifier ),
	logStream( 0 )
{
}

CPtr<IModel> COneVersusAll::Train( const IProblem& trainingClassificationData )
{
	if( logStream != 0 ) {
		*logStream << "\nOne versus all training started:\n";
	}

	CObjectArray<IModel> etalons;
	for( int i = 0; i < trainingClassificationData.GetClassCount(); i++ ) {
		CPtr<IProblem> trainingData = FINE_DEBUG_NEW COneVersusAllTrainingData( &trainingClassificationData, i );
		etalons.Add( baseBinaryClassifier.Train( *trainingData ) );
	}

	if( logStream != 0 ) {
		*logStream << "\nOne versus all training finished\n";
	}

	return FINE_DEBUG_NEW COneVersusAllModel( etalons );
}

} // namespace NeoML
