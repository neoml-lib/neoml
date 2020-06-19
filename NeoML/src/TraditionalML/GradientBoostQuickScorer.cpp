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

#include <NeoML/TraditionalML/GradientBoostQuickScorer.h>
#include <GradientBoostQSEnsemble.h>

namespace NeoML {

#define DBL_LOG_MAX 709.
#define DBL_LOG_MIN -709.

// Exponent function with limitations to avoid NaN
static inline double exponentFunc( double f )
{
	if( f < DBL_LOG_MIN ) {
		return 0;
	} else if( f > DBL_LOG_MAX ) {
		return DBL_MAX;
	} else {
		return exp( f );
	}
}

//------------------------------------------------------------------------------------------------------------

IGradientBoostQSModel::~IGradientBoostQSModel()
{
}

IGradientBoostQSRegressionModel::~IGradientBoostQSRegressionModel()
{
}

// The QuickScorer model
class CGradientBoostQSModel : public IGradientBoostQSModel, public IGradientBoostQSRegressionModel {
public:
	CGradientBoostQSModel(); // for serialization
	CGradientBoostQSModel( const CArray<CGradientBoostEnsemble>& ensembles, CGradientBoost::TLossFunction lossFunction,
		double learningRate );

	// IGradientBoostQSModel interface methods
	int GetClassCount() const override { return ensembles.Size() == 1 ? 2 : ensembles.Size(); };
	bool Classify( const CSparseFloatVectorDesc& data, CClassificationResult& result ) const override;
	bool Classify( const CFloatVector& data, CClassificationResult& result ) const override;

	// IGradientBoostQSModel interface methods
	bool ClassifyEx( const CSparseFloatVector& data, CArray<CClassificationResult>& results ) const override;
	bool ClassifyEx( const CSparseFloatVectorDesc& data, CArray<CClassificationResult>& results ) const override;

	// IRegressionModel interface methods
	double Predict( const CSparseFloatVector& data ) const override;
	double Predict( const CFloatVector& data ) const override;
	double Predict( const CSparseFloatVectorDesc& data ) const override;

	// General methods
	double GetLearningRate() const override { return learningRate; };
	void Serialize( CArchive& archive ) override;

private:
	CPointerArray<CGradientBoostQSEnsemble> ensembles; // optimized trees ensembles, one ensemble per class
	CGradientBoost::TLossFunction lossFunction; // the loss function used for building the trees
	double learningRate; // the learning rate used

	bool classify( double prediction, CClassificationResult& result ) const;
	bool classify( CArray<double>& predictions, CClassificationResult& result ) const;
	double probability( double prediction ) const;
};

REGISTER_NEOML_MODEL( CGradientBoostQSModel, GradientBoostQSModelName )

CGradientBoostQSModel::CGradientBoostQSModel() :
	lossFunction( CGradientBoost::LF_Undefined ),
	learningRate( 0 )
{
}

CGradientBoostQSModel::CGradientBoostQSModel( const CArray<CGradientBoostEnsemble>& gbEnsembles, CGradientBoost::TLossFunction _lossFunction, double _learningRate ) : 
	lossFunction( _lossFunction ),
	learningRate( _learningRate )
{
	ensembles.SetBufferSize( gbEnsembles.Size() );
	for( int ensembleIndex = 0; ensembleIndex < gbEnsembles.Size(); ensembleIndex++ ) {
		const CGradientBoostEnsemble& gbEnsemble = gbEnsembles[ensembleIndex];
		ensembles.Add( FINE_DEBUG_NEW CGradientBoostQSEnsemble() );
		ensembles.Last()->Build( gbEnsemble );
	}
}

double CGradientBoostQSModel::Predict( const CSparseFloatVector& data ) const
{
	return ensembles.First()->Predict( data ) * learningRate;
}

double CGradientBoostQSModel::Predict( const CFloatVector& data ) const
{
	return ensembles.First()->Predict( data ) * learningRate;
}

double CGradientBoostQSModel::Predict( const CSparseFloatVectorDesc& data ) const
{
	return ensembles.First()->Predict( data ) * learningRate;
}

bool CGradientBoostQSModel::Classify( const CSparseFloatVectorDesc& data, CClassificationResult& result ) const 
{
	if( GetClassCount() == 2 ) {
		const double value = ensembles.First()->Predict( data );
		return classify( value * learningRate, result );
	}

	CArray<double> predictions;
	predictions.SetBufferSize( ensembles.Size() );
	for( int ensembleIndex = 0; ensembleIndex < ensembles.Size(); ensembleIndex++ ) {
		predictions.Add( ensembles[ensembleIndex]->Predict( data ) );
	}
	return classify( predictions, result );
}

bool CGradientBoostQSModel::Classify( const CFloatVector& data, CClassificationResult& result ) const
{
	if( GetClassCount() == 2 ) {
		const double value = ensembles.First()->Predict( data );
		return classify( value * learningRate, result );
	}

	CArray<double> predictions;
	predictions.SetBufferSize( ensembles.Size() );
	for( int ensembleIndex = 0; ensembleIndex < ensembles.Size(); ensembleIndex++ ) {
		predictions.Add( ensembles[ensembleIndex]->Predict( data ) );
	}
	return classify( predictions, result );
}

bool CGradientBoostQSModel::ClassifyEx( const CSparseFloatVector& data, CArray<CClassificationResult>& results ) const
{
	return ClassifyEx( data.GetDesc(), results );
}

bool CGradientBoostQSModel::ClassifyEx( const CSparseFloatVectorDesc& data, CArray<CClassificationResult>& results ) const
{
	NeoAssert( !ensembles.IsEmpty() );

	const int classCount = GetClassCount();
	CArray<double> predictions;
	predictions.SetSize( ensembles.Size() );

	results.DeleteAll();

	for( int treeIndex = 0; treeIndex < ensembles.First()->GetTreesCount(); treeIndex++ ) {
		CClassificationResult result;
		result.ExceptionProbability = CClassificationProbability( 0 );
		if( classCount == 2 ) {
			predictions[0] = learningRate * ensembles.First()->Predict( data, treeIndex );
			classify( predictions[0], result );
		} else {
			for( int ensemblesIndex = 0; ensemblesIndex < ensembles.Size(); ensemblesIndex++ ) {
				predictions[ensemblesIndex] = ensembles[ensemblesIndex]->Predict( data, treeIndex );
			}
			classify( predictions, result );
		}

		results.Add( result );
	}

	return true;
}

void CGradientBoostQSModel::Serialize( CArchive& archive )
{
	archive.SerializeVersion( 0 );

	ensembles.Serialize( archive );
	archive.SerializeEnum( lossFunction );

	if( archive.IsStoring() ) {
		archive << learningRate;
	} else if( archive.IsLoading() ) {
		archive >> learningRate;
	} else {
		NeoAssert( false );
	}

	archive.SerializeEnum( lossFunction );
}

// Performs binary classification
bool CGradientBoostQSModel::classify( double prediction, CClassificationResult& result ) const
{
	const double prob = probability( prediction );
	result.ExceptionProbability = CClassificationProbability( 0 );
	result.PreferredClass = prob < 0.5 ? 0 : 1;
	result.Probabilities.Empty();
	result.Probabilities.Add( CClassificationProbability( 1 - prob ) );
	result.Probabilities.Add( CClassificationProbability( prob ) );
	return true;
}

// Performs multi-class classification
bool CGradientBoostQSModel::classify( CArray<double>& predictions, CClassificationResult& result ) const
{
	result.ExceptionProbability = CClassificationProbability( 0 );
	result.PreferredClass = 0;
	double sumPredictions = 0;
	for( int i = 0; i < predictions.Size(); i++ ) {
		predictions[i] = probability( predictions[i] );
		sumPredictions += predictions[i];
		if( predictions[i] > predictions[result.PreferredClass] ) {
			result.PreferredClass = i;
		}
	}

	result.Probabilities.Empty();
	for( int i = 0; i < predictions.Size(); i++ ) {
		result.Probabilities.Add( CClassificationProbability( predictions[i] / sumPredictions ) );
	}
	return true;
}

// Calculates probability from prediction
double CGradientBoostQSModel::probability( double prediction ) const
{
	if( lossFunction == CGradientBoost::LF_L2 ) {
		return 1.0f / ( 1.0f + exponentFunc( -( prediction - 0.5) ) );
	} else if( lossFunction == CGradientBoost::LF_SquaredHinge ) {
		return 1.0f / ( 1.0f + exponentFunc( prediction ) );
	}
	return 1.0f / ( 1.0f + exponentFunc( -prediction ) );
}

//------------------------------------------------------------------------------------------------------------

CPtr<IGradientBoostQSModel> CGradientBoostQuickScorer::Build( const IGradientBoostModel& model )
{
	return FINE_DEBUG_NEW CGradientBoostQSModel( model.GetEnsemble(), model.GetLossFunction(), model.GetLearningRate() );
}

CPtr<IGradientBoostQSRegressionModel> CGradientBoostQuickScorer::BuildRegression( const IGradientBoostRegressionModel& model )
{
	return FINE_DEBUG_NEW CGradientBoostQSModel( model.GetEnsemble(), model.GetLossFunction(), model.GetLearningRate() );
}

} // namespace NeoML
