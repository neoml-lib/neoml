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

#include <GradientBoostModel.h>
#include <RegressionTreeModel.h>

namespace NeoML {

REGISTER_NEOML_MODEL( CGradientBoostModel, GradientBoostModelName )

CGradientBoostModel::CGradientBoostModel( CArray<CGradientBoostEnsemble>& _ensembles, int _valueSize,
	double _learningRate, CGradientBoost::TLossFunction _lossFunction ) :
	learningRate( _learningRate ),
	lossFunction( _lossFunction ),
	valueSize( _valueSize )
{
	_ensembles.MoveTo( ensembles );
}

double CGradientBoostModel::PredictRaw( const CGradientBoostEnsemble& ensemble, int startPos, double learningRate,
	const CSparseFloatVector& vector )
{
	double result = 0;

	for( int i = startPos; i < ensemble.Size(); i++ ) {
		result += dynamic_cast<const CRegressionTreeModel*>( ensemble[i].Ptr() )->Predict( vector );
	}

	return result * learningRate;
}

double CGradientBoostModel::PredictRaw( const CGradientBoostEnsemble& ensemble, int startPos, double learningRate,
	const CSparseFloatVectorDesc& vector )
{
	double result = 0;

	for( int i = startPos; i < ensemble.Size(); i++ ) {
		result += dynamic_cast< const CRegressionTreeModel* >( ensemble[i].Ptr() )->Predict( vector );
	}

	return result * learningRate;
}


double CGradientBoostModel::PredictRaw( const CGradientBoostEnsemble& ensemble, int startPos, double learningRate,
	const CFloatVector& vector )
{
	double result = 0;

	for( int i = startPos; i < ensemble.Size(); i++ ) {
		result += dynamic_cast< const CRegressionTreeModel* >( ensemble[i].Ptr() )->Predict( vector );
	}

	return result * learningRate;
}

void CGradientBoostModel::MultivariatePredictRaw( const CGradientBoostEnsemble& ensemble, int startPos, double learningRate,
	const CSparseFloatVector& vector, CFastArray<double, 1>& predictions )
{
	::memset( predictions.GetPtr(), 0.0, predictions.Size() * sizeof( double ) );
	for( int i = startPos; i < ensemble.Size(); i++ ) {
		const CFastArray<double, 1>& pred = dynamic_cast< const CRegressionTreeModel* >( ensemble[i].Ptr() )->MultivariatePredict( vector );
		NeoAssert( predictions.Size() == pred.Size() );
		for( int j = 0; j < predictions.Size(); j++ ) {
			predictions[j] += learningRate * pred[j];
		}
	}
}

void CGradientBoostModel::MultivariatePredictRaw( const CGradientBoostEnsemble& ensemble, int startPos, double learningRate,
	const CSparseFloatVectorDesc& vector, CFastArray<double, 1>& predictions )
{
	::memset( predictions.GetPtr(), 0.0, predictions.Size() * sizeof( double ) );
	for( int i = startPos; i < ensemble.Size(); i++ ) {
		const CFastArray<double, 1>& pred = dynamic_cast< const CRegressionTreeModel* >( ensemble[i].Ptr() )->MultivariatePredict( vector );
		NeoAssert( predictions.Size() == pred.Size() );
		for( int j = 0; j < predictions.Size(); j++ ) {
			predictions[j] += learningRate * pred[j];
		}
	}
}


void CGradientBoostModel::MultivariatePredictRaw( const CGradientBoostEnsemble& ensemble, int startPos, double learningRate,
	const CFloatVector& vector, CFastArray<double, 1>& predictions )
{
	::memset( predictions.GetPtr(), 0.0, predictions.Size() * sizeof( double ) );
	for( int i = startPos; i < ensemble.Size(); i++ ) {
		const CFastArray<double, 1>& pred = dynamic_cast< const CRegressionTreeModel* >( ensemble[i].Ptr() )->MultivariatePredict( vector );
		NeoAssert( predictions.Size() == pred.Size() );
		for( int j = 0; j < predictions.Size(); j++ ) {
			predictions[j] += learningRate * pred[j];
		}
	}
}

bool CGradientBoostModel::Classify( const CSparseFloatVectorDesc& data, CClassificationResult& result ) const
{
	int classCount = GetClassCount();
	if( classCount == 2 && valueSize == 1 ) {
		return classify( PredictRaw( ensembles[0], 0, learningRate, data ), result );
	}
	
	CFastArray<double, 1> predictions;
	predictions.SetSize( classCount );
	if( valueSize > 1 ) {
		MultivariatePredictRaw( ensembles[0], 0, learningRate, data, predictions );
	} else {
		for( int i = 0; i < ensembles.Size(); i++ ) {
			predictions[i] = PredictRaw( ensembles[i], 0, learningRate, data );
		}
	}

	return classify( predictions, result );
}

bool CGradientBoostModel::Classify( const CFloatVector& data, CClassificationResult& result ) const
{
	int classCount = GetClassCount();
	if( classCount == 2 && valueSize == 1 ) {
		return classify( PredictRaw( ensembles[0], 0, learningRate, data ), result );
	}

	CFastArray<double, 1> predictions;
	predictions.SetSize( classCount );
	if( valueSize > 1 ) {
		MultivariatePredictRaw( ensembles[0], 0, learningRate, data, predictions );
	} else {
		for( int i = 0; i < ensembles.Size(); i++ ) {
			predictions[i] = PredictRaw( ensembles[i], 0, learningRate, data );
		}
	}

	return classify( predictions, result );
}

void CGradientBoostModel::Serialize( CArchive& archive )
{
#ifdef NEOML_USE_FINEOBJ
	const int minSupportedVersion = 0;
#else
	const int minSupportedVersion = 2;
#endif
	int version = archive.SerializeVersion( 3, minSupportedVersion );

	if( archive.IsStoring() ) {
		archive << ensembles.Size();
		for( int i = 0; i < ensembles.Size(); i++ ) {
			CGradientBoostEnsemble& ensemble = ensembles[i];
			archive << ensemble.Size();
			for( int j = 0; j < ensemble.Size(); j++ ) {
				CString modelName = CString( GetModelName( ensemble[j] ) );
				archive << modelName;
				ensemble[j]->Serialize( archive );
			}
		}
		archive << learningRate;
		archive.SerializeEnum( lossFunction );
		archive << valueSize;
	} else if( archive.IsLoading() ) {
		int size = 0;
		archive >> size;
		ensembles.SetSize( size );
		for( int i = 0; i < ensembles.Size(); i++ ) {
			CGradientBoostEnsemble& ensemble = ensembles[i];
			archive >> size;
			ensemble.SetSize( size );
			for( int j = 0; j < ensemble.Size(); j++ ) {
#ifdef NEOML_USE_FINEOBJ
				if( version < 2 ) {
					CUnicodeString modelName = archive.ReadExternalName();
					ensemble[j] = CreateModel<IRegressionTreeModel>( modelName.CreateString() );
				}	
#endif
				if( version >= 2 ) {
					CString modelName;
					archive >> modelName;
					ensemble[j] = CreateModel<IRegressionTreeModel>( modelName );
				}

				ensemble[j]->Serialize( archive );
			}
		}
		archive >> learningRate;
		if( version > 0 ) {
			archive.SerializeEnum( lossFunction );
			if( version >= 3 ) {
				archive >> valueSize;
			} else {
				valueSize = 1;
			}
		}
	} else {
		NeoAssert( false );
	}
}

bool CGradientBoostModel::ClassifyEx( const CSparseFloatVector& data, CArray<CClassificationResult>& results ) const
{
	return ClassifyEx( data.GetDesc(), results );
}

bool CGradientBoostModel::ClassifyEx( const CSparseFloatVectorDesc& data, CArray<CClassificationResult>& results ) const
{
	NeoAssert( !ensembles.IsEmpty() );

	const int classCount = GetClassCount();
	CArray<double> predictions;
	predictions.Add( 0.0, classCount );
	CArray<double> distances;

	results.DeleteAll();
	for( int resultIndex = 0; resultIndex < ensembles[0].Size(); ++resultIndex ) {
		CClassificationResult result;
		result.ExceptionProbability = CClassificationProbability( 0 );
		
		if( classCount == 2 ) {
			predictions[0] += learningRate * dynamic_cast<const CRegressionTreeModel*>( ensembles[0][resultIndex].Ptr() )->Predict( data );
			const double rawValue = probability( predictions[0] );
			result.PreferredClass = rawValue < 0.5 ? 0 : 1;
			result.Probabilities.Add( CClassificationProbability( 1 - rawValue ) );
			result.Probabilities.Add( CClassificationProbability( rawValue ) );
		} else {
			double sumDistance = 0;
			distances.DeleteAll();
			distances.SetBufferSize( classCount );
			result.PreferredClass = 0;

			if( valueSize > 1 ){
				const CFastArray<double, 1>& pred = dynamic_cast< const CRegressionTreeModel* >( ensembles[0][resultIndex].Ptr() )->MultivariatePredict( data );
				for( int i = 0; i < pred.Size(); i++ ) {
					predictions[i] += learningRate * pred[i];
				}
			} else {
				for( int i = 0; i < ensembles.Size(); i++ ) {
					predictions[i] += learningRate * dynamic_cast< const CRegressionTreeModel* >( ensembles[i][resultIndex].Ptr() )->Predict( data );
				}
			}

			for( int i = 0; i < classCount; i++ ) {
				const double distance = probability( predictions[i] );
				distances.Add( distance );
				sumDistance += distance;
				if( distance > distances[result.PreferredClass] ) {
					result.PreferredClass = i;
				}
			}

			for( int i = 0; i < classCount; i++ ) {
				result.Probabilities.Add( CClassificationProbability( distances[i] / sumDistance ) );
			}
		}

		results.Add( result );
	}

	return true;
}

void CGradientBoostModel::CalcFeatureStatistics( int maxFeature, CArray<int>& result ) const
{
	NeoAssert( maxFeature > 0 );
	result.Empty();
	result.Add( 0, maxFeature );

	for( int i = 0; i < ensembles.Size(); i++ ) {
		const CGradientBoostEnsemble& ensemble = ensembles[i];
		for( int j = 0; j < ensemble.Size(); j++ ) {
			CArray<int> oneTreeResult;
			CheckCast<CRegressionTreeModel>(ensemble[j].Ptr())->CalcFeatureStatistics( maxFeature, oneTreeResult );
			for( int k = 0; k < result.Size(); k++ ) {
				result[k] += oneTreeResult[k];
			}
		}
	}
}

void CGradientBoostModel::CutNumberOfTrees( int numberOfTrees )
{
	NeoAssert( numberOfTrees >= 0 );

	for( int i = 0; i < ensembles.Size(); i++ ) {
		if( ensembles[i].Size() > numberOfTrees ) {
			ensembles[i].SetSize( numberOfTrees );
		}
	}
}

// IRegressionModel interface methods

double CGradientBoostModel::Predict( const CSparseFloatVector& data ) const
{
	return PredictRaw( ensembles.First(), 0, learningRate, data );
}

double CGradientBoostModel::Predict( const CFloatVector& data ) const
{
	return PredictRaw( ensembles.First(), 0, learningRate, data );
}

double CGradientBoostModel::Predict( const CSparseFloatVectorDesc& data ) const
{
	return PredictRaw( ensembles.First(), 0, learningRate, data );
}

// The common implementation for the three MultivariatePredict method variations
template<typename TData>
CFloatVector CGradientBoostModel::doMultivariatePredict( const TData& data ) const
{
	if( valueSize > 1 ){
		CFastArray<double, 1> predictions;
		predictions.Add( 0.0, valueSize );
		MultivariatePredictRaw( ensembles[0], 0, learningRate, data, predictions );
		CFloatVector result( valueSize );
		float* resultPtr = result.CopyOnWrite();
		for( int i = 0; i < valueSize; i++ ) {
			resultPtr[i] = predictions[i];
		}
		return result;
	} else {
		CFloatVector result( ensembles.Size() );
		float* resultPtr = result.CopyOnWrite();
		for( int i = 0; i < ensembles.Size(); i++ ) {
			resultPtr[i] = PredictRaw( ensembles[i], 0, learningRate, data );
		}
		return result;
	}
}

// IMultivariateRegressionModel interface methods

CFloatVector CGradientBoostModel::MultivariatePredict( const CSparseFloatVector& data ) const
{
	return doMultivariatePredict( data );
}

CFloatVector CGradientBoostModel::MultivariatePredict( const CFloatVector& data ) const
{
	return doMultivariatePredict( data );
}

// Performs classification
bool CGradientBoostModel::classify( double prediction, CClassificationResult& result ) const
{
	double prob = probability( prediction );
	result.ExceptionProbability = CClassificationProbability( 0 );
	result.PreferredClass = prob < 0.5 ? 0 : 1;
	result.Probabilities.Empty();
	result.Probabilities.Add( CClassificationProbability( 1 - prob ) );
	result.Probabilities.Add( CClassificationProbability( prob ) );
	return true;
}

// Performs classification
bool CGradientBoostModel::classify( CFastArray<double, 1>& predictions, CClassificationResult& result ) const
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

#define DBL_LOG_MAX 709.
#define DBL_LOG_MIN -709.

// An exponent function with limitations to avoid NaN
inline double exponentFunc( double f )
{
	if( f < DBL_LOG_MIN ) {
		return 0;
	} else if( f > DBL_LOG_MAX ) {
		return DBL_MAX;
	} else {
		return exp( f );
	}
}

// Gets the probability from the prediction
double CGradientBoostModel::probability( double prediction ) const
{
	if( lossFunction == CGradientBoost::LF_L2 ) {
		return 1.0f / ( 1.0f + exponentFunc( -( prediction - 0.5) ) );
	} else if( lossFunction == CGradientBoost::LF_SquaredHinge ) {
		return 1.0f / ( 1.0f + exponentFunc( prediction ) );
	}
	return 1.0f / ( 1.0f + exponentFunc( -prediction ) );
}

} // namespace NeoML
