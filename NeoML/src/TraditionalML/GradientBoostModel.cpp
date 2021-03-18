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
#include <CompactRegressionTree.h>

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

bool CGradientBoostModel::Classify( const CFloatVectorDesc& data, CClassificationResult& result ) const
{
	CFastArray<double, 1> predictions;

	if( ensembles.Size() > 1 ) {
		predictions.SetSize( ensembles.Size() );
		CFastArray<double, 1> ensemblePredictions;
		ensemblePredictions.SetSize(1);
		for( int i = 0; i < ensembles.Size(); i++ ) {
			PredictRaw(ensembles[i], 0, learningRate, data, ensemblePredictions);
			predictions[i] = ensemblePredictions[0];
		}
	} else {
		predictions.SetSize( valueSize );
		PredictRaw( ensembles[0], 0, learningRate, data, predictions );
	}

	return classify( predictions, result );
}

bool CGradientBoostModel::Classify( const CFloatVector& data, CClassificationResult& result ) const
{
	CFastArray<double, 1> predictions;

	if( ensembles.Size() > 1 ) {
		predictions.SetSize( ensembles.Size() );
		CFastArray<double, 1> ensemblePredictions;
		ensemblePredictions.SetSize(1);
		for( int i = 0; i < ensembles.Size(); i++ ) {
			PredictRaw(ensembles[i], 0, learningRate, data, ensemblePredictions);
			predictions[i] = ensemblePredictions[0];
		}
	} else {
		predictions.SetSize( valueSize );
		PredictRaw( ensembles[0], 0, learningRate, data, predictions );
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
					ensemble[j] = CreateModel<IRegressionTreeNode>( modelName.CreateString() );
				}	
#endif
				if( version >= 2 ) {
					CString modelName;
					archive >> modelName;
					ensemble[j] = CreateModel<IRegressionTreeNode>( modelName );
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

bool CGradientBoostModel::ClassifyEx( const CFloatVectorDesc& data, CArray<CClassificationResult>& results ) const
{
	NeoAssert( !ensembles.IsEmpty() );

	CFastArray<double, 1> predictions;
	predictions.Add(0.0, ensembles.Size() > 1 ? ensembles.Size() : valueSize);
	CFastArray<double, 1> curPredictions;

	CRegressionTree::CPrediction pred;
	results.DeleteAll();
	for( int resultIndex = 0; resultIndex < ensembles[0].Size(); ++resultIndex ) {
		CClassificationResult result;

		if( ensembles.Size() > 1 ) {
			NeoAssert(predictions.Size() == ensembles.Size());
			for( int i = 0; i < ensembles.Size(); i++ ) {
				static_cast<const CRegressionTree*>( ensembles[i][resultIndex].Ptr() )->Predict( data, pred );
				predictions[i] += learningRate * pred[0];
			}
		} else {
			static_cast<const CRegressionTree*>( ensembles[0][resultIndex].Ptr() )->Predict( data, pred );
			NeoAssert(predictions.Size() == pred.Size());
			for( int i = 0; i < predictions.Size(); i++ ) {
				predictions[i] += learningRate * pred[i];
			}
		}

		predictions.CopyTo(curPredictions);
		classify(curPredictions, result);

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
			static_cast<const CRegressionTree*>( ensemble[j].Ptr() )
				->CalcFeatureStatistics( maxFeature, oneTreeResult );
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

void CGradientBoostModel::ConvertToCompact()
{
	for( int i = 0; i < ensembles.Size() ; i++ ) {
		CGradientBoostEnsemble& ensemble = ensembles[i];
		for( int j = 0; j < ensemble.Size(); j++ ) {
			CPtr<IRegressionTreeNode>& tree = ensemble[j];
			if( dynamic_cast<CCompactRegressionTree*>( tree.Ptr() ) == 0 ) {
				tree = FINE_DEBUG_NEW CCompactRegressionTree( tree );
			}
		}
	}
}

// The common implementation for Predict methods
template<typename TData>
inline double CGradientBoostModel::doPredict( const TData& data ) const
{
	NeoAssert(ensembles.Size() == 1 && valueSize == 1);
	CFastArray<double, 1> predictions;
	predictions.SetSize(1);
	PredictRaw( ensembles.First(), 0, learningRate, data, predictions );
	return predictions[0];
}

double CGradientBoostModel::Predict( const CSparseFloatVector& data ) const
{
	return doPredict( data.GetDesc() );
}

double CGradientBoostModel::Predict( const CFloatVector& data ) const
{
	return doPredict( data );
}

double CGradientBoostModel::Predict( const CFloatVectorDesc& data ) const
{
	return doPredict( data );
}

// The common implementation for the three MultivariatePredict method variations
template<typename TData>
CFloatVector CGradientBoostModel::doMultivariatePredict( const TData& data ) const
{
	CFastArray<double, 1> predictions;

	if( ensembles.Size() == 1 ){		
		predictions.Add( 0.0, valueSize );
		PredictRaw( ensembles[0], 0, learningRate, data, predictions );
		CFloatVector result( valueSize );
		float* resultPtr = result.CopyOnWrite();
		for( int i = 0; i < valueSize; i++ ) {
			resultPtr[i] = static_cast<float>( predictions[i] );
		}
		return result;
	}
	
	predictions.Add( 0.0, 1 );
	CFloatVector result( ensembles.Size() );
	float* resultPtr = result.CopyOnWrite();
	for( int i = 0; i < ensembles.Size(); i++ ) {
		PredictRaw(ensembles[i], 0, learningRate, data, predictions);
		resultPtr[i] = static_cast<float>(predictions[0]);
	}
	return result;
}

// IMultivariateRegressionModel interface methods
CFloatVector CGradientBoostModel::MultivariatePredict( const CFloatVectorDesc& data ) const
{
	return doMultivariatePredict( data );
}

// Performs classification
bool CGradientBoostModel::classify( CFastArray<double, 1>& predictions, CClassificationResult& result ) const
{
	NeoAssert( !predictions.IsEmpty() );

	if( predictions.Size() == 1 ) {
		double prob = probability( predictions[0] );
		result.ExceptionProbability = CClassificationProbability( 0 );
		result.PreferredClass = prob < 0.5 ? 0 : 1;
		result.Probabilities.Empty();
		result.Probabilities.Add( CClassificationProbability( 1 - prob ) );
		result.Probabilities.Add( CClassificationProbability( prob ) );
		return true;
	}

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
