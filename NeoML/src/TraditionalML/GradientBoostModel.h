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

#include <NeoML/TraditionalML/GradientBoost.h>
#include <RegressionTree.h>

namespace NeoML {

// The model trained using gradient boosting
class CGradientBoostModel : public IGradientBoostModel, public IGradientBoostRegressionModel {
public:
	CGradientBoostModel() : learningRate( 0 ), lossFunction( CGradientBoost::LF_Undefined ), valueSize( 1 ) {}
	CGradientBoostModel( CArray<CGradientBoostEnsemble>& models, int valueSize, double learningRate,
		CGradientBoost::TLossFunction lossFunction );

	// Used for serialization
	static CPtr<IModel> Create() { return FINE_DEBUG_NEW CGradientBoostModel(); }

	// Gets the prediction by the tree ensemble
	template<typename TFeatures>
	static void PredictRaw(
		const CGradientBoostEnsemble& models, int startPos, double learningRate,
		const TFeatures& features, CFastArray<double, 1>& predictions );

	// IModel interface methods
	int GetClassCount() const override { return ( valueSize == 1 && ensembles.Size() == 1 ) ? 2 : valueSize * ensembles.Size(); }
	bool Classify( const CFloatVectorDesc& data, CClassificationResult& result ) const override;
	void Serialize( CArchive& archive ) override;

	// IGradientBoostModel inteface methods
	const CArray<CGradientBoostEnsemble>& GetEnsemble() const override { return ensembles; }
	double GetLearningRate() const override { return learningRate; }
	CGradientBoost::TLossFunction GetLossFunction() const override { return lossFunction; }
	bool ClassifyEx( const CSparseFloatVector& data, CArray<CClassificationResult>& results ) const override;
	bool ClassifyEx( const CFloatVectorDesc& data, CArray<CClassificationResult>& results ) const override;
	void CalcFeatureStatistics( int maxFeature, CArray<int>& result ) const override;
	void CutNumberOfTrees( int numberOfTrees ) override;
	virtual void ConvertToCompact() override;

	// IRegressionModel interface methods
	double Predict( const CFloatVectorDesc& data ) const override;

	// IMultivariateRegressionModel interface methods
	CFloatVector MultivariatePredict( const CFloatVectorDesc& data ) const override;

private:
	CArray<CGradientBoostEnsemble> ensembles; // the models
	double learningRate; // the coefficient for each of the models
	CGradientBoost::TLossFunction lossFunction; // the loss function to be optimized
	int valueSize; // the value size of each model, if valueSize > 1 then ensemble consists of multiclass trees

	bool classify( CFastArray<double, 1>& predictions, CClassificationResult& result ) const;
	double probability( double prediction ) const;

	// The common implementation for Predict methods
	template<typename TData>
	double doPredict( const TData& data ) const;
	// The common implementation for MultivariatePredict methods
	template<typename TData>
	CFloatVector doMultivariatePredict( const TData& data ) const;
};

/////////////////////////////////////////////////////////////////////////////////////////

template<typename TFeatures>
void CGradientBoostModel::PredictRaw(
	const CGradientBoostEnsemble& ensemble, int startPos, double learningRate,
	const TFeatures& features, CFastArray<double, 1>& predictions )
{
	const int predictionSize = predictions.Size();
	predictions.Empty();

	if( predictionSize == 1 ) {
		double prediction = 0;
		for( int i = startPos; i < ensemble.Size(); i++ ) {
			prediction +=
				static_cast<const CRegressionTree*>( ensemble[i].Ptr() )->Predict( features );
		}
		predictions.Add( prediction * learningRate );
	} else {
		CRegressionTree::CPrediction pred;
		predictions.Add(0.0, predictionSize);
		for( int i = startPos; i < ensemble.Size(); i++ ) {
			static_cast<const CRegressionTree*>( ensemble[i].Ptr() )->Predict( features, pred );
			NeoPresume( predictionSize == pred.Size() );
			for( int j = 0; j < predictionSize; j++ ) {
				predictions[j] += pred[j];
			}
		}
		for( int j = 0; j < predictionSize; j++ ) {
			predictions[j] *= learningRate;
		}
	}
}

/////////////////////////////////////////////////////////////////////////////////////////

} // namespace NeoML
