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

namespace NeoML {

// The model trained using gradient boosting
class CGradientBoostModel : public IGradientBoostModel, public IGradientBoostRegressionModel {
public:
	CGradientBoostModel() : learningRate( 0 ), lossFunction( CGradientBoost::LF_Undefined ) {}
	CGradientBoostModel( CArray<CGradientBoostEnsemble>& models, double learningRate,
		CGradientBoost::TLossFunction lossFunction );

	// Used for serialization
	static CPtr<IModel> Create() { return FINE_DEBUG_NEW CGradientBoostModel(); }

	// Gets the prediction by the tree ensemble
	static double PredictRaw( const CGradientBoostEnsemble& models, int startPos, double learningRate,
		const CSparseFloatVector& vector );
	static double PredictRaw( const CGradientBoostEnsemble& models, int startPos, double learningRate,
		const CFloatVector& vector );
	static double PredictRaw( const CGradientBoostEnsemble& models, int startPos, double learningRate,
		const CSparseFloatVectorDesc& desc );

	// IModel interface methods
	int GetClassCount() const override { return ensembles.Size() == 1 ? 2 : ensembles.Size(); }
	bool Classify( const CSparseFloatVectorDesc& data, CClassificationResult& result ) const override;
	bool Classify( const CFloatVector& data, CClassificationResult& result ) const override;
	void Serialize( CArchive& archive ) override;

	// IGradientBoostModel inteface methods
	const CArray<CGradientBoostEnsemble>& GetEnsemble() const override { return ensembles; }
	double GetLearningRate() const override { return learningRate; }
	CGradientBoost::TLossFunction GetLossFunction() const override { return lossFunction; }
	bool ClassifyEx( const CSparseFloatVector& data, CArray<CClassificationResult>& results ) const override;
	bool ClassifyEx( const CSparseFloatVectorDesc& data, CArray<CClassificationResult>& results ) const override;
	void CalcFeatureStatistics( int maxFeature, CArray<int>& result ) const override;
	void CutNumberOfTrees( int numberOfTrees ) override;

	// IRegressionModel interface methods
	double Predict( const CSparseFloatVector& data ) const override;
	double Predict( const CFloatVector& data ) const override;
	double Predict( const CSparseFloatVectorDesc& data ) const override;

	// IMultivariateRegressionModel interface methods
	CFloatVector MultivariatePredict( const CSparseFloatVector& data ) const override;
	CFloatVector MultivariatePredict( const CFloatVector& data ) const override;

private:
	CArray<CGradientBoostEnsemble> ensembles; // the models
	double learningRate; // the coefficient for each of the models
	CGradientBoost::TLossFunction lossFunction; // the loss function to be optimized

	bool classify( double prediction, CClassificationResult& result ) const;
	bool classify( CArray<double>& predictions, CClassificationResult& result ) const;
	double probability( double prediction ) const;

	// The common implementation for all three MultivariatePredict methods
	template<typename TData>
	CFloatVector doMultivariatePredict( const TData& data ) const;
};

} // namespace NeoML
