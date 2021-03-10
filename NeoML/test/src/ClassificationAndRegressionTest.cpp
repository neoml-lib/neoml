/* Copyright © 2021 ABBYY Production LLC

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

#include <TestFixture.h>
#include <RandomProblem.h>

using namespace NeoML;
using namespace NeoMLTest;

//---------------------------------------------------------------------------------------------------------------------
// Common functions

void TestClassificationResult( const IModel* modelDense, const IModel* modelSparse,
	const CClassificationRandomProblem* testDataDense, const CClassificationRandomProblem* testDataSparse )
{
	for( int i = 0; i < testDataSparse->GetVectorCount(); i++ ) {
		CClassificationResult result1;
		CClassificationResult result2;
		CClassificationResult result4;
		CClassificationResult result3;

		ASSERT_TRUE( modelDense->Classify( testDataDense->GetVector( i ), result1 ) );
		ASSERT_TRUE( modelDense->Classify( testDataSparse->GetVector( i ), result2 ) );
		ASSERT_TRUE( modelSparse->Classify( testDataDense->GetVector( i ), result3 ) );
		ASSERT_TRUE( modelSparse->Classify( testDataSparse->GetVector( i ), result4 ) );

		ASSERT_EQ( result1.PreferredClass, result2.PreferredClass );
		ASSERT_EQ( result1.PreferredClass, result3.PreferredClass );
		ASSERT_EQ( result1.PreferredClass, result4.PreferredClass );
	}
}

void CrossValidate( int PartsCount, ITrainingModel& trainingModel, const IProblem* dense, const IProblem* sparse )
{
	CCrossValidation CrossValidation( trainingModel, dense );
	CCrossValidationResult result;

	int begin = GetTickCount();
	CrossValidation.Execute( PartsCount, AccuracyScore, result, true );
	GTEST_LOG_( INFO ) << "Dense execution time: " << GetTickCount() - begin;

	ASSERT_EQ( result.Models.Size(), PartsCount );
	ASSERT_EQ( result.Success.Size(), PartsCount );
	ASSERT_EQ( result.Results.Size(), dense->GetVectorCount() );
	CCrossValidation CrossValidationSparse( trainingModel, sparse );

	CCrossValidationResult result2;
	begin = GetTickCount();
	CrossValidationSparse.Execute( PartsCount, AccuracyScore, result2, true );
	GTEST_LOG_( INFO ) << "Sparse execution time: " << GetTickCount() - begin;

	ASSERT_EQ( result2.Models.Size(), PartsCount );
	ASSERT_EQ( result2.Success.Size(), PartsCount );
	ASSERT_EQ( result2.Results.Size(), sparse->GetVectorCount() );

	for( int i = 0; i < PartsCount; ++i ) {
		ASSERT_EQ( result.Success[i], result2.Success[i] );
	}
}

template<class TModel>
void TrainGB( const CGradientBoost::CParams& params, const IProblem& denseProblem, const IProblem& sparseProblem,
	CPtr<TModel>& modelDense, CPtr<TModel>& modelSparse )
{
	GTEST_LOG_( INFO ) << "Random = " << params.Random->Next();
	CGradientBoost boosting( params );
	int begin = GetTickCount();
	modelDense = boosting.TrainModel<TModel>( denseProblem );
	GTEST_LOG_( INFO ) << "Dense train time: " << GetTickCount() - begin;
	GTEST_LOG_( INFO ) << "The last loss: " << boosting.GetLastLossMean();
	ASSERT_TRUE( modelDense != nullptr );

	params.Random->Reset( 0 );
	GTEST_LOG_( INFO ) << "Random = " << params.Random->Next();
	begin = GetTickCount();
	modelSparse = boosting.TrainModel<TModel>( sparseProblem );
	GTEST_LOG_( INFO ) << "Sparse train time: " << GetTickCount() - begin;
	GTEST_LOG_( INFO ) << "The last loss: " << boosting.GetLastLossMean();
	ASSERT_TRUE( modelSparse != nullptr );
}

//---------------------------------------------------------------------------------------------------------------------
// TestFixures for multi and binary classification with sparse and dense training

// Binary
class RandomBinaryClassification4000x20 : public CNeoMLTestFixture {
protected:
	virtual void SetUp();

	// this methods will create common instanses of each dataset ( not thread-safe! )
	CClassificationRandomProblem* getDenseRandomBinaryProblem( CRandom& );
	CClassificationRandomProblem* getDenseBinaryTestData( CRandom& );
	CClassificationRandomProblem* getSparseRandomBinaryProblem();
	CClassificationRandomProblem* getSparseBinaryTestData();

	// binary datasets pointers
	CClassificationRandomProblem* DenseRandomBinaryProblem;
	CClassificationRandomProblem* DenseBinaryTestData;
	CClassificationRandomProblem* SparseRandomBinaryProblem;
	CClassificationRandomProblem* SparseBinaryTestData;

	void TestBinaryClassificationResult( const IModel* modelDense, const IModel* modelSparse ) const
		{ TestClassificationResult( modelDense, modelSparse, DenseBinaryTestData, SparseBinaryTestData ); }
};

CClassificationRandomProblem* RandomBinaryClassification4000x20::getDenseRandomBinaryProblem( CRandom& rand )
{
	static auto denseRandomBinaryProblem = CClassificationRandomProblem::Random( rand, 4000, 20, 2 );
	return denseRandomBinaryProblem.Ptr();
}

CClassificationRandomProblem* RandomBinaryClassification4000x20::getDenseBinaryTestData( CRandom& rand )
{
	static auto denseBinaryTestData = CClassificationRandomProblem::Random( rand, 1000, 20, 2 );
	return denseBinaryTestData.Ptr();
}

CClassificationRandomProblem* RandomBinaryClassification4000x20::getSparseRandomBinaryProblem()
{
	static auto sparseRandomBinaryProblem = DenseRandomBinaryProblem->CreateSparse();
	return sparseRandomBinaryProblem.Ptr();
}

CClassificationRandomProblem* RandomBinaryClassification4000x20::getSparseBinaryTestData()
{
	static auto sparseBinaryTestData = DenseBinaryTestData->CreateSparse();
	return sparseBinaryTestData.Ptr();
}

void RandomBinaryClassification4000x20::SetUp()
{
	static CRandom rand( 0 );
	DenseRandomBinaryProblem = getDenseRandomBinaryProblem( rand );
	DenseBinaryTestData = getDenseBinaryTestData( rand );
	SparseRandomBinaryProblem = getSparseRandomBinaryProblem();
	SparseBinaryTestData = getSparseBinaryTestData();
}

class RandomBinaryRegression4000x20 : public CNeoMLTestFixture {
};

// use classification problems to create CMultivatieateRegressionOverBinaryClassification in GradientBoost
class RandomBinaryGBRegression4000x20 : public RandomBinaryClassification4000x20 {
protected:
	CPtr<IRegressionModel> ModelDense;
	CPtr<IRegressionModel> ModelSparse;

	void TrainBinaryGradientBoost( const CGradientBoost::CParams& params )
		{ TrainGB( params, *DenseRandomBinaryProblem, *SparseRandomBinaryProblem, ModelDense, ModelSparse ); }

	void TestBinaryRegressionResult() const
	{
		for( int i = 0; i < SparseBinaryTestData->GetVectorCount(); i++ ) {
			double result1 = ModelDense->Predict( DenseBinaryTestData->GetVector( i ) );
			double result2 = ModelDense->Predict( SparseBinaryTestData->GetVector( i ) );
			double result4 = ModelSparse->Predict( DenseBinaryTestData->GetVector( i ) );
			double result3 = ModelSparse->Predict( SparseBinaryTestData->GetVector( i ) );

			ASSERT_DOUBLE_EQ( result1, result2 );
			ASSERT_DOUBLE_EQ( result1, result3 );
			ASSERT_DOUBLE_EQ( result1, result4 );
		}
	}
};

// Multi 
class RandomMultiClassification2000x20 : public CNeoMLTestFixture {
protected:
	virtual void SetUp();

	// this methods will create common instanses of each dataset ( not thread-safe! )
	CClassificationRandomProblem* getDenseRandomMultiProblem( CRandom& );
	CClassificationRandomProblem* getDenseMultiTestData( CRandom& );
	CClassificationRandomProblem* getSparseRandomMultiProblem();
	CClassificationRandomProblem* getSparseMultiTestData();

	// multi-class datasets pointers
	CClassificationRandomProblem* DenseRandomMultiProblem;
	CClassificationRandomProblem* DenseMultiTestData;
	CClassificationRandomProblem* SparseRandomMultiProblem;
	CClassificationRandomProblem* SparseMultiTestData;

	CPtr<IModel> ModelDenseGB;
	CPtr<IModel> ModelSparseGB;

	void TrainMultiGradientBoost( const CGradientBoost::CParams& params )
		{ TrainGB( params, *DenseRandomMultiProblem, *SparseRandomMultiProblem, ModelDenseGB, ModelSparseGB ); }

	void TestMultiClassificationResult( const IModel* modelDense, const IModel* modelSparse ) const
		{ TestClassificationResult( modelDense, modelSparse, DenseMultiTestData, SparseMultiTestData ); }
};

CClassificationRandomProblem* RandomMultiClassification2000x20::getDenseRandomMultiProblem( CRandom& rand )
{
	static auto denseRandomMultiProblem = CClassificationRandomProblem::Random( rand, 2000, 20, 10 );
	return denseRandomMultiProblem.Ptr();
}

CClassificationRandomProblem* RandomMultiClassification2000x20::getDenseMultiTestData( CRandom& rand )
{
	static auto denseMultiTestData = CClassificationRandomProblem::Random( rand, 500, 20, 10 );
	return denseMultiTestData.Ptr();
}

CClassificationRandomProblem* RandomMultiClassification2000x20::getSparseRandomMultiProblem()
{
	static auto sparseRandomMultiProblem = DenseRandomMultiProblem->CreateSparse();
	return sparseRandomMultiProblem.Ptr();
}

CClassificationRandomProblem* RandomMultiClassification2000x20::getSparseMultiTestData()
{
	static auto sparseMultiTestData = DenseMultiTestData->CreateSparse();
	return sparseMultiTestData.Ptr();
}

void RandomMultiClassification2000x20::SetUp()
{
	static CRandom rand( 0 );
	DenseRandomMultiProblem = getDenseRandomMultiProblem( rand );
	DenseMultiTestData = getDenseMultiTestData( rand );
	SparseRandomMultiProblem = getSparseRandomMultiProblem();
	SparseMultiTestData = getSparseMultiTestData();
}

// use classification problems to create CMultivatieateRegressionOverClassification in GradientBoost
class RandomMultiGBRegression2000x20 : public RandomMultiClassification2000x20 {
protected:
	CPtr<IMultivariateRegressionModel> ModelDense;
	CPtr<IMultivariateRegressionModel> ModelSparse;

	void TrainMultiGradientBoost( const CGradientBoost::CParams& params )
		{ TrainGB( params, *DenseRandomMultiProblem, *SparseRandomMultiProblem, ModelDense, ModelSparse ); }

	void TestMultiRegressionResult() const
	{
		auto cmpFloatVectors = []( const CSparseFloatVectorDesc& v1, const CSparseFloatVectorDesc& v2 ) {
			ASSERT_EQ( v1.Size, v2.Size );
			ASSERT_EQ( ::memcmp( v1.Values, v2.Values, v2.Size * sizeof( float ) ), 0 );
			ASSERT_EQ( ::memcmp( v1.Indexes, v2.Indexes, ( v2.Indexes == nullptr ? 0 : v2.Size )*sizeof( float ) ), 0 );
		};

		for( int i = 0; i < SparseMultiTestData->GetVectorCount(); i++ ) {
			CFloatVector result1 = ModelDense->MultivariatePredict( DenseMultiTestData->GetVector( i ) );
			CFloatVector result2 = ModelDense->MultivariatePredict( SparseMultiTestData->GetVector( i ) );
			CFloatVector result4 = ModelSparse->MultivariatePredict( DenseMultiTestData->GetVector( i ) );
			CFloatVector result3 = ModelSparse->MultivariatePredict( SparseMultiTestData->GetVector( i ) );

			cmpFloatVectors( result1.GetDesc(), result2.GetDesc() );
			cmpFloatVectors( result1.GetDesc(), result3.GetDesc() );
			cmpFloatVectors( result1.GetDesc(), result4.GetDesc() );
		}
	}
};

//---------------------------------------------------------------------------------------------------------------------
// Tests

TEST_F( RandomBinaryClassification4000x20, Linear )
{
	CLinearBinaryClassifierBuilder::CParams params( EF_SquaredHinge );
	params.L1Coeff = 0.05f;
	CLinearBinaryClassifierBuilder linear( params );

	int begin = GetTickCount();
	auto model = linear.Train( *DenseRandomBinaryProblem );
	GTEST_LOG_( INFO ) << "Dense train time: " << GetTickCount() - begin;
	ASSERT_TRUE( model != nullptr );

	begin = GetTickCount();
	auto model2 = linear.Train( *SparseRandomBinaryProblem );
	GTEST_LOG_( INFO ) << "Sparse train time: " << GetTickCount() - begin;
	ASSERT_TRUE( model2 != nullptr );

	TestBinaryClassificationResult( model, model2 );
}

TEST_F( RandomBinaryClassification4000x20, SvmLinear )
{
	CSvmBinaryClassifierBuilder::CParams params( CSvmKernel::KT_Linear );
	CSvmBinaryClassifierBuilder svmLinear( params );

	int begin = GetTickCount();
	auto model = svmLinear.Train( *DenseRandomBinaryProblem );
	GTEST_LOG_( INFO ) << "dense train time: " << GetTickCount() - begin;
	ASSERT_TRUE( model != nullptr );

	begin = GetTickCount();
	auto model2 = svmLinear.Train( *SparseRandomBinaryProblem );
	GTEST_LOG_( INFO ) << "sparse train time: " << GetTickCount() - begin;
	ASSERT_TRUE( model2 != nullptr );

	TestBinaryClassificationResult( model, model2 );
}

TEST_F( RandomBinaryClassification4000x20, SvmRbf )
{
	CSvmBinaryClassifierBuilder::CParams params( CSvmKernel::KT_RBF );
	CSvmBinaryClassifierBuilder svmRbf( params );

	int begin = GetTickCount();
	auto model = svmRbf.Train( *DenseRandomBinaryProblem );
	GTEST_LOG_( INFO ) << "Dense train time: " << GetTickCount() - begin;
	ASSERT_TRUE( model != nullptr );

	begin = GetTickCount();
	auto model2 = svmRbf.Train( *SparseRandomBinaryProblem );
	GTEST_LOG_( INFO ) << "Sparse train time: " << GetTickCount() - begin;
	ASSERT_TRUE( model2 != nullptr );

	TestBinaryClassificationResult( model, model2 );
}

TEST_F( RandomBinaryClassification4000x20, DecisionTree )
{
	CDecisionTreeTrainingModel::CParams param;
	CDecisionTreeTrainingModel decisionTree( param );

	int begin = GetTickCount();
	auto model = decisionTree.TrainModel<IDecisionTreeModel>( *DenseRandomBinaryProblem );
	GTEST_LOG_( INFO ) << "Dense train time: " << GetTickCount() - begin;
	ASSERT_TRUE( model != nullptr );

	begin = GetTickCount();
	auto model2 = decisionTree.Train( *SparseRandomBinaryProblem );
	GTEST_LOG_( INFO ) << "Sparse train time: " << GetTickCount() - begin;
	ASSERT_TRUE( model2 != nullptr );

	TestBinaryClassificationResult( model, model2 );
}

TEST_F( RandomMultiClassification2000x20, GBTB_Full )
{
	CRandom random( 0 );
	CGradientBoost::CParams params;
	params.Random = &random;
	params.IterationsCount = 10;
	TrainMultiGradientBoost( params );
	TestMultiClassificationResult( ModelDenseGB, ModelSparseGB );
}

TEST_F( RandomMultiClassification2000x20, GBTB_FastHist )
{
	CRandom random( 0 );
	CGradientBoost::CParams params;
	params.Random = &random;
	params.IterationsCount = 10;
	params.TreeBuilder = GBTB_FastHist;
	TrainMultiGradientBoost( params );
	TestMultiClassificationResult( ModelDenseGB, ModelSparseGB );
}

TEST_F( RandomMultiClassification2000x20, GBTB_MultiFull )
{
	CRandom random( 0 );
	CGradientBoost::CParams params;
	params.Random = &random;
	params.IterationsCount = 10;
	params.TreeBuilder = GBTB_MultiFull;
	TrainMultiGradientBoost( params );
	TestMultiClassificationResult( ModelDenseGB, ModelSparseGB );
}

TEST_F( RandomMultiClassification2000x20, GBMR_Linked )
{
	CRandom random( 0 );
	CGradientBoost::CParams params;
	params.Random = &random;
	params.IterationsCount = 10;
	params.Representation = GBMR_Linked;
	TrainMultiGradientBoost( params );
	TestMultiClassificationResult( ModelDenseGB, ModelSparseGB );
}

TEST_F( RandomMultiClassification2000x20, GBMR_Compact )
{
	CRandom random( 0 );
	CGradientBoost::CParams params;
	params.Random = &random;
	params.IterationsCount = 10;
	params.Representation = GBMR_Compact;
	TrainMultiGradientBoost( params );
	TestMultiClassificationResult( ModelDenseGB, ModelSparseGB );
}

TEST_F( RandomMultiClassification2000x20, GBMR_QuickScorer )
{
	CRandom random( 0 );
	CGradientBoost::CParams params;
	params.Random = &random;
	params.IterationsCount = 10;
	params.Representation = GBMR_QuickScorer;
	TrainMultiGradientBoost( params );
	TestMultiClassificationResult( ModelDenseGB, ModelSparseGB );
}

TEST_F( RandomMultiClassification2000x20, OneVsAllLinear )
{
	CLinearBinaryClassifierBuilder linear( EF_SquaredHinge );
	COneVersusAll ovaLinear( linear );

	int begin = GetTickCount();
	auto model = ovaLinear.TrainModel<NeoML::IOneVersusAllModel>( *DenseRandomMultiProblem );
	GTEST_LOG_( INFO ) << "Dense train time: " << GetTickCount() - begin;
	ASSERT_TRUE( model != nullptr );

	begin = GetTickCount();
	auto model2 = ovaLinear.TrainModel<NeoML::IOneVersusAllModel>( *SparseRandomMultiProblem );
	GTEST_LOG_( INFO ) << "Sparse train time: " << GetTickCount() - begin;
	ASSERT_TRUE( model2 != nullptr );

	TestMultiClassificationResult( model, model2 );
}

TEST_F( RandomMultiClassification2000x20, OneVsAllRbf )
{
	CSvmBinaryClassifierBuilder svmRbf( CSvmKernel::KT_RBF );
	COneVersusAll ovaRbf( svmRbf );

	int begin = GetTickCount();
	auto model = ovaRbf.TrainModel<NeoML::IOneVersusAllModel>( *DenseRandomMultiProblem );
	GTEST_LOG_( INFO ) << "Dense train time: " << GetTickCount() - begin;
	ASSERT_TRUE( model != nullptr );

	begin = GetTickCount();
	auto model2 = ovaRbf.TrainModel<NeoML::IOneVersusAllModel>( *SparseRandomMultiProblem );
	GTEST_LOG_( INFO ) << "Sparse train time: " << GetTickCount() - begin;
	ASSERT_TRUE( model2 != nullptr );

	TestMultiClassificationResult( model, model2 );
}

TEST_F( RandomBinaryClassification4000x20, CrossValidationLinear )
{
	CLinearBinaryClassifierBuilder linear( EF_SquaredHinge );
	CrossValidate( 10, linear, DenseRandomBinaryProblem, SparseRandomBinaryProblem );
}

TEST_F( RandomBinaryClassification4000x20, CrossValidationSvmLinear )
{
	CSvmBinaryClassifierBuilder svmLinear( CSvmKernel::KT_Linear );
	CrossValidate( 10, svmLinear, DenseRandomBinaryProblem, SparseRandomBinaryProblem );
}

TEST_F( RandomBinaryClassification4000x20, CrossValidationSvmRbf )
{
	CSvmBinaryClassifierBuilder svmLinear( CSvmKernel::KT_RBF );
	CrossValidate( 10, svmLinear, DenseRandomBinaryProblem, SparseRandomBinaryProblem );
}

TEST_F( RandomBinaryClassification4000x20, CrossValidationDecisionTree )
{
	CDecisionTreeTrainingModel::CParams param;
	CDecisionTreeTrainingModel decisionTree( param );
	CrossValidate( 10, decisionTree, DenseRandomBinaryProblem, SparseRandomBinaryProblem );
}

// Test regression
TEST_F( RandomBinaryRegression4000x20, Linear )
{
	CRandom rand( 0 );
	auto denseRandomBinaryProblem = CRegressionRandomProblem::Random( rand, 4000, 20, 2 );
	auto denseBinaryTestData = CRegressionRandomProblem::Random( rand, 1000, 20, 2 );
	auto sparseRandomBinaryProblem = denseRandomBinaryProblem->CreateSparse();
	auto sparseBinaryTestData = denseBinaryTestData->CreateSparse();

	CLinearBinaryClassifierBuilder::CParams params( EF_L2_Regression );
	CLinearBinaryClassifierBuilder linear( params );

	int begin = GetTickCount();
	auto model = linear.TrainRegression( *denseRandomBinaryProblem );
	GTEST_LOG_( INFO ) << "Dense train time: " << GetTickCount() - begin;
	ASSERT_TRUE( model != nullptr );

	begin = GetTickCount();
	auto model2 = linear.TrainRegression( *sparseRandomBinaryProblem );
	GTEST_LOG_( INFO ) << "Sparse train time: " << GetTickCount() - begin;
	ASSERT_TRUE( model2 != nullptr );

	for( int i = 0; i < sparseBinaryTestData->GetVectorCount(); i++ ) {
		double result1 = model->Predict( denseBinaryTestData->GetVector( i ) );
		double result2 = model->Predict( sparseBinaryTestData->GetVector( i ) );
		double result4 = model2->Predict( denseBinaryTestData->GetVector( i ) );
		double result3 = model2->Predict( sparseBinaryTestData->GetVector( i ) );

		ASSERT_DOUBLE_EQ( result1, result2 );
		ASSERT_DOUBLE_EQ( result1, result3 );
		ASSERT_DOUBLE_EQ( result1, result4 );
	}
}

// GB binary tree builders
TEST_F( RandomBinaryGBRegression4000x20, Full )
{
	CRandom random( 0 );
	CGradientBoost::CParams params;
	params.Random = &random;
	params.IterationsCount = 10;
	TrainBinaryGradientBoost( params );
	TestBinaryRegressionResult();
}

TEST_F( RandomBinaryGBRegression4000x20, FastHist )
{
	CRandom random( 0 );
	CGradientBoost::CParams params;
	params.Random = &random;
	params.IterationsCount = 10;
	params.TreeBuilder = GBTB_FastHist;
	TrainBinaryGradientBoost( params );
	TestBinaryRegressionResult();
}

// GB multi tree builders
TEST_F( RandomMultiGBRegression2000x20, Full )
{
	CRandom random( 0 );
	CGradientBoost::CParams params;
	params.Random = &random;
	params.IterationsCount = 10;
	TrainMultiGradientBoost( params );
	TestMultiRegressionResult();
}

TEST_F( RandomMultiGBRegression2000x20, FastHist )
{
	CRandom random( 0 );
	CGradientBoost::CParams params;
	params.Random = &random;
	params.IterationsCount = 10;
	params.TreeBuilder = GBTB_FastHist;
	TrainMultiGradientBoost( params );
	TestMultiRegressionResult();
}

TEST_F( RandomMultiGBRegression2000x20, MultiFull )
{
	CRandom random( 0 );
	CGradientBoost::CParams params;
	params.Random = &random;
	params.IterationsCount = 10;
	params.TreeBuilder = GBTB_MultiFull;
	TrainMultiGradientBoost( params );
	TestMultiRegressionResult();
}

// test GB binary model's representations (to test Predict)
TEST_F( RandomBinaryGBRegression4000x20, Linked )
{
	CRandom random( 0 );
	CGradientBoost::CParams params;
	params.Random = &random;
	params.IterationsCount = 10;
	params.Representation = GBMR_Linked;
	TrainBinaryGradientBoost( params );
	TestBinaryRegressionResult();
}

TEST_F( RandomBinaryGBRegression4000x20, Compact )
{
	CRandom random( 0 );
	CGradientBoost::CParams params;
	params.Random = &random;
	params.IterationsCount = 10;
	params.Representation = GBMR_Compact;
	TrainBinaryGradientBoost( params );
	TestBinaryRegressionResult();
}

TEST_F( RandomBinaryGBRegression4000x20, QuickScorer )
{
	CRandom random( 0 );
	CGradientBoost::CParams params;
	params.Random = &random;
	params.IterationsCount = 10;
	params.Representation = GBMR_QuickScorer;
	TrainBinaryGradientBoost( params );
	TestBinaryRegressionResult();
}

// test GB multi model's representations (to test MultivariatePredict)
TEST_F( RandomMultiGBRegression2000x20, Linked )
{
	CRandom random( 0 );
	CGradientBoost::CParams params;
	params.Random = &random;
	params.IterationsCount = 10;
	params.Representation = GBMR_Linked;
	TrainMultiGradientBoost( params );
	TestMultiRegressionResult();
}

TEST_F( RandomMultiGBRegression2000x20, Compact )
{
	CRandom random( 0 );
	CGradientBoost::CParams params;
	params.Random = &random;
	params.IterationsCount = 10;
	params.Representation = GBMR_Compact;
	TrainMultiGradientBoost( params );
	TestMultiRegressionResult();
}
