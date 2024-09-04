/* Copyright © 2021-2024 ABBYY

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

void Train( ITrainingModel& trainingModel, const IProblem& denseProblem, const IProblem& sparseProblem,
	CPtr<IModel>& modelDense, CPtr<IModel>& modelSparse )
{
	int begin = GetTickCount();
	modelDense = trainingModel.Train( denseProblem );
	GTEST_LOG_( INFO ) << "Dense train time: " << ( GetTickCount() - begin );
	ASSERT_TRUE( modelDense != nullptr );

	begin = GetTickCount();
	modelSparse = trainingModel.Train( sparseProblem );
	GTEST_LOG_( INFO ) << "Sparse train time: " << ( GetTickCount() - begin );
	ASSERT_TRUE( modelSparse != nullptr );
}

template<class TModel>
void TrainGB( const CGradientBoost::CParams& params, const IProblem& denseProblem, const IProblem& sparseProblem,
	CPtr<TModel>& modelDense, CPtr<TModel>& modelSparse )
{
	CGradientBoost boosting( params );
	int begin = GetTickCount();
	modelDense = boosting.TrainModel<TModel>( denseProblem );
	GTEST_LOG_( INFO ) << "\n Random = " << params.Random->Next()
		<< "\n Dense train time: " << ( GetTickCount() - begin )
		<< "\n The last loss: " << boosting.GetLastLossMean();
	ASSERT_TRUE( modelDense != nullptr );

	params.Random->Reset( 0 );
	begin = GetTickCount();
	modelSparse = boosting.TrainModel<TModel>( sparseProblem );
	GTEST_LOG_( INFO ) << "\n Random = " << params.Random->Next()
		<< "\n Sparse train time: " << ( GetTickCount() - begin )
		<< "\n The last loss: " << boosting.GetLastLossMean();
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

	CPtr<IModel> ModelDense;
	CPtr<IModel> ModelSparse;

	void TrainBinary( ITrainingModel& trainingModel )
	{ Train( trainingModel, *DenseRandomBinaryProblem, *SparseRandomBinaryProblem, ModelDense, ModelSparse ); }

	void TestBinaryClassificationResult() const
	{ TestClassificationResult( ModelDense, ModelSparse, DenseBinaryTestData, SparseBinaryTestData ); }
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

	CPtr<IModel> ModelDense;
	CPtr<IModel> ModelSparse;

	void TrainMulti( ITrainingModel& trainingModel )
	{ Train( trainingModel, *DenseRandomMultiProblem, *SparseRandomMultiProblem, ModelDense, ModelSparse ); }

	void TrainMultiGradientBoost( const CGradientBoost::CParams& params )
	{ TrainGB( params, *DenseRandomMultiProblem, *SparseRandomMultiProblem, ModelDense, ModelSparse ); }

	void TestMultiClassificationResult() const
	{ TestClassificationResult( ModelDense, ModelSparse, DenseMultiTestData, SparseMultiTestData ); }
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
		auto cmpFloatVectors = []( const CFloatVectorDesc& v1, const CFloatVectorDesc& v2 )
		{
			ASSERT_EQ( v1.Size, v2.Size );
			ASSERT_EQ( ::memcmp( v1.Values, v2.Values, v2.Size * sizeof( float ) ), 0 );
			ASSERT_EQ( ::memcmp( v1.Indexes, v2.Indexes, ( v2.Indexes == nullptr ? 0 : v2.Size ) * sizeof( float ) ), 0 );
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
	CLinear::CParams params( EF_SquaredHinge );
	params.L1Coeff = 0.05f;
	CLinear linear( params );
	TrainBinary( linear );
	TestBinaryClassificationResult();
}

TEST_F( RandomBinaryClassification4000x20, SvmLinear )
{
	CSvm::CParams params( CSvmKernel::KT_Linear );
	CSvm svmLinear( params );
	TrainBinary( svmLinear );
	TestBinaryClassificationResult();
}

TEST_F( RandomBinaryClassification4000x20, SvmRbf )
{
	CSvm::CParams params( CSvmKernel::KT_RBF );
	CSvm svmRbf( params );
	TrainBinary( svmRbf );
	TestBinaryClassificationResult();
}

TEST_F( RandomBinaryClassification4000x20, DecisionTree )
{
	CDecisionTree::CParams param;
	CDecisionTree decisionTree( param );
	TrainBinary( decisionTree );
	TestBinaryClassificationResult();
}

TEST_F( RandomMultiClassification2000x20, GBTB_Full )
{
	CRandom random( 0 );
	CGradientBoost::CParams params;
	params.Random = &random;
	params.IterationsCount = 10;
	TrainMultiGradientBoost( params );
	TestMultiClassificationResult();
}

TEST_F( RandomMultiClassification2000x20, GBTB_FastHist )
{
	CRandom random( 0 );
	CGradientBoost::CParams params;
	params.Random = &random;
	params.IterationsCount = 10;
	params.TreeBuilder = GBTB_FastHist;
	TrainMultiGradientBoost( params );
	TestMultiClassificationResult();
}

TEST_F( RandomMultiClassification2000x20, GBTB_MultiFull )
{
	CRandom random( 0 );
	CGradientBoost::CParams params;
	params.Random = &random;
	params.IterationsCount = 10;
	params.TreeBuilder = GBTB_MultiFull;
	TrainMultiGradientBoost( params );
	TestMultiClassificationResult();
}

TEST_F( RandomMultiClassification2000x20, GBTB_MultiFastHist )
{
	CRandom random( 0 );
	CGradientBoost::CParams params;
	params.Random = &random;
	params.IterationsCount = 10;
	params.TreeBuilder = GBTB_MultiFastHist;
	TrainMultiGradientBoost( params );
	TestMultiClassificationResult();
}

TEST_F( RandomMultiClassification2000x20, GBMR_Linked )
{
	CRandom random( 0 );
	CGradientBoost::CParams params;
	params.Random = &random;
	params.IterationsCount = 10;
	params.Representation = GBMR_Linked;
	TrainMultiGradientBoost( params );
	TestMultiClassificationResult();
}

TEST_F( RandomMultiClassification2000x20, GBMR_Compact )
{
	CRandom random( 0 );
	CGradientBoost::CParams params;
	params.Random = &random;
	params.IterationsCount = 10;
	params.Representation = GBMR_Compact;
	TrainMultiGradientBoost( params );
	TestMultiClassificationResult();
}

TEST_F( RandomMultiClassification2000x20, GBMR_QuickScorer )
{
	CRandom random( 0 );
	CGradientBoost::CParams params;
	params.Random = &random;
	params.IterationsCount = 10;
	params.Representation = GBMR_QuickScorer;
	TrainMultiGradientBoost( params );
	TestMultiClassificationResult();
}

TEST_F( RandomMultiClassification2000x20, OneVsAllLinear )
{
	CLinear linear( EF_SquaredHinge );
	COneVersusAll ovaLinear( linear );
	TrainMulti( ovaLinear );
	TestMultiClassificationResult();

	GTEST_LOG_( INFO ) << "Train implicitly and compare";
	CPtr<IModel> modelImplicitDense;
	CPtr<IModel> modelImplicitSparse;
	Train( linear, *DenseRandomMultiProblem, *SparseRandomMultiProblem, modelImplicitDense, modelImplicitSparse );
	TestClassificationResult( ModelDense, modelImplicitDense, DenseMultiTestData, SparseMultiTestData );
	TestClassificationResult( ModelSparse, modelImplicitDense, DenseMultiTestData, SparseMultiTestData );
	TestClassificationResult( ModelSparse, modelImplicitSparse, DenseMultiTestData, SparseMultiTestData );
}

TEST_F( RandomMultiClassification2000x20, OneVsAllRbf )
{
	CSvm svmRbf( CSvmKernel::KT_RBF );
	COneVersusAll ovaRbf( svmRbf );
	TrainMulti( ovaRbf );
	TestMultiClassificationResult();

	GTEST_LOG_( INFO ) << "Train implicitly and compare";
	CPtr<IModel> modelImplicitDense;
	CPtr<IModel> modelImplicitSparse;
	Train( svmRbf, *DenseRandomMultiProblem, *SparseRandomMultiProblem, modelImplicitDense, modelImplicitSparse );
	TestClassificationResult( ModelDense, modelImplicitDense, DenseMultiTestData, SparseMultiTestData );
	TestClassificationResult( ModelSparse, modelImplicitDense, DenseMultiTestData, SparseMultiTestData );
	TestClassificationResult( ModelSparse, modelImplicitSparse, DenseMultiTestData, SparseMultiTestData );
}

TEST_F( RandomMultiClassification2000x20, OneVsAllDecisionTree )
{
	CDecisionTree::CParams param;
	CDecisionTree decisionTree( param );
	COneVersusAll ovaDecisionTree( decisionTree );
	TrainMulti( ovaDecisionTree );
	TestMultiClassificationResult();

	GTEST_LOG_( INFO ) << "Train implicitly and compare";
	CPtr<IModel> modelImplicitDense;
	CPtr<IModel> modelImplicitSparse;
	param.MulticlassMode = MM_OneVsAll;
	CDecisionTree decisionTree2( param );
	Train( decisionTree2, *DenseRandomMultiProblem, *SparseRandomMultiProblem, modelImplicitDense, modelImplicitSparse );
	TestClassificationResult( ModelDense, modelImplicitDense, DenseMultiTestData, SparseMultiTestData );
	TestClassificationResult( ModelSparse, modelImplicitDense, DenseMultiTestData, SparseMultiTestData );
	TestClassificationResult( ModelSparse, modelImplicitSparse, DenseMultiTestData, SparseMultiTestData );
}

TEST_F( RandomMultiClassification2000x20, OneVsOneLinear )
{
	CLinear::CParams params( EF_SquaredHinge );
	params.MulticlassMode = MM_OneVsOne;

	CLinear linear( params );
	COneVersusOne ovoLinear( linear );
	TrainMulti( ovoLinear );
	TestMultiClassificationResult();

	GTEST_LOG_( INFO ) << "Train implicitly and compare";
	CPtr<IModel> modelImplicitDense;
	CPtr<IModel> modelImplicitSparse;
	Train( linear, *DenseRandomMultiProblem, *SparseRandomMultiProblem, modelImplicitDense, modelImplicitSparse );
	TestClassificationResult( ModelDense, modelImplicitDense, DenseMultiTestData, SparseMultiTestData );
	TestClassificationResult( ModelSparse, modelImplicitDense, DenseMultiTestData, SparseMultiTestData );
	TestClassificationResult( ModelSparse, modelImplicitSparse, DenseMultiTestData, SparseMultiTestData );
}

TEST_F( RandomMultiClassification2000x20, OneVsOneRbf )
{
	CSvm::CParams params( CSvmKernel::KT_RBF );
	params.MulticlassMode = MM_OneVsOne;

	CSvm svmRbf( params );
	COneVersusOne ovoRbf( svmRbf );
	TrainMulti( ovoRbf );
	TestMultiClassificationResult();

	GTEST_LOG_( INFO ) << "Train implicitly and compare";
	CPtr<IModel> modelImplicitDense;
	CPtr<IModel> modelImplicitSparse;
	Train( svmRbf, *DenseRandomMultiProblem, *SparseRandomMultiProblem, modelImplicitDense, modelImplicitSparse );
	TestClassificationResult( ModelDense, modelImplicitDense, DenseMultiTestData, SparseMultiTestData );
	TestClassificationResult( ModelSparse, modelImplicitDense, DenseMultiTestData, SparseMultiTestData );
	TestClassificationResult( ModelSparse, modelImplicitSparse, DenseMultiTestData, SparseMultiTestData );
}

TEST_F( RandomMultiClassification2000x20, OneVsOneDecisionTree )
{
	CDecisionTree::CParams param;
	CDecisionTree decisionTree( param );
	COneVersusOne ovoDecisionTree( decisionTree );
	TrainMulti( ovoDecisionTree );
	TestMultiClassificationResult();

	GTEST_LOG_( INFO ) << "Train implicitly and compare";
	CPtr<IModel> modelImplicitDense;
	CPtr<IModel> modelImplicitSparse;
	param.MulticlassMode = MM_OneVsOne;
	CDecisionTree decisionTree2( param );
	Train( decisionTree2, *DenseRandomMultiProblem, *SparseRandomMultiProblem, modelImplicitDense, modelImplicitSparse );
	TestClassificationResult( ModelDense, modelImplicitDense, DenseMultiTestData, SparseMultiTestData );
	TestClassificationResult( ModelSparse, modelImplicitDense, DenseMultiTestData, SparseMultiTestData );
	TestClassificationResult( ModelSparse, modelImplicitSparse, DenseMultiTestData, SparseMultiTestData );
}

TEST_F( RandomBinaryClassification4000x20, CrossValidationLinear )
{
	CLinear linear( EF_SquaredHinge );
	CrossValidate( 10, linear, DenseRandomBinaryProblem, SparseRandomBinaryProblem );
}

TEST_F( RandomBinaryClassification4000x20, CrossValidationSvmLinear )
{
	CSvm svmLinear( CSvmKernel::KT_Linear );
	CrossValidate( 10, svmLinear, DenseRandomBinaryProblem, SparseRandomBinaryProblem );
}

TEST_F( RandomBinaryClassification4000x20, CrossValidationSvmRbf )
{
	CSvm svmLinear( CSvmKernel::KT_RBF );
	CrossValidate( 10, svmLinear, DenseRandomBinaryProblem, SparseRandomBinaryProblem );
}

TEST_F( RandomBinaryClassification4000x20, CrossValidationDecisionTree )
{
	CDecisionTree::CParams param;
	CDecisionTree decisionTree( param );
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

	CLinear::CParams params( EF_L2_Regression );
	CLinear linear( params );

	int begin = GetTickCount();
	auto model = linear.TrainRegression( *denseRandomBinaryProblem );
	GTEST_LOG_( INFO ) << "Dense train time: " << ( GetTickCount() - begin );
	ASSERT_TRUE( model != nullptr );

	begin = GetTickCount();
	auto model2 = linear.TrainRegression( *sparseRandomBinaryProblem );
	GTEST_LOG_( INFO ) << "Sparse train time: " << ( GetTickCount() - begin );
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

TEST_F( RandomMultiGBRegression2000x20, MultiFastHist )
{
	CRandom random( 0 );
	CGradientBoost::CParams params;
	params.Random = &random;
	params.IterationsCount = 10;
	params.TreeBuilder = GBTB_MultiFastHist;
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

TEST( FunctionSetArgumentTest, InitialGradient )
{
	auto testImpl = [] ( CFunctionWithGradient&& func ) -> void
	{
		CFloatVector zeroArg( 11, 0.f );
		func.SetArgument( zeroArg );

		ASSERT_GE( 1e5, func.Gradient().Norm() );
	};

	CRandom rand( 42 );
	CPtr<IProblem> classProblem = CClassificationRandomProblem::Random( rand, 22, 10, 2 ).Ptr();
	CPtr<IRegressionProblem> regrProblem = CRegressionRandomProblem::Random( rand, 22, 10, 2 ).Ptr();

	testImpl( CSquaredHinge( *classProblem, 1, 0.001f, 16 ) );
	testImpl( CL2Regression( *regrProblem, 1, 0.5, 0.001f, 16 ) );
	testImpl( CLogRegression( *classProblem, 1, 0.001f, 16 ) );
	testImpl( CSmoothedHinge( *classProblem, 1, 0.001f, 16 ) );
}
