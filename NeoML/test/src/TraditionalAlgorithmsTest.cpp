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
#include <DenseMemoryProblem.h>

using namespace NeoML;
using namespace NeoMLTest;

class RandomBinary4000x20 : public CNeoMLTestFixture {
public:
	static bool InitTestFixture() { return true; } // Инициализация теста
	static void DeinitTestFixture() {} // Деинициализация теста
};

class RandomMulti2000x20 : public CNeoMLTestFixture {
public:
	static bool InitTestFixture() { return true; } // Инициализация теста
	static void DeinitTestFixture() {} // Деинициализация теста
};

//------------------------------------------------------------------------------------------------

CPtr<CDenseMemoryProblem> DenseRandomBinaryProblem = CDenseMemoryProblem::Random( 4000, 20, 2 );
CPtr<CMemoryProblem> SparseRandomBinaryProblem = DenseRandomBinaryProblem->CreateSparse();

CPtr<CDenseMemoryProblem> DenseRandomMultiProblem = CDenseMemoryProblem::Random( 2000, 20, 10 );
CPtr<CMemoryProblem> SparseRandomMultiProblem = DenseRandomMultiProblem->CreateSparse();

TEST_F( RandomBinary4000x20, SvmLinear )
{
	CSvmBinaryClassifierBuilder::CParams params( CSvmKernel::KT_Linear );
	CSvmBinaryClassifierBuilder svmLinear( params );

	int begin = GetTickCount();
	auto model = svmLinear.Train( *DenseRandomBinaryProblem );
	GTEST_LOG_( INFO ) << "Dense train time: " << GetTickCount() - begin;
	ASSERT_TRUE( model != nullptr );

	begin = GetTickCount();
	auto model2 = svmLinear.Train( *SparseRandomBinaryProblem );
	GTEST_LOG_( INFO ) << "Sparse train time: " << GetTickCount() - begin;
	ASSERT_TRUE( model2 != nullptr );
}

TEST_F( RandomBinary4000x20, SvmRbf )
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
}

TEST_F( RandomBinary4000x20, Linear )
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
}

TEST_F( RandomBinary4000x20, DISABLED_DecisionTree )
{
	CDecisionTreeTrainingModel::CParams param;
	param.MaxTreeDepth = 10;
	param.MaxNodesCount = 4096;
	param.AvailableMemory = Megabyte;
	CDecisionTreeTrainingModel decisionTree( param );

	int begin = GetTickCount();
	auto model = decisionTree.TrainModel<IDecisionTreeModel>( *DenseRandomBinaryProblem );
	ASSERT_TRUE( model != nullptr );
	GTEST_LOG_( INFO ) << "Dense train time: " << GetTickCount() - begin;

	begin = GetTickCount();
	auto model2 = decisionTree.Train( *SparseRandomBinaryProblem );
	GTEST_LOG_( INFO ) << "Sparse train time: " << GetTickCount() - begin;
	ASSERT_TRUE( model2 != nullptr );
}

TEST_F( RandomMulti2000x20, GradientBoostingFull )
{
	CRandom random;
	random.Reset( 0 );
	GTEST_LOG_( INFO ) << "Random = " << random.Next();
	CGradientBoost::CParams params;
	params.Random = &random;
	params.LossFunction = CGradientBoost::LF_Binomial;
	params.IterationsCount = 10;
	params.LearningRate = 0.3f;
	params.MaxTreeDepth = 8;
	params.ThreadCount = 1;
	params.Subsample = 1;
	params.Subfeature = 1;
	params.MinSubsetWeight = 8;
	params.TreeBuilder = GBTB_Full;

	CGradientBoost boosting( params );
	int begin = GetTickCount();
	auto model = boosting.TrainModel<IGradientBoostModel>( *DenseRandomMultiProblem );
	GTEST_LOG_( INFO ) << "Dense train time: " << GetTickCount() - begin;
	GTEST_LOG_( INFO ) << "The last loss: " << boosting.GetLastLossMean();
	ASSERT_TRUE( model != nullptr );

	random.Reset( 0 );
	GTEST_LOG_( INFO ) << "Random = " << random.Next();
	begin = GetTickCount();
	auto model2 = boosting.TrainModel<IGradientBoostModel>( *SparseRandomMultiProblem );
	GTEST_LOG_( INFO ) << "Sparse train time: " << GetTickCount() - begin;
	GTEST_LOG_( INFO ) << "The last loss: " << boosting.GetLastLossMean();
	ASSERT_TRUE( model2 != nullptr );
}

TEST_F( RandomMulti2000x20, GradientBoostingFastHist )
{
	CRandom random;
	random.Reset( 0 );
	GTEST_LOG_( INFO ) << "Random = " << random.Next();
	CGradientBoost::CParams params;
	params.Random = &random;
	params.LossFunction = CGradientBoost::LF_Binomial;
	params.IterationsCount = 10;
	params.LearningRate = 0.3f;
	params.MaxTreeDepth = 8;
	params.ThreadCount = 1;
	params.Subsample = 1;
	params.Subfeature = 1;
	params.MinSubsetWeight = 8;
	params.TreeBuilder = GBTB_FastHist;

	CGradientBoost boosting( params );
	int begin = GetTickCount();
	auto model = boosting.TrainModel<IGradientBoostModel>( *DenseRandomMultiProblem );
	GTEST_LOG_( INFO ) << "Dense train time: " << GetTickCount() - begin;
	GTEST_LOG_( INFO ) << "The last loss: " << boosting.GetLastLossMean();
	ASSERT_TRUE( model != nullptr );

	random.Reset( 0 );
	GTEST_LOG_( INFO ) << "Random = " << random.Next();
	begin = GetTickCount();
	auto model2 = boosting.TrainModel<IGradientBoostModel>( *SparseRandomMultiProblem );
	GTEST_LOG_( INFO ) << "Sparse train time: " << GetTickCount() - begin;
	GTEST_LOG_( INFO ) << "The last loss: " << boosting.GetLastLossMean();
	ASSERT_TRUE( model2 != nullptr );
}

TEST_F( RandomBinary4000x20, CrossValidation )
{
	CLinearBinaryClassifierBuilder::CParams params( EF_SquaredHinge );
	CLinearBinaryClassifierBuilder linear( params );

	CCrossValidation crossValidation( linear, DenseRandomBinaryProblem );
	const int PartsCount = 10;
	CCrossValidationResult result;

	int begin = GetTickCount();
	crossValidation.Execute( PartsCount, AccuracyScore, result, true );
	GTEST_LOG_( INFO ) << "Dense execution time: " << GetTickCount() - begin;

	ASSERT_EQ( result.Models.Size(), PartsCount );
	ASSERT_EQ( result.Success.Size(), PartsCount );
	ASSERT_EQ( result.Results.Size(), DenseRandomBinaryProblem->GetVectorCount() );

	CSigmoid sigmoid;
	CalcSigmoidCoefficients( result, sigmoid );

	params.SigmoidCoefficients = sigmoid;
	CLinearBinaryClassifierBuilder builderS( params );

	begin = GetTickCount();
	auto model = builderS.Train( *DenseRandomBinaryProblem );
	GTEST_LOG_( INFO ) << "Dense train time: " << GetTickCount() - begin;
	ASSERT_TRUE( model != nullptr );
}

TEST_F( RandomMulti2000x20, OneVsAllLinear )
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
}

TEST_F( RandomMulti2000x20, OneVsAllRbf )
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
}

