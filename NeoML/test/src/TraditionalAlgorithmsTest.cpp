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

#include <TestFixture.h>

#include <random>
#include <limits>

using namespace NeoML;
using namespace NeoMLTest;

class RandomDense4000x20 : public CNeoMLTestFixture {
public:
	static bool InitTestFixture(); // Инициализация теста
	static void DeinitTestFixture(); // Деинициализация теста
};

bool RandomDense4000x20::InitTestFixture()
{
	return true;
}

void RandomDense4000x20::DeinitTestFixture()
{
}

//------------------------------------------------------------------------------------------------
class CDenseMemoryProblem : public IProblem {
public:
	CDenseMemoryProblem( int height, int width, float* values, const int* _classes, const float* _weights ) :
		classCount( 0 ),
		classes( _classes ),
		weights( _weights )
	{
		desc.Height = height;
		desc.Width = width;
		desc.Values = values;

		for( int i = 0; i < height; i++ ) {
			if( classCount < classes[i] ) {
				classCount = classes[i];
			}
		}
		classCount++;
	}

	// IProblem interface methods:
	virtual int GetClassCount() const { return classCount; }
	virtual int GetFeatureCount() const { return desc.Width; }
	virtual bool IsDiscreteFeature( int ) const { return false; }
	virtual int GetVectorCount() const { return desc.Height; }
	virtual int GetClass( int index ) const { return classes[index]; }
	virtual CFloatMatrixDesc GetMatrix() const { return desc; }
	virtual double GetVectorWeight( int index ) const { return weights[index]; };

	static CPtr<CDenseMemoryProblem> Random( int samples, int features, int classes )
	{
		CPtr<CDenseMemoryProblem> res = new CDenseMemoryProblem();

		std::random_device rd;
		std::mt19937 gen( rd() );
		//std::uniform_real_distribution<float> df( std::numeric_limits<float>::min(), std::numeric_limits<float>::max() );
		std::uniform_real_distribution<float> df( -10, 10 );
		std::uniform_int_distribution<int> di( 0, classes - 1 );
		res->valuesArr.SetBufferSize( samples * features );
		res->classesArr.SetBufferSize( samples );
		for( int i = 0; i < samples; ++i ) {
			for( int j = 0; j < features; ++j ) {
				res->valuesArr.Add( df( gen ) );
			}
			res->classesArr.Add( di( gen ) );
		}
		// set weights to 1
		res->weightsArr.Add( 1., samples );
		res->classCount = classes;
		res->classes = res->classesArr.GetPtr();
		res->weights = res->weightsArr.GetPtr();
		res->desc.Height = samples;
		res->desc.Width = features;
		res->desc.Values = res->valuesArr.GetPtr();

		return res;
	}

protected:
	~CDenseMemoryProblem() override = default;

private:
	CDenseMemoryProblem() = default;

	CFloatMatrixDesc desc;
	int classCount;
	const int* classes;
	const float* weights;

	// memory holders when applicable
	CArray<float> valuesArr;
	CArray<int> classesArr;
	CArray<float> weightsArr;
};

CPtr<CDenseMemoryProblem> DenseBinaryProblem = CDenseMemoryProblem::Random( 4000, 20, 2 );

//------------------------------------------------------------------------------------------------

TEST_F( RandomDense4000x20, SvmLinear )
{
	CSvmBinaryClassifierBuilder::CParams params( CSvmKernel::KT_Linear );
	CSvmBinaryClassifierBuilder svmLinear( params );
	auto model = svmLinear.Train( *DenseBinaryProblem );
	ASSERT_TRUE( model != nullptr );
}

TEST_F( RandomDense4000x20, SvmRbf )
{
	CSvmBinaryClassifierBuilder::CParams params( CSvmKernel::KT_RBF );
	CSvmBinaryClassifierBuilder svmRbf( params );
	auto model = svmRbf.Train( *DenseBinaryProblem );
	ASSERT_TRUE( model != nullptr );
}

TEST_F( RandomDense4000x20, Linear )
{
	CLinearBinaryClassifierBuilder::CParams params( EF_SquaredHinge );
	params.L1Coeff = 0.05f;
	CLinearBinaryClassifierBuilder linear( params );
	auto model = linear.Train( *DenseBinaryProblem );
	ASSERT_TRUE( model != nullptr );
}

TEST_F( RandomDense4000x20, DecisionTree )
{
	CDecisionTreeTrainingModel::CParams param;
	param.MaxTreeDepth = 10;
	param.MaxNodesCount = 4096;
	param.AvailableMemory = Megabyte;
	CDecisionTreeTrainingModel decisionTree( param );
	auto model = decisionTree.TrainModel<IDecisionTreeModel>( *DenseBinaryProblem );
	ASSERT_TRUE( model != nullptr );
}

TEST_F( RandomDense4000x20, GradientBoosting )
{
	int begin = GetTickCount();

	CRandom random;
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

	CGradientBoost boosting( params );
	auto model = boosting.TrainModel<IGradientBoostModel>( *DenseBinaryProblem );
	ASSERT_TRUE( model != nullptr );

	GTEST_LOG_( INFO ) << "Train time: " << GetTickCount() - begin;
	GTEST_LOG_( INFO ) << "The last loss: " << boosting.GetLastLossMean();
}

TEST_F( RandomDense4000x20, CrossValidation )
{
	CLinearBinaryClassifierBuilder::CParams params( EF_SquaredHinge );
	CLinearBinaryClassifierBuilder linear( params );

	CCrossValidation crossValidation( linear, DenseBinaryProblem );
	const int PartsCount = 10;
	CCrossValidationResult result;
	crossValidation.Execute( PartsCount, AccuracyScore, result, true );

	ASSERT_EQ( result.Models.Size(), PartsCount );
	ASSERT_EQ( result.Success.Size(), PartsCount );
	ASSERT_EQ( result.Results.Size(), DenseBinaryProblem->GetVectorCount() );

	CSigmoid sigmoid;
	CalcSigmoidCoefficients( result, sigmoid );

	params.SigmoidCoefficients = sigmoid;
	CLinearBinaryClassifierBuilder builderS( params );

	auto model = builderS.Train( *DenseBinaryProblem );
	ASSERT_TRUE( model != nullptr );
}

TEST_F( RandomDense4000x20, OneVsAll )
{
	CLinearBinaryClassifierBuilder linear( EF_SquaredHinge );
	COneVersusAll ovaLinear( linear );
	auto model = ovaLinear.TrainModel<NeoML::IOneVersusAllModel>( *DenseBinaryProblem );
	ASSERT_TRUE( model != nullptr );
}

