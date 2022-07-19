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

static const CString ArchiveName = "iterative_gb";

static void store( CGradientBoost& boosting )
{
	CArchiveFile file( ArchiveName, CArchive::store );
	CArchive archive( &file, CArchive::SD_Storing );
	boosting.Serialize( archive );
	archive.Close();
}

static void load( CGradientBoost& boosting )
{
	CArchiveFile file( ArchiveName, CArchive::load );
	CArchive archive( &file, CArchive::SD_Loading );
	boosting.Serialize( archive );
	archive.Close();
}

template<typename T>
static void iterativeTrainModel( const T* train,
	const CGradientBoost::CParams& params )
{
	const int iterationsCount = params.IterationsCount;
	{
		CGradientBoost iterativeBoosting( params );
		store( iterativeBoosting );
	}

	{
		CGradientBoost iterativeBoosting( params );
		load( iterativeBoosting );
		for( int i = 0; i < iterationsCount / 2; i++ ) {
			iterativeBoosting.TrainStep( *train );
		}
		store( iterativeBoosting );
	}

	{
		CGradientBoost iterativeBoosting( params );
		load( iterativeBoosting );
		for( int i = iterationsCount / 2; i < iterationsCount - 1; i++ ) {
			iterativeBoosting.TrainStep( *train );
		}
		store( iterativeBoosting );
	}

	CGradientBoost iterativeBoosting( params );
	load( iterativeBoosting );
	ASSERT_TRUE( iterativeBoosting.TrainStep( *train ) );
	store( iterativeBoosting );
}

static void classificationTest( const CPtr<CClassificationRandomProblem>& train,
	const CPtr<CClassificationRandomProblem>& test,
	const CGradientBoost::CParams& params )
{
	CGradientBoost boosting( params );
	auto trained = boosting.Train( *train );

	iterativeTrainModel( train.Ptr(), params );
	CGradientBoost iterativeBoosting( params );
	load( iterativeBoosting );
	auto iterativeTrained = iterativeBoosting.GetClassificationModel( *train );

	for( int i = 0; i < test->GetVectorCount(); i++ ) {
		CClassificationResult result1;
		CClassificationResult result2;

		ASSERT_TRUE( trained->Classify( test->GetVector( i ), result1 ) );
		ASSERT_TRUE( iterativeTrained->Classify( test->GetVector( i ), result2 ) );

		ASSERT_EQ( result1.PreferredClass, result2.PreferredClass );
	}
}

TEST( CGradientBoostingTest, IterativeClassificationTest )
{
	CRandom rand( 42 );
	auto train = CClassificationRandomProblem::Random( rand, 2000, 5, 3 );
	auto test = CClassificationRandomProblem::Random( rand, 500, 5, 3 );

	CGradientBoost::CParams params;
	params.IterationsCount = 50;
	params.MaxTreeDepth = 3;
	for( auto type : { GBTB_Full, GBTB_MultiFull, GBTB_FastHist } ) {
		params.TreeBuilder = type;
		classificationTest( train.Ptr(), test.Ptr(), params );
	}
}

static void regressionTest( const CPtr<CRegressionRandomProblem>& train,
	const CPtr<CRegressionRandomProblem>& test,
	const CGradientBoost::CParams& params )
{
	CGradientBoost boosting( params );
	auto trained = boosting.TrainRegression( *train );

	iterativeTrainModel( train.Ptr(), params );
	CGradientBoost iterativeBoosting( params );
	load( iterativeBoosting );
	auto iterativeTrained = iterativeBoosting.GetRegressionModel( *train );

	for( int i = 0; i < test->GetVectorCount(); i++ ) {
		double result1 = trained->Predict( test->GetVector( i ) );
		double result2 = iterativeTrained->Predict( test->GetVector( i ) );

		ASSERT_EQ( result1, result2 );
	}
}

TEST( CGradientBoostingTest, IterativeRegressionTest )
{
	CRandom rand( 42 );
	auto train = CRegressionRandomProblem::Random( rand, 2000, 20, 10 );
	auto test = CRegressionRandomProblem::Random( rand, 500, 20, 10 );

	CGradientBoost::CParams params;
	params.IterationsCount = 50;
	params.MaxTreeDepth = 3;
	for( auto type : { GBTB_Full, GBTB_MultiFull, GBTB_FastHist } ) {
		params.TreeBuilder = type;
		regressionTest( train.Ptr(), test.Ptr(), params );
	}
}