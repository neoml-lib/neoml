/* Copyright © 2024 ABBYY

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

using namespace NeoML;
using namespace NeoMLTest;

TEST( CLossLayerTest, DnnCrossEntropyLossLayerTest )
{
	const float maxDelta = 0.01f;
	const int vectorSize = 20;
	CPtr<CCrossEntropyLossLayer> lossLayer = new CCrossEntropyLossLayer( MathEngine() );

	CRandom random;
	float err = lossLayer->TestRandom( random, 64, 0.f, 1.f, maxDelta, vectorSize, /*oneHot*/true );
	EXPECT_LT( err, maxDelta * maxDelta * vectorSize );

	err = lossLayer->TestRandom( random, 64, 0.f, 1.f, vectorSize, maxDelta, vectorSize );
	EXPECT_LT( err, maxDelta * maxDelta * vectorSize ); // int labels
}

TEST( CLossLayerTest, DnnBinaryCrossEntropyLossLayerTest )
{
	const float MaxDelta = 0.01f;
	const int VectorSize = 16;
	CPtr<CBinaryCrossEntropyLossLayer> lossLayer = new CBinaryCrossEntropyLossLayer( MathEngine() );

	CRandom random;
	const float err = lossLayer->TestRandom( random, 64, -1.f, 1.f, MaxDelta, VectorSize );
	EXPECT_LT( err, MaxDelta * MaxDelta * VectorSize );
}

TEST( CLossLayerTest, DnnEuclideanLossLayerTest )
{
	const float MaxDelta = 0.01f;
	const int VectorSize = 100;
	CPtr<CEuclideanLossLayer> lossLayer = new CEuclideanLossLayer( MathEngine() );

	CRandom random;
	float err = lossLayer->TestRandom( random, 64, -1.f, 1.f, MaxDelta, VectorSize );
	EXPECT_LT( err, MaxDelta * MaxDelta * VectorSize );
}

TEST( CLossLayerTest, DnnHingeLossLayerTest )
{
	const float MaxDelta = 0.01f;
	const int VectorSize = 16;
	CPtr<CHingeLossLayer> lossLayer = new CHingeLossLayer( MathEngine() );

	CRandom random;
	float err = lossLayer->TestRandom( random, 64, -1.f, 1.f, MaxDelta, VectorSize );
	EXPECT_LT( err, MaxDelta * MaxDelta * VectorSize );
}

TEST( CLossLayerTest, DnnSquaredHingeLossLayerTest )
{
	const float MaxDelta = 0.01f;
	const int VectorSize = 16;
	CPtr<CSquaredHingeLossLayer> lossLayer = new CSquaredHingeLossLayer( MathEngine() );

	CRandom random;
	float err = lossLayer->TestRandom( random, 64, -1.f, 1.f, MaxDelta, VectorSize );
	EXPECT_LT( err, MaxDelta * MaxDelta * VectorSize );
}

TEST( CLossLayerTest, DnnMultiHingeLossLayerTest )
{
	const float MaxDelta = 0.01f;
	const int VectorSize = 2;
	const int VectorSizeInt = 16;
	CPtr<CMultiHingeLossLayer> lossLayer = new CMultiHingeLossLayer( MathEngine() );

	CRandom random;
	float err = lossLayer->TestRandom( random, 64, -1.f, 1.f, MaxDelta, VectorSize );
	EXPECT_LT( err, MaxDelta * MaxDelta * VectorSize );

	err = lossLayer->TestRandom( random, 64, -1.f, 1.f, VectorSizeInt, MaxDelta, VectorSizeInt );
	EXPECT_LT( err, MaxDelta * MaxDelta * VectorSizeInt ); // int labels
}

TEST( CLossLayerTest, DnnMultiSquaredHingeLossLayerTest )
{
	const float MaxDelta = 0.01f;
	const int VectorSize = 2;
	const int VectorSizeInt = 16;
	CPtr<CMultiSquaredHingeLossLayer> lossLayer = new CMultiSquaredHingeLossLayer( MathEngine() );

	CRandom random;
	float err = lossLayer->TestRandom( random, 64, -1.f, 1.f, MaxDelta, VectorSize );
	EXPECT_LT( err, MaxDelta * MaxDelta * VectorSize );

	err = lossLayer->TestRandom( random, 64, -1.f, 1.f, VectorSizeInt, MaxDelta, VectorSizeInt );
	EXPECT_LT( err, MaxDelta * MaxDelta * VectorSizeInt ); // int labels
}

TEST( CLossLayerTest, DnnCenterLossLayerTest )
{
	const float MaxDelta = 0.01f;
	const int VectorSizeInt = 16;
	CPtr<CCenterLossLayer> lossLayer = new CCenterLossLayer( MathEngine() );
	lossLayer->SetNumberOfClasses( VectorSizeInt );

	CRandom random;
	const float err = lossLayer->TestRandom( random, 64, -1.f, 1.f, VectorSizeInt, MaxDelta, VectorSizeInt );
	EXPECT_LT( err, MaxDelta * MaxDelta * VectorSizeInt ); // int labels
}

TEST( CLossLayerTest, DnnFocalLossLayerTest )
{
	const float MaxDelta = 0.01f;
	const int VectorSize = 8;
	CPtr<CFocalLossLayer> lossLayer = new CFocalLossLayer( MathEngine() );

	CRandom random;
	const float err = lossLayer->TestRandom( random, 64, 0.f, 1.f, MaxDelta, VectorSize );
	EXPECT_LT( err, MaxDelta * MaxDelta * VectorSize );
}

TEST( CLossLayerTest, DnnBinaryFocalLossLayerTest )
{
	const float MaxDelta = 0.01f;
	const int VectorSize = 8;
	CPtr<CBinaryFocalLossLayer> lossLayer = new CBinaryFocalLossLayer( MathEngine() );

	CRandom random;
	const float err = lossLayer->TestRandom( random, 64, 0.f, 1.f, MaxDelta, VectorSize );
	EXPECT_LT( err, MaxDelta * MaxDelta * VectorSize );
}

