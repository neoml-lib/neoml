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
#include <NeoML/NeoML.h>

using namespace NeoML;
using namespace NeoMLTest;

//----------------------------------------------------------------------------------------------------------------------

TEST(TiedEmbeddingTest, CompositeTest)
{
	const auto met = MathEngine().GetType();
	if( met != MET_Cpu && met != MET_Cuda ) {
		NEOML_HILIGHT( GTEST_LOG_( INFO ) ) << "Skipped rest of test for MathEngine type=" << met << " because no implementation.\n";
		// CrossEntropyLoss --> VectorEltwiseNotNegative
		return;
	}

	CRandom random( 42 );
	CDnn net(random, MathEngine());

	const int vectorCount = 2;
	const int embeddingSize = 2;
	CPtr<CSourceLayer> data = Source(net, "data");
	CPtr<CDnnBlob> dataBlob = CDnnBlob::CreateDataBlob(MathEngine(), CT_Float, 1, 1, 1);
	dataBlob->GetData().SetValue(1.f);
	data->SetBlob(dataBlob);

	CPtr<CMultichannelLookupLayer> lookup = new CMultichannelLookupLayer(MathEngine());
	lookup->SetDimensions({ { vectorCount, embeddingSize } });
	lookup->SetName("lookup");
	lookup->SetUseFrameworkLearning(true);
	CPtr<CDnnInitializer> embeddingInitializer = new CDnnUniformInitializer(random);
	lookup->Initialize(embeddingInitializer);

	CPtr<CCompositeLayer> compositeInner = new CCompositeLayer(net.GetMathEngine());
	compositeInner->SetName("compositeInner");
	compositeInner->SetInputMapping(*lookup);
	compositeInner->SetOutputMapping(*lookup);
	compositeInner->AddLayer(*lookup);

	CPtr<CCompositeLayer> composite = new CCompositeLayer(net.GetMathEngine());
	composite->SetName("composite");
	composite->Connect(*data);
	composite->AddLayer(*compositeInner);
	composite->SetInputMapping(*compositeInner);
	composite->SetOutputMapping(*compositeInner);
	net.AddLayer(*composite);

	CPtr<CTiedEmbeddingsLayer> tiedEmb = new CTiedEmbeddingsLayer(MathEngine());
	tiedEmb->SetName("tiedEmb");
	tiedEmb->SetEmbeddingsLayerPath({ "composite", "compositeInner", "lookup" });
	net.AddLayer(*tiedEmb);
	tiedEmb->Connect(*composite);

	CPtr<CSoftmaxLayer> softmax = new CSoftmaxLayer(MathEngine());
	softmax->SetName("softmax");
	net.AddLayer(*softmax);
	softmax->Connect(*tiedEmb);

	CPtr<CSourceLayer> targets = Source(net, "targets");
	CPtr<CDnnBlob> targetsBlob = CDnnBlob::CreateDataBlob(MathEngine(), CT_Int, 1, 1, 1);
	targetsBlob->GetData<int>().SetValueAt(0, 1);
	targets->SetBlob(targetsBlob);

	CPtr<CCrossEntropyLossLayer> loss = new CCrossEntropyLossLayer(MathEngine());
	loss->SetName("loss");
	net.AddLayer(*loss);
	loss->SetApplySoftmax(false);
	loss->Connect(0, *softmax);
	loss->Connect(1, *targets);

	CPtr<CDnnSimpleGradientSolver> solver = new CDnnSimpleGradientSolver(MathEngine());
	solver->SetL1Regularization(0);
	solver->SetL2Regularization(0);
	solver->SetLearningRate(1.f);
	net.SetSolver(solver.Ptr());

	const int numOfEpochs = 5;
	for (int i = 0; i < numOfEpochs; ++i) {
		net.RunAndLearnOnce();
	}

	ASSERT_NEAR(loss->GetLastLoss(), 0.f, 1e-3f);
	ASSERT_EQ(dynamic_cast<CMultichannelLookupLayer*>(net.GetLayer({ "composite", "compositeInner", "lookup" }).Ptr()), lookup.Ptr());
}

TEST(TiedEmbeddingTest, NoCompositeTest)
{
	const auto met = MathEngine().GetType();
	if( met != MET_Cpu && met != MET_Cuda ) {
		NEOML_HILIGHT( GTEST_LOG_( INFO ) ) << "Skipped rest of test for MathEngine type=" << met << " because no implementation.\n";
		// CrossEntropyLoss --> VectorEltwiseNotNegative
		return;
	}

	CRandom random( 42 );
	CDnn net(random, MathEngine());

	const int vectorCount = 2;
	const int embeddingSize = 2;
	CPtr<CSourceLayer> data = Source(net, "data");
	CPtr<CDnnBlob> dataBlob = CDnnBlob::CreateDataBlob(MathEngine(), CT_Float, 1, 1, 1);
	dataBlob->GetData().SetValue(1.f);
	data->SetBlob(dataBlob);

	CPtr<CMultichannelLookupLayer> lookup = new CMultichannelLookupLayer(MathEngine());
	lookup->SetDimensions({ { vectorCount, embeddingSize } });
	lookup->SetName("lookup");
	lookup->SetUseFrameworkLearning(true);
	CPtr<CDnnInitializer> embeddingInitializer = new CDnnUniformInitializer(random);
	lookup->Initialize(embeddingInitializer);
	net.AddLayer(*lookup);
	lookup->Connect(*data);

	CPtr<CTiedEmbeddingsLayer> tiedEmb = new CTiedEmbeddingsLayer(MathEngine());
	tiedEmb->SetName("tiedEmb");
	tiedEmb->SetEmbeddingsLayerName("lookup");
	net.AddLayer(*tiedEmb);
	tiedEmb->Connect(*lookup);

	CPtr<CSoftmaxLayer> softmax = new CSoftmaxLayer(MathEngine());
	softmax->SetName("softmax");
	net.AddLayer(*softmax);
	softmax->Connect(*tiedEmb);

	CPtr<CSourceLayer> targets = Source(net, "targets");
	CPtr<CDnnBlob> targetsBlob = CDnnBlob::CreateDataBlob(MathEngine(), CT_Int, 1, 1, 1);
	targetsBlob->GetData<int>().SetValueAt(0, 1);
	targets->SetBlob(targetsBlob);

	CPtr<CCrossEntropyLossLayer> loss = new CCrossEntropyLossLayer(MathEngine());
	loss->SetName("loss");
	net.AddLayer(*loss);
	loss->SetApplySoftmax(false);
	loss->Connect(0, *softmax);
	loss->Connect(1, *targets);

	CPtr<CDnnSimpleGradientSolver> solver = new CDnnSimpleGradientSolver(MathEngine());
	solver->SetL1Regularization(0);
	solver->SetL2Regularization(0);
	solver->SetLearningRate(1.f);
	net.SetSolver(solver.Ptr());

	net.RunOnce();

	const int numOfEpochs = 5;
	for (int i = 0; i < numOfEpochs; ++i) {
		net.RunAndLearnOnce();
	}

	ASSERT_NEAR(loss->GetLastLoss(), 0.f, 1e-3f);
	ASSERT_EQ(dynamic_cast<CMultichannelLookupLayer*>(net.GetLayer("lookup").Ptr()), lookup.Ptr());
}
