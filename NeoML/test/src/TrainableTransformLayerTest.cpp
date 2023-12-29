/* Copyright � 2017-2020 ABBYY Production LLC

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

 TEST(CTrainableTransformLayerTest, InferenceTest)
 {
 	CRandom random(0x123);
 	CDnn net(random, MathEngine());

 	CPtr<CParameterLayer> params = new CParameterLayer(MathEngine());
 	CPtr<CDnnBlob> paramBlob = CDnnBlob::CreateTensor(MathEngine(), CT_Float, { 1, 1, 1, 1, 1, 1, 50 });
 	paramBlob->Fill(0.f);
 	params->SetBlob(paramBlob);
 	params->SetName("test_layer");
 	net.AddLayer(*params);

 	CPtr<CSourceLayer> dataLayer = new CSourceLayer(MathEngine());
 	CPtr<CDnnBlob> dataBlob = CDnnBlob::CreateTensor(MathEngine(), CT_Float, { 1, 1, 1, 1, 1, 1, 50 });
 	dataLayer->SetName("in");
 	dataBlob->Fill(0.5f);
 	dataLayer->SetBlob(dataBlob);
 	net.AddLayer(*dataLayer);

 	CPtr<CEuclideanLossLayer> loss = new CEuclideanLossLayer(MathEngine());
 	loss->SetName("loss");
 	loss->Connect(0, *params);
 	loss->Connect(1, *dataLayer);
 	net.AddLayer(*loss);

 	CPtr<CDnnAdaptiveGradientSolver> solver = new CDnnAdaptiveGradientSolver(MathEngine());
 	net.SetSolver(solver.Ptr());

 	for (int i = 0; i < 2; ++i) {
 		net.RunAndLearnOnce();
 	}
 	auto res = params->GetBlob()->GetData().GetValueAt(5);
 }