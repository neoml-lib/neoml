/* Copyright @ 2024 ABBYY Production LLC

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


TEST(ParameterLayerTest, ParameterRunLearnSerializeTest)
{
    CRandom random(0x123);
    CDnn net(random, MathEngine());

    CPtr<CDnnBlob> paramBlob = CDnnBlob::CreateTensor(MathEngine(), CT_Float, { 1, 1, 1, 4, 2, 3, 10 });
    CPtr<CParameterLayer> params = Parameter(net, "params_layer", paramBlob);

    CPtr<CSinkLayer> output = new CSinkLayer(MathEngine());
    net.AddLayer(*output);
    output->Connect(*params);

    CPtr<CSourceLayer> dataLayer = Source(net, "in");
    CPtr<CDnnBlob> dataBlob = CDnnBlob::CreateTensor(MathEngine(), CT_Float, { 1, 1, 1, 4, 2, 3, 10 });
    dataLayer->SetBlob(dataBlob);

    for (int i = 0; i < paramBlob->GetDataSize(); ++i) {
        const float currVal = static_cast<float>(random.Uniform(0, 10));
        paramBlob->GetData().SetValueAt(i, currVal);
        dataBlob->GetData().SetValueAt(i, currVal + 1);
    }

    CPtr<CL1LossLayer> loss = new CL1LossLayer(MathEngine());
    loss->SetName("loss");
    loss->Connect(0, *params);
    loss->Connect(1, *dataLayer);
    net.AddLayer(*loss);

    CPtr<CDnnSimpleGradientSolver> solver = new CDnnSimpleGradientSolver(MathEngine());
    const float learningRate = 1e-3f;
    solver->SetLearningRate(learningRate);
    solver->SetMomentDecayRate(0.f);
    net.SetSolver(solver.Ptr());

    const int numOfIterations = 3;
    CPtr<CDnnBlob> expected = paramBlob->GetClone();
    CFloatHandleStackVar diff(MathEngine(), 1);
    diff.SetValue(learningRate * numOfIterations);
    MathEngine().VectorAddValue(params->GetBlob()->GetData(), expected->GetData(),
        params->GetBlob()->GetDataSize(), diff);

    EXPECT_TRUE(net.HasLayer("params_layer"));

    for (int i = 0; i < numOfIterations; ++i) {
        net.RunAndLearnOnce();
        EXPECT_TRUE(CompareBlobs(*(output->GetBlob()), *(params->GetBlob())));
    }

    EXPECT_TRUE(CompareBlobs(*paramBlob, *expected));
}
