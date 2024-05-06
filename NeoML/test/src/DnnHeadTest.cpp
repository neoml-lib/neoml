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

#include <NeoML/Dnn/DnnHead.h>
#include <NeoML/Dnn/Layers/DnnHeadAdapterLayer.h>
#include <TestFixture.h>

using namespace NeoML;
using namespace NeoMLTest;

//----------------------------------------------------------------------------------------------------------------------

namespace NeoMLTest {

static void initializeDnnBlobs(CDnn& dnn)
{
    CRandom random(0);
    CDnnUniformInitializer init(random, -0.5, 0.5);

    CDnnBlob* source1Blob = CDnnBlob::CreateTensor(MathEngine(), CT_Float, { 1, 1, 1, 4, 2, 3, 10 });
    init.InitializeLayerParams(*source1Blob, -1);
    static_cast<CSourceLayer*>(dnn.GetLayer("source1").Ptr())->SetBlob(source1Blob);

    CDnnBlob* source2Blob = CDnnBlob::CreateTensor(MathEngine(), CT_Float, { 1, 1, 1, 4, 2, 3, 10 });
    init.InitializeLayerParams(*source2Blob, -1);
    static_cast<CSourceLayer*>(dnn.GetLayer("source2").Ptr())->SetBlob(source2Blob);

    CDnnBlob* source3Blob = CDnnBlob::CreateTensor(MathEngine(), CT_Float, { 1, 1, 1, 4, 2, 3, 10 });
    init.InitializeLayerParams(*source3Blob, -1);
    static_cast<CSourceLayer*>(dnn.GetLayer("source3").Ptr())->SetBlob(source3Blob);

    CDnnBlob* targetBlob = CDnnBlob::CreateTensor(MathEngine(), CT_Float, { 1, 1, 1, 1, 1, 1, 3 });
    targetBlob->GetData().SetValueAt(0, -1.5f);
    targetBlob->GetData().SetValueAt(1, 2.4f);
    targetBlob->GetData().SetValueAt(2, 4.8f);
    static_cast<CSourceLayer*>(dnn.GetLayer("target").Ptr())->SetBlob(targetBlob);
}

static void createDnnHeadAdapter(CDnn& dnn, bool useDropout = true)
{
    //        +----------------+
    //        |                |                                                    [Target]
    //        |                v                                                        |
    //    [source1]        |-----------------------------------------|                  v
    //                     | [Fc1]-->[Gelu]-->[Fc2]-->[Relu]-->[Fc3] | -> [Concat] ->[Loss]
    //    [source2]        |-----------------------------------------|
    //        |                ^
    //        |                |
    //        +----------------+

    CPtr<CDnnHead> head = new CDnnHead(
        dnn.Random(), dnn.GetMathEngine(),
        FullyConnected(3000),
        Gelu(),
        FullyConnected(1000),
        Relu(),
        Dropout(useDropout ? 0.3f : 0.f),
        FullyConnected(1)
    );

    CPtr<CSourceLayer> source1 = Source(dnn, "source1");

    CPtr<CFullyConnectedLayer> fc1 = new CFullyConnectedLayer(MathEngine(), "fc1");
    fc1->SetNumberOfElements(50);
    fc1->Connect(*source1);
    dnn.AddLayer(*fc1);

    CPtr<CDnnHeadAdapterLayer> head1 = new CDnnHeadAdapterLayer(MathEngine());
    head1->SetName("head1");
    head1->Connect(*fc1);
    head1->SetDnnHead(head);
    dnn.AddLayer(*head1);

    CPtr<CSourceLayer> source2 = Source(dnn, "source2");

    CPtr<CFullyConnectedLayer> fc2 = new CFullyConnectedLayer(MathEngine(), "fc2");
    fc2->SetNumberOfElements(50);
    fc2->Connect(*source2);
    dnn.AddLayer(*fc2);

    CPtr<CDnnHeadAdapterLayer> head2 = new CDnnHeadAdapterLayer(MathEngine());
    head2->SetName("head2");
    head2->Connect(*fc2);
    head2->SetDnnHead(head);
    dnn.AddLayer(*head2);

    CPtr<CSourceLayer> source3 = Source(dnn, "source3");

    CPtr<CFullyConnectedLayer> fc3 = new CFullyConnectedLayer(MathEngine(), "fc3");
    fc3->SetNumberOfElements(50);
    fc3->Connect(*source3);
    dnn.AddLayer(*fc3);

    CPtr<CDnnHeadAdapterLayer> head3 = new CDnnHeadAdapterLayer(MathEngine());
    head3->SetName("head3");
    head3->Connect(*fc3);
    head3->SetDnnHead(head);
    dnn.AddLayer(*head3);

    CPtr<CConcatChannelsLayer> concat = new CConcatChannelsLayer(MathEngine());
    dnn.AddLayer(*concat);
    concat->Connect(0, *head1, 0);
    concat->Connect(1, *head2, 0);
    concat->Connect(2, *head3, 0);

    CPtr<CSourceLayer> targets = Source(dnn, "target");

    CPtr<CEuclideanLossLayer> loss = new CEuclideanLossLayer(MathEngine());
    loss->SetName("loss");
    dnn.AddLayer(*loss);
    loss->Connect(0, *concat, 0);
    loss->Connect(1, *targets, 0);

    CPtr<CSinkLayer> sink = new CSinkLayer(MathEngine());
    sink->SetName("sink");
    dnn.AddLayer(*sink);
    sink->Connect(0, *concat, 0);
    
    CPtr<CDnnAdaptiveGradientSolver> solver = new CDnnAdaptiveGradientSolver(MathEngine());
    const float learningRate = 1e-3f;
    solver->SetLearningRate(learningRate);
    dnn.SetSolver(solver.Ptr());

    initializeDnnBlobs(dnn);
}

static void createDnnHeadNaive(CDnn& dnn, bool useDropout = true)
{
    // Same architecture but without Head to compare
    //
    //                                                                       [Target]
    //                                                                          |
    //    [source1]-->[   ]-->[Gelu]-->[   ]-->[Relu]-->[   ]--->[        ]     v
    //                |Fc1]            |Fc2|            |Fc3|    | Concat |-->[Loss]
    //    [source2]-->[   ]-->[Gelu]-->[   ]-->[Relu]-->[   ]--->[        ]
    //

    const float dropoutRate = useDropout ? 0.3f : 0.f;
    CPtr<CSourceLayer> source1 = Source(dnn, "source1");;

    CPtr<CFullyConnectedLayer> fc0_1 = new CFullyConnectedLayer(MathEngine());
    fc0_1->SetName("fc0_1");
    dnn.AddLayer(*fc0_1);
    fc0_1->SetNumberOfElements(50);
    fc0_1->Connect(0, *source1);

    CPtr<CSourceLayer> source2 = Source(dnn, "source2");

    CPtr<CFullyConnectedLayer> fc0_2 = new CFullyConnectedLayer(MathEngine());
    fc0_2->SetName("fc0_2");
    dnn.AddLayer(*fc0_2);
    fc0_2->SetNumberOfElements(50);
    fc0_2->Connect(0, *source2);

    CPtr<CSourceLayer> source3 = Source(dnn, "source3");

    CPtr<CFullyConnectedLayer> fc0_3 = new CFullyConnectedLayer(MathEngine());
    fc0_3->SetName("fc0_3");
    dnn.AddLayer(*fc0_3);
    fc0_3->SetNumberOfElements(50);
    fc0_3->Connect(0, *source3);

    CPtr<CFullyConnectedLayer> fc1 = new CFullyConnectedLayer(MathEngine());
    fc1->SetName("fc1");
    dnn.AddLayer(*fc1);
    fc1->SetNumberOfElements(3000);
    fc1->Connect(0, *fc0_1);
    fc1->Connect(1, *fc0_2);
    fc1->Connect(2, *fc0_3);

    CPtr<CGELULayer> gelu1 = new CGELULayer(MathEngine());
    gelu1->SetName("gelu1");
    dnn.AddLayer(*gelu1);
    gelu1->Connect(0, *fc1, 0);

    CPtr<CGELULayer> gelu2 = new CGELULayer(MathEngine());
    gelu2->SetName("gelu2");
    dnn.AddLayer(*gelu2);
    gelu2->Connect(0, *fc1, 1);

    CPtr<CGELULayer> gelu3 = new CGELULayer(MathEngine());
    gelu3->SetName("gelu3");
    dnn.AddLayer(*gelu3);
    gelu3->Connect(0, *fc1, 2);

    CPtr<CFullyConnectedLayer> fc2 = new CFullyConnectedLayer(MathEngine());
    fc2->SetName("fc2");
    dnn.AddLayer(*fc2);
    fc2->SetNumberOfElements(1000);
    fc2->Connect(0, *gelu1);
    fc2->Connect(1, *gelu2);
    fc2->Connect(2, *gelu3);

    CPtr<CReLULayer> relu1 = new CReLULayer(MathEngine());
    relu1->SetName("relu1");
    dnn.AddLayer(*relu1);
    relu1->Connect(0, *fc2, 0);

    CPtr<CReLULayer> relu2 = new CReLULayer(MathEngine());
    relu2->SetName("relu2");
    dnn.AddLayer(*relu2);
    relu2->Connect(0, *fc2, 1);

    CPtr<CReLULayer> relu3 = new CReLULayer(MathEngine());
    relu3->SetName("relu3");
    dnn.AddLayer(*relu3);
    relu3->Connect(0, *fc2, 2);

    CPtr<CDropoutLayer> dropout1 = new CDropoutLayer(MathEngine());
    dropout1->SetName("dp1");
    dnn.AddLayer(*dropout1);
    dropout1->SetDropoutRate(dropoutRate);
    dropout1->Connect(0, *relu1);

    CPtr<CDropoutLayer> dropout2 = new CDropoutLayer(MathEngine());
    dropout2->SetName("dp2");
    dnn.AddLayer(*dropout2);
    dropout2->SetDropoutRate(dropoutRate);
    dropout2->Connect(0, *relu2);

    CPtr<CDropoutLayer> dropout3 = new CDropoutLayer(MathEngine());
    dropout3->SetName("dp3");
    dnn.AddLayer(*dropout3);
    dropout3->SetDropoutRate(dropoutRate);
    dropout3->Connect(0, *relu3);

    CPtr<CFullyConnectedLayer> fc3 = new CFullyConnectedLayer(MathEngine());
    fc3->SetName("fc3");
    fc3->SetNumberOfElements(1);
    dnn.AddLayer(*fc3);
    fc3->Connect(0, *dropout1);
    fc3->Connect(1, *dropout2);
    fc3->Connect(2, *dropout3);

    CPtr<CConcatChannelsLayer> concat = new CConcatChannelsLayer(MathEngine());
    dnn.AddLayer(*concat);
    concat->Connect(0, *fc3, 0);
    concat->Connect(1, *fc3, 1);
    concat->Connect(2, *fc3, 2);

    CPtr<CSourceLayer> targets = Source(dnn, "target");

    CPtr<CEuclideanLossLayer> loss = new CEuclideanLossLayer(MathEngine());
    loss->SetName("loss");
    dnn.AddLayer(*loss);
    loss->Connect(0, *concat, 0);
    loss->Connect(1, *targets, 0);

    CPtr<CSinkLayer> sink = new CSinkLayer(MathEngine());
    sink->SetName("sink");
    dnn.AddLayer(*sink);
    sink->Connect(0, *concat, 0);

    CPtr<CDnnAdaptiveGradientSolver> solver = new CDnnAdaptiveGradientSolver(MathEngine());
    const float learningRate = 1e-3f;
    solver->SetLearningRate(learningRate);
    dnn.SetSolver(solver.Ptr());

    initializeDnnBlobs(dnn);
}

} // namespace NeoMLTest

//----------------------------------------------------------------------------------------------------------------------

TEST( CDnnHeadTest, DnnHeadAdapterLearnTest )
{
    CRandom random( 0 );
    CDnn dnn( random, MathEngine());
    createDnnHeadAdapter(dnn);

    for(int i = 0; i < 100; ++i) {
        dnn.RunAndLearnOnce();
    }

    EXPECT_NEAR(static_cast<CLossLayer*>(dnn.GetLayer("loss").Ptr())->GetLastLoss(), 0, 1e-3f);
}

TEST(CDnnHeadTest, CheckDnnHeadAdapterInferenceMatch)
{
    CRandom random(0);
    CPtr<CDnnUniformInitializer> init = new CDnnUniformInitializer(random, 0.05f, 0.05f);

    CDnn dnnNoAdapters(random, MathEngine());
    dnnNoAdapters.SetInitializer(init.Ptr());
    createDnnHeadNaive(dnnNoAdapters);
    dnnNoAdapters.RunOnce();
    CDnnBlob* output0 = static_cast<CSinkLayer*>(dnnNoAdapters.GetLayer("sink").Ptr())->GetBlob();

    CDnn dnnWithAdapters(random, MathEngine());
    dnnWithAdapters.SetInitializer(init.Ptr());
    createDnnHeadAdapter(dnnWithAdapters);
    dnnWithAdapters.RunOnce();
    CDnnBlob* output1 = static_cast<CSinkLayer*>(dnnWithAdapters.GetLayer("sink").Ptr())->GetBlob();

    EXPECT_TRUE(CompareBlobs(*output0, *output1));
}

TEST(CDnnHeadTest, CheckDnnHeadAdapterLearningMatch)
{
    CRandom random(0);
    CPtr<CDnnUniformInitializer> init = new CDnnUniformInitializer(random, 0.05f, 0.05f);
    
    CDnn dnnNoAdapters(random, MathEngine());
    dnnNoAdapters.SetInitializer(init.Ptr());
    createDnnHeadNaive(dnnNoAdapters, false);

    CDnn dnnWithAdapters(random, MathEngine());
    dnnWithAdapters.SetInitializer(init.Ptr());
    createDnnHeadAdapter(dnnWithAdapters, false);

    for(int i = 0; i < 100; ++i) {
        dnnNoAdapters.RunAndLearnOnce();
        dnnWithAdapters.RunAndLearnOnce();
        EXPECT_EQ(static_cast<CLossLayer*>(dnnNoAdapters.GetLayer("loss").Ptr())->GetLastLoss(),
            static_cast<CLossLayer*>(dnnWithAdapters.GetLayer("loss").Ptr())->GetLastLoss());
    }
}

TEST(CDnnHeadTest, DnnHeadAdapterSerializationTest)
{
    CRandom random(0);
    CDnn dnn(random, MathEngine());

    createDnnHeadAdapter(dnn);
    dnn.RunOnce();
    CPtr<CDnnBlob> expected = static_cast<CSinkLayer*>(dnn.GetLayer("sink").Ptr())->GetBlob();
    {
        CMemoryFile file;
        {
            CArchive archive( &file, CArchive::store );
            dnn.Serialize( archive );
        }
        file.SeekToBegin();
        {
            CArchive archive( &file, CArchive::load );
            dnn.Serialize( archive );
        }
    }
    initializeDnnBlobs(dnn);
    dnn.RunOnce();
    CPtr<CDnnBlob> output = static_cast<CSinkLayer*>(dnn.GetLayer("sink").Ptr())->GetBlob();
    EXPECT_TRUE( CompareBlobs( *expected, *output ) );
}

TEST( CDnnHeadTest, DISABLED_DnnHeadAdapterInferencePerfomance )
{
    CRandom random( 0 );
    CDnn dnn( random, MathEngine() );

    createDnnHeadAdapter( dnn );
    OptimizeDnn( dnn );
    dnn.RunOnce();

    IPerformanceCounters* counters = MathEngine().CreatePerformanceCounters();
    counters->Synchronise();
    for( int i = 0; i < 1000; ++i ) {
        dnn.RunOnce();
    }
    counters->Synchronise();

    std::cerr << "Inference Time: " << ( double( ( *counters )[0].Value ) / 1000000 ) << " ms."
        << '\t' << "Peak.Mem: " << ( double( MathEngine().GetPeakMemoryUsage() ) / 1024 / 1024 ) << " MB"
        << '\n';
    delete counters;
}

TEST( CDnnHeadTest, DISABLED_DnnHeadNaiveInferencePerfomance )
{
    CRandom random( 0 );
    CDnn dnn( random, MathEngine() );

    createDnnHeadNaive( dnn );
    OptimizeDnn( dnn );
    dnn.RunOnce();

    IPerformanceCounters* counters = MathEngine().CreatePerformanceCounters();
    counters->Synchronise();
    for( int i = 0; i < 1000; ++i ) {
        dnn.RunOnce();
    }
    counters->Synchronise();

    std::cerr << "Inference Time: " << ( double( ( *counters )[0].Value ) / 1000000 ) << " ms."
        << '\t' << "Peak.Mem: " << ( double( MathEngine().GetPeakMemoryUsage() ) / 1024 / 1024 ) << " MB"
        << '\n';
    delete counters;
}
