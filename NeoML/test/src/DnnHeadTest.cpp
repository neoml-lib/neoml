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

static void initializeDnnBlobs( CDnn& dnn )
{
    CRandom random( 0 );
    CDnnUniformInitializer init( random, -0.5, 0.5 );

    CDnnBlob* source1Blob = CDnnBlob::CreateTensor( MathEngine(), CT_Float, { 1, 1, 1, 4, 2, 3, 10 } );
    init.InitializeLayerParams( *source1Blob, -1 );
    CheckCast<CSourceLayer>( dnn.GetLayer( "source1" ).Ptr() )->SetBlob( source1Blob );

    CDnnBlob* source2Blob = CDnnBlob::CreateTensor( MathEngine(), CT_Float, { 1, 1, 1, 4, 2, 3, 10 } );
    init.InitializeLayerParams( *source2Blob, -1 );
    CheckCast<CSourceLayer>( dnn.GetLayer( "source2" ).Ptr() )->SetBlob( source2Blob );

    CDnnBlob* source3Blob = CDnnBlob::CreateTensor( MathEngine(), CT_Float, { 1, 1, 1, 4, 2, 3, 10 } );
    init.InitializeLayerParams( *source3Blob, -1 );
    CheckCast<CSourceLayer>( dnn.GetLayer( "source3" ).Ptr() )->SetBlob( source3Blob );

    CDnnBlob* targetBlob = CDnnBlob::CreateTensor( MathEngine(), CT_Float, { 1, 1, 1, 1, 1, 1, 3 } );
    targetBlob->GetData().SetValueAt( 0, -1.5f );
    targetBlob->GetData().SetValueAt( 1, 2.4f );
    targetBlob->GetData().SetValueAt( 2, 4.8f );
    CheckCast<CSourceLayer>( dnn.GetLayer( "target" ).Ptr() )->SetBlob( targetBlob );
}

static void createDnn( CDnn& dnn, bool isNaive, int complexity = 1000, float dropoutRate = 0.3f, bool freeTerm = false )
{
    CPtr<CSourceLayer> source1 = Source( dnn, "source1" );
    CPtr<CSourceLayer> source2 = Source( dnn, "source2" );
    CPtr<CSourceLayer> source3 = Source( dnn, "source3" );
    CPtr<CSourceLayer> targets = Source( dnn, "target" );

    CPtr<CFullyConnectedLayer> fc01 = FullyConnected( complexity, freeTerm )( "fc01", source1.Ptr() );
    CPtr<CFullyConnectedLayer> fc02 = FullyConnected( complexity, freeTerm )( "fc02", source2.Ptr() );
    CPtr<CFullyConnectedLayer> fc03 = FullyConnected( complexity, freeTerm )( "fc03", source3.Ptr() );

    CPtr<CConcatChannelsLayer> concat;
    
    if( isNaive ) {
        // Same architecture but without Head to compare                                         [target]
        //                                                                                          |
        //    [source1] --> [fc01] -->  [   ]-->[gelu]-->[   ]-->[relu]-->[   ]  --> [      ]       v
        //                              |fc1]            |fc2|            |fc3|      |concat| --> [loss]
        //    [source2] --> [fc02] -->  [   ]-->[gelu]-->[   ]-->[relu]-->[   ]  --> [      ]
        //

        CPtr<CFullyConnectedLayer> fc1 = FullyConnected( complexity / 20, freeTerm )( "fc1", fc01.Ptr(), fc02.Ptr(), fc03.Ptr() );
        CPtr<CGELULayer> gelu1 = Gelu()( "gelu1", CDnnLayerLink{ fc1, 0 } );
        CPtr<CGELULayer> gelu2 = Gelu()( "gelu2", CDnnLayerLink{ fc1, 1 } );
        CPtr<CGELULayer> gelu3 = Gelu()( "gelu3", CDnnLayerLink{ fc1, 2 } );

        CPtr<CFullyConnectedLayer> fc2 = FullyConnected( complexity / 60, freeTerm )( "fc2", gelu1.Ptr(), gelu2.Ptr(), gelu3.Ptr() );
        CPtr<CReLULayer> relu1 = Relu()( "relu1", CDnnLayerLink{ fc2, 0 } );
        CPtr<CReLULayer> relu2 = Relu()( "relu2", CDnnLayerLink{ fc2, 1 } );
        CPtr<CReLULayer> relu3 = Relu()( "relu3", CDnnLayerLink{ fc2, 2 } );

        CPtr<CDropoutLayer> dropout1 = Dropout( dropoutRate )( "dp1", relu1.Ptr() );
        CPtr<CDropoutLayer> dropout2 = Dropout( dropoutRate )( "dp2", relu2.Ptr() );
        CPtr<CDropoutLayer> dropout3 = Dropout( dropoutRate )( "dp3", relu3.Ptr() );
        CPtr<CFullyConnectedLayer> fc3 = FullyConnected( 1 )( "fc3", dropout1.Ptr(), dropout2.Ptr(), dropout3.Ptr() );

        concat = ConcatChannels()( "concat",
            CDnnLayerLink{ fc3, 0 }, CDnnLayerLink{ fc3, 1 }, CDnnLayerLink{ fc3, 2 } );

    } else {
        //        +-----[fc01]- ---+
        //        |                |                              +-----------+     [target]
        //        |                v                              |           |        |
        //    [source1]     |-----------------------------------------|       v        v
        //                  |[fc1]->[gelu]->[fc2]->[relu]->[dp]->[fc3]|    [concat]->[loss]
        //    [source2]     |-----------------------------------------|       ^
        //        |                ^                              |           |
        //        |                |                              +-----------+
        //        +-----[fc02]-----+

        CPtr<CDnnHead> head = new CDnnHead(
            dnn.Random(), dnn.GetMathEngine(),
            FullyConnected( complexity / 20, freeTerm ), // "fc1"
            Gelu(),
            FullyConnected( complexity / 60, freeTerm ), // "fc2"
            Relu(),
            Dropout( dropoutRate ),
            FullyConnected( 1 ) // "fc3",
        );

        CPtr<CDnnHeadAdapterLayer> head1 = DnnHeadAdapter( head )( "head1", fc01.Ptr() );
        CPtr<CDnnHeadAdapterLayer> head2 = DnnHeadAdapter( head )( "head2", fc02.Ptr() );
        CPtr<CDnnHeadAdapterLayer> head3 = DnnHeadAdapter( head )( "head3", fc03.Ptr() );

        concat = ConcatChannels()( "concat", head1.Ptr(), head2.Ptr(), head3.Ptr() );
    }

    CPtr<CEuclideanLossLayer> loss = EuclideanLoss()( "loss", concat.Ptr(), targets.Ptr() );
    CPtr<CSinkLayer> sink = Sink( concat.Ptr(), "sink" );
    
    CPtr<CDnnAdaptiveGradientSolver> solver = new CDnnAdaptiveGradientSolver( MathEngine() );
    solver->SetLearningRate( /*learningRate*/1e-3f );
    dnn.SetSolver( solver.Ptr() );

    initializeDnnBlobs( dnn );
}

static void testDnnAdapterPerformace( bool isNaive, int interations = 1000, bool train = true )
{
    IPerformanceCounters* counters = MathEngine().CreatePerformanceCounters();
    const char* fileName = "DnnAdapter.cnnarch";

    GTEST_LOG_( INFO ) << "\n interations = " << interations << "   is_naive = " << isNaive << "\n"
        << "|" << std::setw( 10 ) << "size "
        << "|" << std::setw( 21 ) << "Train " << "|" << std::setw( 21 ) << "Inference " << "|\n"
        << "|" << std::setw( 10 ) << ""
        << "|" << std::setw( 10 ) << "time (ms) " << "|" << std::setw( 10 ) << "mem (MB) "
        << "|" << std::setw( 10 ) << "time (ms) " << "|" << std::setw( 10 ) << "mem (MB) " << "|\n";

    const int complexity = 1000;
    for( int size = 1 * complexity; size <= 4 * complexity; size += complexity ) {
        {
            CRandom random( 0 );
            CDnn dnn( random, MathEngine() );

            createDnn( dnn, isNaive, size );
            OptimizeDnn( dnn );

            dnn.CleanUp( /*force*/true );
            initializeDnnBlobs( dnn );

            MathEngine().CleanUp();
            MathEngine().ResetPeakMemoryUsage();

            if( train ) {
                dnn.RunAndLearnOnce();
                counters->Synchronise();
                for( int i = 0; i < interations; ++i ) {
                    dnn.RunAndLearnOnce();
                }
                counters->Synchronise();
            }
            CArchiveFile file( fileName, CArchive::store, GetPlatformEnv() );
            CArchive archive( &file, CArchive::store );
            archive << dnn;
        }
        double train_time = train ? ( double( ( *counters )[0].Value ) / 1000000 ) : 0.;
        double train_mem = train ? ( double( MathEngine().GetPeakMemoryUsage() ) / 1024 / 1024 ) : 0.;

        {
            CRandom random( 0 );
            CDnn dnn( random, MathEngine() );

            CArchiveFile file( fileName, CArchive::load, GetPlatformEnv() );
            CArchive archive( &file, CArchive::load );
            archive >> dnn;

            dnn.CleanUp( /*force*/true );
            initializeDnnBlobs( dnn );

            MathEngine().CleanUp();
            MathEngine().ResetPeakMemoryUsage();

            dnn.RunOnce();
            counters->Synchronise();
            for( int i = 0; i < interations; ++i ) {
                dnn.RunOnce();
            }
            counters->Synchronise();
        }
        double inference_time = double( ( *counters )[0].Value ) / 1000000;
        double inference_mem = double( MathEngine().GetPeakMemoryUsage() ) / 1024 / 1024;

        std::cout
            << "|" << std::setw( 10 ) << size
            << "|" << std::setw( 10 ) << train_time << "|" << std::setw( 10 ) << train_mem
            << "|" << std::setw( 10 ) << inference_time << "|" << std::setw( 10 ) << inference_mem << "|\n";
    }
    delete counters;
}

} // namespace NeoMLTest

//----------------------------------------------------------------------------------------------------------------------

TEST( CDnnHeadTest, DnnHeadAdapterLearnTest )
{
    CRandom random( 0x17 );
    CDnn dnn( random, MathEngine() );
    createDnn( dnn, /*isNaive*/false, /*complexity*/1000, /*dropout*/0.f );

    for( int i = 0; i < 200; ++i ) {
        dnn.RunAndLearnOnce();
    }

    EXPECT_NEAR( CheckCast<CLossLayer>( dnn.GetLayer( "loss" ).Ptr() )->GetLastLoss(), 0, 1e-3f );
}

TEST( CDnnHeadTest, DnnHeadAdapterInferenceMatch )
{
    auto runOnce = []( bool isNaive )
    {
        CRandom random( 0x11 );
        CPtr<CDnnUniformInitializer> init = new CDnnUniformInitializer( random, 0.05f, 0.05f );

        CDnn dnn( random, MathEngine() );
        dnn.SetInitializer( init.Ptr() );
        createDnn( dnn, isNaive );

        dnn.RunOnce();
        return CheckCast<CSinkLayer>( dnn.GetLayer( "sink" ).Ptr() )->GetBlob();
    };

    CPtr<CDnnBlob> expected = runOnce( /*isNaive*/false );
    CPtr<CDnnBlob> output = runOnce( /*isNaive*/true );

    EXPECT_TRUE( CompareBlobs( *expected, *output ) );
}

TEST( CDnnHeadTest, DnnHeadAdapterLearningMatch )
{
    CRandom random( 0x01 );
    CPtr<CDnnUniformInitializer> init = new CDnnUniformInitializer( random, 0.05f, 0.05f );

    CDnn dnnNoAdapters( random, MathEngine() );
    dnnNoAdapters.SetInitializer( init.Ptr() );
    createDnn( dnnNoAdapters, /*isNaive*/true, /*complexity*/1000, /*dropout*/0.f, /*freeTerm*/false );

    CRandom randomWithAdapters( 0x01 );
    CDnn dnnWithAdapters( randomWithAdapters, MathEngine() );
    dnnWithAdapters.SetInitializer( init.Ptr() );
    createDnn( dnnWithAdapters, /*isNaive*/false, /*complexity*/1000, /*dropout*/0.f, /*freeTerm*/false );

    CPtr<CLossLayer> expectedLoss = CheckCast<CLossLayer>( dnnNoAdapters.GetLayer( "loss" ).Ptr() );
    CPtr<CLossLayer> outputLoss = CheckCast<CLossLayer>( dnnWithAdapters.GetLayer( "loss" ).Ptr() );

    for( int i = 0; i < 100; ++i ) {
        dnnNoAdapters.RunAndLearnOnce();
        dnnWithAdapters.RunAndLearnOnce();
        EXPECT_NEAR( expectedLoss->GetLastLoss(), outputLoss->GetLastLoss(), 1e-3f );
    }
}

TEST( CDnnHeadTest, DnnHeadAdapterSerializationTest )
{
    CRandom random( 0 );
    CDnn dnn( random, MathEngine() );

    createDnn( dnn, /*isNaive*/false );
    dnn.RunOnce();

    CPtr<CDnnBlob> expected = CheckCast<CSinkLayer>( dnn.GetLayer( "sink" ).Ptr() )->GetBlob();
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
    initializeDnnBlobs( dnn );
    dnn.RunOnce();
    CPtr<CDnnBlob> output = CheckCast<CSinkLayer>( dnn.GetLayer( "sink" ).Ptr() )->GetBlob();
    EXPECT_TRUE( CompareBlobs( *expected, *output ) );
}

TEST( CDnnHeadTest, DISABLED_DnnHeadAdapterInferencePerfomance )
{
    DeleteMathEngine();
    testDnnAdapterPerformace( /*isNaive*/false, /*interations*/200 );

    DeleteMathEngine();
    testDnnAdapterPerformace( /*isNaive*/true, /*interations*/200 );
}
