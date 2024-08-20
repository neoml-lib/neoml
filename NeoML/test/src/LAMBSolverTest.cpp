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

using namespace NeoML;
using namespace NeoMLTest;

static NeoML::CFullyConnectedLayer* createSimpleNetwork( NeoML::CDnn& dnn )
{
    // Single FC architecture
    auto input = NeoML::Source( dnn, "Source" );
    auto fc = NeoML::FullyConnected( 16, false )( input );
    auto etalon = NeoML::Source( dnn, "etalon" );
    NeoML::CrossEntropyLoss()( fc, etalon );

    // Non-zero free term
    CPtr<CDnnBlob> freeTerm = NeoML::CDnnBlob::CreateDataBlob( dnn.GetMathEngine(), NeoML::CT_Float, 1, 1, 4);
    dnn.GetInitializer()->InitializeLayerParams( *freeTerm, 4 );
    fc->SetFreeTermData( freeTerm );

    // Simple data for forward pass
    CPtr<CDnnBlob> inputData = NeoML::CDnnBlob::CreateDataBlob( dnn.GetMathEngine(), NeoML::CT_Float, 4, 4, 4 );
    inputData->Fill<float>( 1.f );
    input->SetBlob( inputData );

    CPtr<CDnnBlob> etalonData = NeoML::CDnnBlob::CreateDataBlob( dnn.GetMathEngine(), NeoML::CT_Int, 4, 4, 1 );
    etalonData->Fill<int>( 0 );
    etalon->SetBlob( etalonData );

    return fc;
}

static float blobsDiff( const NeoML::CDnnBlob* left, const NeoML::CDnnBlob* right )
{
    NeoAssert( left != nullptr );
    NeoAssert( right != nullptr );
    const int elementsCount = left->GetDataSize();
    NeoAssert( elementsCount == right->GetDataSize() );

    NeoML::IMathEngine& me = left->GetMathEngine();

    // Summary of two vector by element abs differences
    NeoML::CFloatHandleStackVar diff( me, elementsCount );
    me.VectorSub( left->GetData(), right->GetData(),  diff, elementsCount );
    me.VectorAbs( diff, diff, elementsCount );
    NeoML::CFloatHandleStackVar summ( me, 1 );
    me.VectorSum( diff, elementsCount, summ );

    return summ.GetValue();
}

TEST(CLAMBSolverTest, ExcludeByLayerName)
{
    const auto met = MathEngine().GetType();
    if(met != MET_Cpu && met != MET_Cuda) {
        NEOML_HILIGHT( GTEST_LOG_( INFO ) ) << "Skipped rest of test for MathEngine type=" << met << " because no implementation.\n";
        // CrossEntropyLoss --> VectorEltwiseNotNegative
        return;
    }

    struct CNetwork {
        NeoML::CRandom Random;
        NeoML::CDnn Dnn;
        NeoML::CFullyConnectedLayer* FC;
        CPtr<NeoML::CDnnLambGradientSolver> Solver;

        CNetwork() :
            Dnn( Random, NeoMLTest::MathEngine() ),
            FC( createSimpleNetwork( Dnn ) ),
            Solver( new NeoML::CDnnLambGradientSolver( NeoMLTest::MathEngine() ) )
        {
            Solver->SetL2Regularization( 1.f );
            Solver->SetLearningRate( 1.f );
            Dnn.SetSolver( Solver );
        }
    };
    // DNN with default settings
    CNetwork noExclusion;
    noExclusion.Dnn.RunAndLearnOnce();

    // DNN with bias parameters excluded
    CNetwork withoutBias;
    withoutBias.Solver->ExcludeBiasParamLayers();
    withoutBias.Dnn.RunAndLearnOnce();

    // Compare main weights (should be the same) and bias (different)
    static const float epsilon = 1.e-5f;
    EXPECT_LE( blobsDiff(
        noExclusion.FC->GetWeightsData(),
        withoutBias.FC->GetWeightsData()
    ), epsilon ) << "weights are too different";
    EXPECT_GE( blobsDiff(
        noExclusion.FC->GetFreeTermData(),
        withoutBias.FC->GetFreeTermData()
    ), epsilon ) << "free terms (bias) is the same";
}