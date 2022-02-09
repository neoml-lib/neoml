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

using namespace NeoML;
using namespace NeoMLTest;

class CCustomDataset : public IDistributedDataset {
public:
    CCustomDataset( int _inputSize, int _labelSize )
        : inputSize( _inputSize ), labelSize( _labelSize )  {};

    void SetInputBatch( CDnn& cnn, int thread ) override
    {
        CArray<float> inArr;
        inArr.Add( 1, inputSize );
        CPtr<CDnnBlob> in = CDnnBlob::CreateTensor( cnn.GetMathEngine(), CT_Float, { 1, 1, 1, 1, 1, 1, inputSize} );
        in->CopyFrom( inArr.GetPtr() );
        CArray<float> labelArr;
        labelArr.Add( 1, labelSize );
        CPtr<CDnnBlob> labels = CDnnBlob::CreateTensor( cnn.GetMathEngine(), CT_Float, { 1, 1, 1, 1, 1, 1, labelSize } );
        labels->CopyFrom( labelArr.GetPtr() );
        CheckCast<CSourceLayer>( cnn.GetLayer( "in" ) )->SetBlob( in );
        CheckCast<CSourceLayer>( cnn.GetLayer( "label" ) )->SetBlob( labels );
    }

    ~CCustomDataset(){};

private:
    int inputSize;
    int labelSize;
};

TEST( CDnnDistributedTest, DnnDistributedBasicTest )
{
    IMathEngine* mathEngine = CreateCpuMathEngine( 1, 0 );
    CRandom rand( 42 );

    int inputSize = 1000;
    int outputSize = 2;
    CDnn cnn( rand, *mathEngine );
    CPtr<CSourceLayer> dataLayer = new CSourceLayer( *mathEngine );
    dataLayer->SetName( "in" );
    cnn.AddLayer( *dataLayer );

    CPtr<CFullyConnectedLayer> full = new CFullyConnectedLayer( *mathEngine );
    full->SetNumberOfElements( outputSize );
    full->SetName( "full" );
    full->Connect( *dataLayer );
    cnn.AddLayer( *full );

    CPtr<CSourceLayer> label = new CSourceLayer( *mathEngine );
    label->SetName( "label" );
    cnn.AddLayer( *label );

    CPtr<CEuclideanLossLayer> loss = new CEuclideanLossLayer( *mathEngine );
    loss->SetName( "loss" );
    loss->Connect( 0, *full );
    loss->Connect( 1, *label );
    cnn.AddLayer( *loss );

    CPtr<CSinkLayer> out = new CSinkLayer( *mathEngine );
    out->SetName( "sink" );
    out->Connect( *full );
    cnn.AddLayer( *out );

    CPtr<CDnnSimpleGradientSolver> solver = new CDnnSimpleGradientSolver( *mathEngine );
    solver->SetLearningRate( 1e-5f );
    cnn.SetSolver( solver.Ptr() );

    CDistributedTraining distributed( cnn, 2 );
    CCustomDataset dataset( inputSize, outputSize );
    distributed.RunAndLearnOnce( dataset );

    CObjectArray<CDnnBlob> blobs;
    distributed.GetLastBlob( "sink", blobs );

    CArray<float> losses;
    distributed.GetLastLoss( "loss", losses );

    ASSERT_EQ( 2, distributed.GetModelCount() );
}


