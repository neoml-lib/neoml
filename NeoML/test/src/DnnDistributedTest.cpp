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

#ifdef NEOML_USE_OMP

using namespace NeoML;
using namespace NeoMLTest;

class CCustomDataset : public IDistributedDataset {
public:
    CCustomDataset( int _inputSize, int _labelSize )
        : inputSize( _inputSize ), labelSize( _labelSize )  {};

    int SetInputBatch( CDnn& cnn, int ) override
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
        return 1;
    }

    ~CCustomDataset(){};

private:
    int inputSize;
    int labelSize;
};

static void buildDnn( CDnn& cnn, int outputSize )
{
    CPtr<CSourceLayer> dataLayer = new CSourceLayer( cnn.GetMathEngine() );
    dataLayer->SetName( "in" );
    cnn.AddLayer( *dataLayer );

    CPtr<CFullyConnectedLayer> full = new CFullyConnectedLayer( cnn.GetMathEngine() );
    full->SetNumberOfElements( outputSize );
    full->SetName( "full" );
    full->SetZeroFreeTerm( true );
    full->Connect( *dataLayer );
    cnn.AddLayer( *full );

    CPtr<CSourceLayer> label = new CSourceLayer( cnn.GetMathEngine() );
    label->SetName( "label" );
    cnn.AddLayer( *label );

    CPtr<CEuclideanLossLayer> loss = new CEuclideanLossLayer( cnn.GetMathEngine() );
    loss->SetName( "loss" );
    loss->Connect( 0, *full );
    loss->Connect( 1, *label );
    cnn.AddLayer( *loss );

    CPtr<CSinkLayer> out = new CSinkLayer( cnn.GetMathEngine() );
    out->SetName( "sink" );
    out->Connect( *full );
    cnn.AddLayer( *out );

    CPtr<CDnnAdaptiveGradientSolver> solver = new CDnnAdaptiveGradientSolver( cnn.GetMathEngine() );
    cnn.SetSolver( solver.Ptr() );
}

TEST( CDnnDistributedTest, DnnDistributedNoArchiveTest )
{
    IMathEngine* mathEngine = CreateCpuMathEngine( 1, 0 );
    CRandom rand( 42 );

    int inputSize = 1000;
    int outputSize = 5;
    CDnn cnn( rand, *mathEngine );
    buildDnn( cnn, outputSize );

    CDistributedTraining distributed( cnn, 2 );
    CCustomDataset dataset( inputSize, outputSize );
    distributed.RunOnce( dataset );
    distributed.RunAndLearnOnce( dataset );

    CObjectArray<CDnnBlob> blobs;
    distributed.GetLastBlob( "sink", blobs );
    for( int i = 0; i < 2; i++ ) {
        ASSERT_EQ( outputSize, blobs[i]->GetDataSize() );
    }
    CArray<float> losses;
    distributed.GetLastLoss( "loss", losses );
    ASSERT_EQ( 2, losses.Size() );
    ASSERT_EQ( losses[0], losses[1] );
    ASSERT_EQ( 2, distributed.GetModelCount() );
}


TEST( CDnnDistributedTest, DnnDistributedArchiveTest )
{
    IMathEngine* mathEngine = CreateCpuMathEngine( 1, 0 );
    CRandom rand( 42 );

    int inputSize = 1000;
    int outputSize = 5;
    CDnn cnn( rand, *mathEngine );
    buildDnn( cnn, outputSize );

    CString archiveName = "distributed";
    {
        CArchiveFile storeFile( archiveName, CArchive::store, GetPlatformEnv() );
        CArchive storeArchive( &storeFile, CArchive::SD_Storing );
        storeArchive.Serialize( cnn );
    }

    CArchiveFile archiveFile( archiveName, CArchive::load, GetPlatformEnv() );
    CArchive archive( &archiveFile, CArchive::SD_Loading );
    CDistributedTraining distributed( archive, 2 );
    CCustomDataset dataset( inputSize, outputSize );
    archive.Close();
    archiveFile.Close();
    CPtr<CDnnSolver> solver = new CDnnAdaptiveGradientSolver( cnn.GetMathEngine() );

    {
        cnn.SetSolver( solver.Ptr() );
        CArchiveFile storeFile( archiveName, CArchive::store, GetPlatformEnv() );
        CArchive storeArchive( &storeFile, CArchive::SD_Storing );
        SerializeSolver( storeArchive, cnn, solver );
    }

    {
        CArchiveFile loadFile( archiveName, CArchive::load, GetPlatformEnv() );
        CArchive loadArchive( &loadFile, CArchive::SD_Loading );
        distributed.SetSolver( loadArchive );
    }

    distributed.RunAndBackwardOnce( dataset );
    distributed.Train();
    CArray<float> losses;
    distributed.GetLastLoss( "loss", losses );
    ASSERT_EQ( 2, losses.Size() );
    ASSERT_EQ( losses[0], losses[1] );
    ASSERT_EQ( 2, distributed.GetModelCount() );
}


TEST( CDnnDistributedTest, DnnDistributedSerializeTest )
{
    IMathEngine* mathEngine = CreateCpuMathEngine( 1, 0 );
    CRandom rand( 42 );

    int inputSize = 1000;
    int outputSize = 5;
    CDnn cnn( rand, *mathEngine );
    buildDnn( cnn, outputSize );

    CDistributedTraining distributed( cnn, 3 );
    CCustomDataset dataset( inputSize, outputSize );
    distributed.RunAndLearnOnce( dataset );
    distributed.RunOnce( dataset );

    CArray<float> losses;
    distributed.GetLastLoss( "loss", losses );

    CString archiveName = "distributedSerialized";
    {
        CArchiveFile archiveFile( archiveName, CArchive::store, GetPlatformEnv() );
        CArchive archive( &archiveFile, CArchive::SD_Storing );
        distributed.Serialize( archive );
    }

    CRandom rand2( 42 );
    CDnn serializedCnn( rand2, *mathEngine );
    {
        CArchiveFile archiveFile( archiveName, CArchive::load, GetPlatformEnv() );
        CArchive archive( &archiveFile, CArchive::SD_Loading );
        serializedCnn.Serialize( archive );
    }

    dataset.SetInputBatch( serializedCnn, 0 );
    serializedCnn.RunOnce();
    float loss = static_cast< CLossLayer* >( serializedCnn.GetLayer( "loss" ).Ptr() )->GetLastLoss();
    ASSERT_EQ( loss, losses[0] );

    CArray<float> distributedWeights;
    CPtr<CDnnBlob> weightsBlob = static_cast< CFullyConnectedLayer* >( serializedCnn.GetLayer( "full" ).Ptr() )->GetWeightsData();
    distributedWeights.SetSize( weightsBlob->GetDataSize() );
    weightsBlob->CopyTo( distributedWeights.GetPtr() );

    dataset.SetInputBatch( cnn, 0 );
    cnn.RunAndLearnOnce();
    CArray<float> weights;
    weightsBlob = static_cast< CFullyConnectedLayer* >( cnn.GetLayer( "full" ).Ptr() )->GetWeightsData();
    weights.SetSize( weightsBlob->GetDataSize() );
    weightsBlob->CopyTo( weights.GetPtr() );

    ASSERT_EQ( weights.Size(), distributedWeights.Size() );
    for( int i = 0; i < weights.Size(); i++ ) {
        ASSERT_NEAR( weights[i], distributedWeights[i], 1e-4 );
    }
}

#endif // NEOML_USE_OMP
