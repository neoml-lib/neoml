/* Copyright © 2023-2024 ABBYY

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

#include <NeoML/Dnn/Rowwise/Activation.h>

using namespace NeoML;
using namespace NeoMLTest;

static constexpr int RowwiseTestChannels = 16;

static CPtr<CDnnBlob> rowwiseSampleInput( CRandom& random )
{
	CPtr<CDnnBlob> blob = CDnnBlob::Create2DImageBlob( MathEngine(), CT_Float, 2, 3, 512, 512, RowwiseTestChannels );
	CDnnBlobBuffer<float> buffer( *blob, TDnnBlobBufferAccess::Write );
	for( int i = 0; i < buffer.Size(); ++i ) {
		buffer[i] = static_cast<float>( random.Uniform( -5., 5. ) );
	}
	return blob;
}

typedef CBaseLayer* ( *TChainBuilder )( CSourceLayer* source );

static void rowwiseTestImpl( TChainBuilder buildChain, int seed )
{
	CRandom random( seed );
	CDnn dnn( random, MathEngine() );
	CSourceLayer* source = Source( dnn, "source" );
	CBaseLayer* chainOutput = buildChain( source );
	CSinkLayer* sink = Sink( chainOutput, "sink" );

	CPtr<CDnnBlob> inputBlob = rowwiseSampleInput( random );
	CPtr<CDnnBlob> originalInput = inputBlob->GetCopy();

	source->SetBlob( inputBlob );
	dnn.RunOnce();
	CPtr<CDnnBlob> originalOutput = sink->GetBlob()->GetCopy();

	// Just to be on a safe side
	// Let's check that layers didn't overwrite input data
	EXPECT_TRUE( CompareBlobs( *originalInput, *source->GetBlob() ) );

	CDnnOptimizationReport report = OptimizeDnn( dnn );
	EXPECT_EQ( 1, report.RowwiseChainCount );
	EXPECT_EQ( 3, dnn.GetLayerCount() );

	dnn.RunOnce();

	// Check that rowwise doesn't overwrite its input
	EXPECT_TRUE( CompareBlobs( *originalInput, *source->GetBlob() ) );
	// Check that rowwise returns the same output
	EXPECT_TRUE( CompareBlobs( *originalOutput, *sink->GetBlob() ) );
}

TEST( RowwiseTest, ActivationOp )
{
	const auto met = MathEngine().GetType();
	if(met != MET_Cpu && met != MET_Cuda) {
		NEOML_HILIGHT( GTEST_LOG_( INFO ) ) << "Skipped rest of test for MathEngine type=" << met << " because no implementation.\n";
		return;
	}

	auto buildChain = [] ( CSourceLayer* source ) -> CBaseLayer* {
		CBaseLayer* curr = source;
		curr = Elu( 0.01f )( curr );
		curr = LeakyRelu()( curr );
		curr = HardSigmoid( 0.5f, 0.5f )( curr );
		curr = HardTanh()( curr );
		curr = HSwish()( curr );
		curr = Linear( -2.f, 1.f )( curr );
		curr = Relu()( curr );
		curr = Linear( 3.f, -1.5f )( curr );
		curr = LeakyRelu( 0.666f )( curr );
		curr = Elu( 0.9f )( curr );
		curr = Sigmoid()( curr );
		curr = HardSigmoid( 0.1f, 0.2f )( curr );
		curr = Relu( 0.5f )( curr );
		curr = Tanh()( curr );
		return curr;
	};
	rowwiseTestImpl( buildChain, 0xBADF00D );
}

TEST( RowwiseTest, ChannelwiseConvOp )
{
	const auto met = MathEngine().GetType();
	if(met != MET_Cpu && met != MET_Cuda) {
		NEOML_HILIGHT( GTEST_LOG_( INFO ) ) << "Skipped rest of test for MathEngine type=" << met << " because no implementation.\n";
		return;
	}

	auto buildChain = [] ( CSourceLayer* source ) -> CBaseLayer* {
		CBaseLayer* curr = source;
		curr = ChannelwiseConv( RowwiseTestChannels, CConvAxisParams( 3 ), CConvAxisParams( 3 ), true )( curr );
		curr = ChannelwiseConv( RowwiseTestChannels, CConvAxisParams( 3, 1, 2 ), CConvAxisParams( 3, 1, 2 ), false )( curr );
		curr = ChannelwiseConv( RowwiseTestChannels, CConvAxisParams( 7, 2, 2 ), CConvAxisParams( 1, 0, 2 ), true )( curr );
		curr = ChannelwiseConv( RowwiseTestChannels, CConvAxisParams( 1, 0, 2 ), CConvAxisParams( 7, 3, 2 ), false )( curr );
		curr = ChannelwiseConv( RowwiseTestChannels, CConvAxisParams( 1, 0, 1 ), CConvAxisParams( 1, 0, 1 ), true )( curr );
		curr = ChannelwiseConv( RowwiseTestChannels, CConvAxisParams( 9, 8, 9 ), CConvAxisParams( 5, 4, 5 ), false )( curr );
		return curr;
	};
	rowwiseTestImpl( buildChain, 0xBADFACE );
}

TEST( RowwiseTest, ConvOp )
{
	const auto met = MathEngine().GetType();
	if(met != MET_Cpu && met != MET_Cuda) {
		NEOML_HILIGHT( GTEST_LOG_( INFO ) ) << "Skipped rest of test for MathEngine type=" << met << " because no implementation.\n";
		return;
	}

	auto buildChain = [] ( CSourceLayer* source ) -> CBaseLayer* {
		CBaseLayer* curr = source;
		curr = Conv( 34, CConvAxisParams( 4, 3, 2, 1 ), CConvAxisParams( 5, 4, 3, 2 ), true )( curr );
		curr = Conv( 46, CConvAxisParams( 5, 4, 3, 2 ),CConvAxisParams( 4, 3, 2, 1 ),  false )( curr );
		curr = Conv( 31, CConvAxisParams( 1 ), CConvAxisParams( 1 ), false )( curr );
		curr = Conv( 23, CConvAxisParams( 1 ), CConvAxisParams( 1 ), true )( curr );
		curr = Conv( 37, CConvAxisParams( 9, 7, 5, 3 ), CConvAxisParams( 9, 7, 5, 3 ), false )( curr );
		curr = Conv( 19, CConvAxisParams( 9, 7, 5, 3 ), CConvAxisParams( 9, 7, 5, 3 ), true )( curr );
		return curr;
	};
	rowwiseTestImpl( buildChain, 0xF00DFACE );
}

TEST( RowwiseTest, PoolingOp )
{
	const auto met = MathEngine().GetType();
	if(met != MET_Cpu && met != MET_Cuda) {
		NEOML_HILIGHT( GTEST_LOG_( INFO ) ) << "Skipped rest of test for MathEngine type=" << met << " because no implementation.\n";
		return;
	}

	auto buildChain = [] ( CSourceLayer* source ) -> CBaseLayer* {
		CBaseLayer* curr = source;
		curr = MaxPooling( 3, 3 )( curr );
		curr = MeanPooling( 3, 3 )( curr );
		curr = MaxPooling( 7, 9, 2, 2 )( curr );
		curr = MaxPooling( 9, 5, 3, 5 )( curr );
		curr = MeanPooling( 7, 9, 2, 2 )( curr );
		curr = MeanPooling( 9, 5, 3, 5 )( curr );
		return curr;
	};
	rowwiseTestImpl( buildChain, 0xFACEF00D );
}

TEST( RowwiseTest, ResizeImageOp )
{
	const auto met = MathEngine().GetType();
	if(met != MET_Cpu && met != MET_Cuda) {
		NEOML_HILIGHT( GTEST_LOG_( INFO ) ) << "Skipped rest of test for MathEngine type=" << met << " because no implementation.\n";
		return;
	}

	auto buildChain = [] ( CSourceLayer* source ) -> CBaseLayer* {
		CBaseLayer* curr = source;
		curr = ImageResize( 2, 3, 4, 5, 0.f, TBlobResizePadding::Edge )( curr );
		curr = ImageResize( 2, 3, 4, 5, 2.f, TBlobResizePadding::Constant )( curr );
		curr = ImageResize( 2, 3, 4, 5, 0.f, TBlobResizePadding::Reflect )( curr );
		curr = ImageResize( -10, -10, 20, 20, 666.f, TBlobResizePadding::Constant )( curr );
		curr = ImageResize( 20, 20, -10, -10, 0.f, TBlobResizePadding::Reflect )( curr );
		curr = ImageResize( -10, 20, 20, -10, 0.f, TBlobResizePadding::Edge )( curr );
		curr = ImageResize( 3, 1, 5, 7, 88.f, TBlobResizePadding::Constant )( curr );
		return curr;
	};
	rowwiseTestImpl( buildChain, 0xBADBEE );
}

TEST( RowwiseTest, Optimize2Chains )
{
	const auto met = MathEngine().GetType();
	if(met != MET_Cpu && met != MET_Cuda) {
		NEOML_HILIGHT( GTEST_LOG_( INFO ) ) << "Skipped rest of test for MathEngine type=" << met << " because no implementation.\n";
		return;
	}

	auto buildChain = [] ( CSourceLayer* source ) -> CBaseLayer* {
		IMathEngine& mathEngine = source->MathEngine();
		CDnn& dnn = *source->GetDnn();

		CPtr<CRowwiseOperationChainLayer> firstChain = new CRowwiseOperationChainLayer( mathEngine );
		firstChain->SetName( "firstChain" );
		firstChain->AddOperation( new CRowwiseActivation( mathEngine,
			CActivationDesc( AF_ELU, CELUActivationParam{ 0.01f } ) ) );
		firstChain->AddOperation( new CRowwiseActivation( mathEngine,
			CActivationDesc( AF_LeakyReLU, CLeakyReLUActivationParam() ) ) );
		dnn.AddLayer( *firstChain );
		firstChain->Connect( *source );

		CPtr<CRowwiseOperationChainLayer> secondChain = new CRowwiseOperationChainLayer( mathEngine );
		secondChain->SetName( "secondChain" );
		secondChain->AddOperation( new CRowwiseActivation( mathEngine,
			CActivationDesc( AF_HardSigmoid, CHardSigmoidActivationParam() ) ) );
		secondChain->AddOperation( new CRowwiseActivation( mathEngine, CActivationDesc( AF_HardTanh ) ) );
		dnn.AddLayer( *secondChain );
		secondChain->Connect( *firstChain );

		return secondChain.Ptr();
	};
	rowwiseTestImpl( buildChain, 0xBEE );
}

TEST( RowwiseTest, OptimizeOpInFrontOfChain )
{
	const auto met = MathEngine().GetType();
	if(met != MET_Cpu && met != MET_Cuda) {
		NEOML_HILIGHT( GTEST_LOG_( INFO ) ) << "Skipped rest of test for MathEngine type=" << met << " because no implementation.\n";
		return;
	}

	auto buildChain = [] ( CSourceLayer* source ) -> CBaseLayer* {
		CBaseLayer* curr = source;
		IMathEngine& mathEngine = source->MathEngine();
		CDnn& dnn = *source->GetDnn();

		curr = Elu()( curr );

		CPtr<CRowwiseOperationChainLayer> chain = new CRowwiseOperationChainLayer( mathEngine );
		chain->SetName( "chain" );
		chain->AddOperation( new CRowwiseActivation( mathEngine,
			CActivationDesc( AF_LeakyReLU, CLeakyReLUActivationParam() ) ) );
		chain->AddOperation( new CRowwiseActivation( mathEngine,
			CActivationDesc( AF_HardSigmoid, CHardSigmoidActivationParam() ) ) );
		chain->AddOperation( new CRowwiseActivation( mathEngine, CActivationDesc( AF_HardTanh ) ) );
		dnn.AddLayer( *chain );
		chain->Connect( *curr );

		return chain.Ptr();
	};
	rowwiseTestImpl( buildChain, 0xBEE );
}
