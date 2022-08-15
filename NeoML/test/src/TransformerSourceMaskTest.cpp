/* Copyright © 2017-2022 ABBYY Production LLC

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

TEST( CTransformerSourceMaskTest, OutputTest )
{
	CRandom random( 87 );

	const int batchSize = 3;
	const int headCount = 2;
	const int maxWidth = 3;

	// Input (widths and Q-matrix)
	CPtr<CDnnBlob> widthsBlob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Int, 1, batchSize, 1 );
	CPtr<CDnnBlob> qBlob = CDnnBlob::CreateListBlob( MathEngine(), CT_Float, 1, batchSize, maxWidth, 1 );

	CArray<int> widths = { 3, 1, 2 };
	widthsBlob->CopyFrom( widths.GetPtr() );

	// Expected output
	CArray<float> expectedOutput = {
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1,
		0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1
	};

	// Simple net
	CDnn net( random, MathEngine() );
	CPtr<CSourceLayer> widthsSourceLayer = AddLayer<CSourceLayer>( "widths", net );
	CPtr<CSourceLayer> qSourceLayer = AddLayer<CSourceLayer>( "Q", net );
	CPtr<CTransformerSourceMaskLayer> maskLayer = AddLayer<CTransformerSourceMaskLayer>( 
		"mask", { widthsSourceLayer, qSourceLayer } );
	maskLayer->SetHeadCount( headCount );
	CPtr<CSinkLayer> outputLayer = AddLayer<CSinkLayer>( "output", { maskLayer } );

	widthsSourceLayer->SetBlob( widthsBlob );
	qSourceLayer->SetBlob( qBlob );

	net.RunOnce();

	// Checking for equality of all elements
	CArray<float> outputBuff;
	outputBuff.SetSize( outputLayer->GetBlob()->GetDataSize() );
	outputLayer->GetBlob()->CopyTo( outputBuff.GetPtr() );

	for( int i = 0; i < expectedOutput.Size(); ++i ) {
		ASSERT_TRUE( FloatEq( expectedOutput[i], outputBuff[i], 1e-4f ) );
	}
}
