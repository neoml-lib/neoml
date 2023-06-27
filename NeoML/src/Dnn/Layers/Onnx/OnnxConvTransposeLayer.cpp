/* Copyright © 2017-2023 ABBYY

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

#include <NeoML/Dnn/Layers/Onnx/OnnxConvTransposeLayer.h>

namespace NeoML {

static const int OnnxConvTransposeLayerVersion = 1;

void COnnxConvTransposeLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( OnnxConvTransposeLayerVersion );
	CTransposedConvLayer::Serialize( archive );
	archive.Serialize( autoPad );
	pads.Serialize( archive );
	outputPadding.Serialize( archive );
	outputShape.Serialize( archive );
}

void COnnxConvTransposeLayer::Reshape()
{
	calcTotalPadding();

	CBlobDesc origInputDesc;
	if( !useExternalPadding ) {
		paddingHeight = totalPadding[0]; //1d- convTranspose
		if( OutputPadding().Size() == /*convDims*/2 ) { //2d- convTranspose
			paddingWidth = totalPadding[1];
		}
	}

	CTransposedConvLayer::Reshape();

	if( useExternalPadding ) {
		neomlConvOutputDesc = outputDescs[0];
		outputDescs[0] = getPaddedDesc( outputDescs[0] );
	}
}

void COnnxConvTransposeLayer::RunOnce()
{
	CPtr<CDnnBlob> origOutput = nullptr;
	if( useExternalPadding ) {
		origOutput = outputBlobs[0];
		outputBlobs[0] = CDnnBlob::CreateBlob( MathEngine(), neomlConvOutputDesc );
	}

	CTransposedConvLayer::RunOnce();

	if( useExternalPadding ) {
		const bool isConv2D = ( OutputPadding().Size() == /*convDims*/2 );
		const int padding[4]{
			isConv2D ? ( -totalPadding[1] ) : 0,
			isConv2D ? ( -totalPadding[3] ) : 0,
			-totalPadding[0],
			-totalPadding[isConv2D ? 2 : 1]
		};

		MathEngine().BlobResizeImage( outputBlobs[0]->GetDesc(), outputBlobs[0]->GetData(),
			/*left*/padding[0], /*right*/padding[1], /*top*/padding[2], /*bottom*/padding[3], TBlobResizePadding::Constant,
			/*default*/0.f, origOutput->GetDesc(), origOutput->GetData() );
		outputBlobs[0] = origOutput;
	}
}

// Calculates the result padding based on all the padding-related attributes
// and determines whether padding must be done manually or CTransposedConvLayer can calculated it itself
void COnnxConvTransposeLayer::calcTotalPadding()
{
	NeoPresume( OutputPadding().Size() == /*convDims*/2 || OutputPadding().Size() == /*convDims*/1 );
	const int convDims = OutputPadding().Size();

	useExternalPadding = false;
	totalPadding.SetSize( 2 * convDims );

	for( int i = 0; i < convDims; ++i ) {
		int startPad = Pads().IsEmpty() ? 0 : Pads()[i];
		int endPad = Pads().IsEmpty() ? 0 : Pads()[i];
		if( Pads().IsEmpty() && !OutputShape().IsEmpty() ) {
			const int axisPad = outputDescs[0].DimSize( static_cast<int>( BD_Height ) + i )
				+ OutputPadding()[i] - OutputShape()[i + 2];
			startPad = ( ( autoPad != "SAME_UPPER" ) ? axisPad : ( axisPad + 1 ) ) / 2;
			endPad = axisPad - startPad;
		}

		totalPadding[i] = startPad;
		totalPadding[i + convDims] = endPad - OutputPadding()[i];
		useExternalPadding |= totalPadding[i] != totalPadding[i + convDims] || totalPadding[i + convDims] < 0;
	}
}

// Calculates the desc after the external padding
CBlobDesc COnnxConvTransposeLayer::getPaddedDesc( const CBlobDesc& inputDesc )
{
	CBlobDesc paddedDesc = inputDesc;
	const bool isConv2D = ( OutputPadding().Size() == /*convDims*/2 );
	paddedDesc.SetDimSize( BD_Height, paddedDesc.Height() - totalPadding[0] - totalPadding[isConv2D ? 2 : 1] ); //1d- convTranspose
	if( isConv2D ) { //2d- convTranspose
		paddedDesc.SetDimSize( BD_Width, paddedDesc.Width() - totalPadding[1] - totalPadding[3] );
	}
	return paddedDesc;
}

} // namespace NeoML
