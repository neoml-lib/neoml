/* Copyright © 2017-2023 ABBYY Production LLC

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

static const int OnnxConvTransposeLayerVersion = 0;

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
		SetPaddingHeight( totalPadding[0] );
		SetPaddingWidth( totalPadding[1] );
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
		MathEngine().BlobResizeImage( outputBlobs[0]->GetDesc(), outputBlobs[0]->GetData(),
			-totalPadding[1], -totalPadding[3], -totalPadding[0], -totalPadding[2], 0.f,
			origOutput->GetDesc(), origOutput->GetData() );
		outputBlobs[0] = origOutput;
	}
}

void COnnxConvTransposeLayer::calcTotalPadding()
{
	const int convDims = 2;

	NeoPresume( OutputPadding().Size() == convDims );

	useExternalPadding = false;
	totalPadding.SetSize( 2 * convDims );
	for( int i = 0; i < convDims; ++i ) {
		int startPad = Pads().IsEmpty() ? 0 : Pads()[i];
		int endPad = Pads().IsEmpty() ? 0 : Pads()[i];
		if( Pads().IsEmpty() && !OutputShape().IsEmpty() ) {
			const int axisPad = outputDescs[0].DimSize( static_cast< int >( BD_Height ) + i )
				+ OutputPadding()[i] - OutputShape()[i + 2];
			startPad = autoPad != "SAME_UPPER" ? axisPad / 2 : ( axisPad + 1 ) / 2;
			endPad = axisPad - startPad;
		}

		totalPadding[i] = startPad;
		totalPadding[i + convDims] = endPad - OutputPadding()[i];
		useExternalPadding |= totalPadding[i] != totalPadding[i + convDims]
			|| totalPadding[i + convDims] < 0;
	}
}

CBlobDesc COnnxConvTransposeLayer::getPaddedDesc( const CBlobDesc& inputDesc )
{
	CBlobDesc paddedDesc = inputDesc;
	paddedDesc.SetDimSize( BD_Height, paddedDesc.Height() - totalPadding[0] - totalPadding[2] );
	paddedDesc.SetDimSize( BD_Width, paddedDesc.Width() - totalPadding[1] - totalPadding[3] );
	return paddedDesc;
}

} // namespace NeoML
