/* Copyright © 2017-2020 ABBYY Production LLC

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

#include <NeoML/Dnn/Layers/ReorgLayer.h>
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

CReorgLayer::CReorgLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CCnnReorgLayer", false ),
	stride( 1 )
{
}

void CReorgLayer::Reshape()
{
	CheckInputs();
	CheckOutputs();
	// The input size should not be smaller than stride
	CheckArchitecture( min( inputDescs[0].Height(), inputDescs[0].Width() ) >= stride,
		GetName(), "reorg layer Too small input size" );

	// Division by zero if count of input channels less than stride^2
	CheckArchitecture( inputDescs[0].Channels() >= stride * stride, GetName(),
		"reorg layer Too small count of input channels" );

	CheckArchitecture( stride >= 1, GetName(), "reorg layer Too small stride" );
	CheckArchitecture( inputDescs[0].Depth() == 1, GetName(), "reorg layer Too big depth" );

	// The layer needs only one input and one output
	CheckArchitecture( GetInputCount() == 1, GetName(), "reorg layer with multiple inputs" );
	CheckArchitecture( GetOutputCount() == 1, GetName(), "reorg layer with multiple outputs" );

	// The input size should be divisible by the window size
	CheckArchitecture( inputDescs[0].Height() % stride == 0,
		GetName(), "reorg layer The height of the entrance is not a multiple of the size of the window" );
	CheckArchitecture( inputDescs[0].Width() % stride == 0,
		GetName(), "reorg layer The width of the entrance is not a multiple of the size of the window" );

	// Calculate the output size
	outputDescs[0] = inputDescs[0];
	outputDescs[0].SetDimSize( BD_Height, outputDescs[0].Height() / stride );
	outputDescs[0].SetDimSize( BD_Width, outputDescs[0].Width() / stride );
	outputDescs[0].SetDimSize( BD_Channels, outputDescs[0].Channels() * stride * stride );
}

void CReorgLayer::RunOnce()
{
	MathEngine().Reorg( inputBlobs[0]->GetDesc(), inputBlobs[0]->GetData(), stride, true,
		outputBlobs[0]->GetDesc(), outputBlobs[0]->GetData() );
}

void CReorgLayer::BackwardOnce()
{
	MathEngine().Reorg( outputDiffBlobs[0]->GetDesc(), outputDiffBlobs[0]->GetData(), stride, false,
		inputDiffBlobs[0]->GetDesc(), inputDiffBlobs[0]->GetData() );
}

static const int ReorgLayerVersion = 2000;

void CReorgLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( ReorgLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize( archive );
	
	archive.Serialize( stride );
}

int CReorgLayer::GetStride() const
{
	return stride;
}

void CReorgLayer::SetStride( int _stride )
{
	NeoAssert( _stride > 0 );
	stride = _stride;
}

CLayerWrapper<CReorgLayer> Reorg()
{
	return CLayerWrapper<CReorgLayer>( "Reorg" );
}

} // namespace NeoML
