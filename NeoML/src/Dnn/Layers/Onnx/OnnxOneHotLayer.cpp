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

#include <NeoML/Dnn/Layers/Onnx/OnnxOneHotLayer.h>

namespace NeoML {

void onnxOneHotImpl( const CDnnBlob& input, CDnnBlob& output )
{
	IMathEngine& mathEngine = input.GetMathEngine();

	CPtr<CDnnBlob> enumBinarizationOutput = &output;
	if( output.GetDataType() == CT_Int ) {
		// Enum binarization always returns float
		// That's why we need float buffer which will be converted to integer afterwards
		enumBinarizationOutput = CDnnBlob::CreateBlob( mathEngine, CT_Float, output.GetDesc() );
	}

	if( input.GetDataType() == CT_Float ) {
		mathEngine.EnumBinarization( input.GetDataSize(), input.GetData(), output.GetChannelsCount(),
			enumBinarizationOutput->GetData() );
	} else {
		mathEngine.EnumBinarization( input.GetDataSize(), input.GetData<int>(), output.GetChannelsCount(),
			enumBinarizationOutput->GetData() );
	}

	if( output.GetDataType() == CT_Int ) {
		mathEngine.VectorConvert( enumBinarizationOutput->GetData(), output.GetData<int>(), output.GetDataSize() );
	}
}

CBlobDesc onnxOneHotOutputDesc( const CBlobDesc& inputDesc, const CDnnBlob& depthBlob )
{
	CBlobDesc outputDesc = inputDesc;
	outputDesc.SetDimSize( BD_Channels, depthBlob.GetData<int>().GetValue() );
	return outputDesc;
}

//---------------------------------------------------------------------------------------------------------------

static const int OnnxOneHotLayerVersion = 0;

void COnnxOneHotLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( OnnxOneHotLayerVersion );
	COnnxLayerBase::Serialize( archive );
}

void COnnxOneHotLayer::CalculateShapes()
{
	CheckArchitecture( GetInputCount() == 2, GetPath(), "Layer must have 2 inputs" );
	CheckArchitecture( GetOutputCount() == 1, GetPath(), "Layer must have 1 output" );
	CheckArchitecture( inputShapeBlobs[1] != nullptr, GetPath(), "Depth shape input is missing" );
	CheckArchitecture( inputShapeBlobs[1]->GetDataType() == CT_Int, GetPath(), "Depth shape must be integer" );
	CheckArchitecture( inputShapeBlobs[1]->GetDataSize() == 1, GetPath(), "Depth shape must contain 1 element" );

	if( inputShapeBlobs[0] == nullptr ) {
		CheckArchitecture( inputDescs[0].Channels() == 1, GetPath(), "Input data must have 1 channel" );
		outputDescs[0] = onnxOneHotOutputDesc( inputDescs[0], *inputShapeBlobs[1] );
		return;
	}

	CheckArchitecture( inputShapeBlobs[0]->GetChannelsCount() == 1, GetPath(), "Input data must have 1 channel" );
	CBlobDesc outputDesc = onnxOneHotOutputDesc( inputShapeBlobs[0]->GetDesc(), *inputShapeBlobs[1] );
	outputShapeBlobs[0] = CDnnBlob::CreateBlob( inputShapeBlobs[1]->GetMathEngine(), outputDesc.GetDataType(), outputDesc );
	onnxOneHotImpl( *inputShapeBlobs[0], *outputShapeBlobs[0] );
}

void COnnxOneHotLayer::RunOnce()
{
	if( inputShapeBlobs[0] != nullptr ) {
		return;
	}

	onnxOneHotImpl( *inputBlobs[0], *outputBlobs[0] );
}

} // namespace NeoML
