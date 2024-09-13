/* Copyright © 2017-2024 ABBYY

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

// Applies scale and shift if needed
template<class T>
void onnxOneHotShiftAndScale( const CDnnBlob& valuesBlob, CDnnBlob& outputBlob )
{
	IMathEngine& mathEngine = outputBlob.GetMathEngine();
	CTypedMemoryHandle<const T> values = valuesBlob.GetData<const T>();
	const T offValue = values.GetValueAt( 0 );
	const T onValue = values.GetValueAt( 1 );
	CTypedMemoryHandle<T> output = outputBlob.GetData<T>();
	if( onValue - offValue != 1 ) {
		const T scale = ( onValue - offValue );
		mathEngine.VectorMultiply( output, output, outputBlob.GetDataSize(), scale );
	}
	if( offValue != 0 ) {
		const T shift = ( offValue );
		mathEngine.VectorAddValue( output, output, outputBlob.GetDataSize(), shift );
	}
}

// Calculates output data of OneHot based on the given input
void onnxOneHotImpl( const CDnnBlob& input, const CDnnBlob& values, CDnnBlob& output )
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

	if( output.GetDataType() == CT_Float ) {
		onnxOneHotShiftAndScale<float>( values, output );
	} else {
		onnxOneHotShiftAndScale<int>( values, output );
	}
}

// Calculates output size of OneHot based on the given input size and depth value
CBlobDesc onnxOneHotOutputDesc( const CBlobDesc& inputDesc, const CDnnBlob& depthBlob, const CDnnBlob& valuesBlob )
{
	CBlobDesc outputDesc = inputDesc;
	outputDesc.SetDataType( valuesBlob.GetDataType() );
	if( depthBlob.GetDataType() == CT_Int ) {
		outputDesc.SetDimSize( BD_Channels, depthBlob.GetData<int>().GetValue() );
	} else {
		outputDesc.SetDimSize( BD_Channels, static_cast<int>( depthBlob.GetData<float>().GetValue() ) );
	}
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
	CheckLayerArchitecture( GetInputCount() == 3, "Layer must have 2 inputs" );
	CheckLayerArchitecture( GetOutputCount() == 1, "Layer must have 1 output" );
	CheckLayerArchitecture( inputShapeBlobs[1] != nullptr, "Depth shape input is missing" );
	CheckLayerArchitecture( inputShapeBlobs[1]->GetDataSize() == 1, "Depth shape must contain 1 element" );
	CheckLayerArchitecture( inputShapeBlobs[2] != nullptr, "Values shape input is missing" );
	CheckLayerArchitecture( inputShapeBlobs[2]->GetDataSize() == 2, "Values shape input contain 2 elements" );

	if( inputShapeBlobs[0] == nullptr ) {
		CheckLayerArchitecture( inputDescs[0].Channels() == 1, "Input data must have 1 channel" );
		outputDescs[0] = onnxOneHotOutputDesc( inputDescs[0], *inputShapeBlobs[1], *inputShapeBlobs[2]);
		return;
	}

	CheckLayerArchitecture( inputShapeBlobs[0]->GetChannelsCount() == 1, "Input data must have 1 channel" );
	CBlobDesc outputDesc = onnxOneHotOutputDesc( inputShapeBlobs[0]->GetDesc(), *inputShapeBlobs[1], *inputShapeBlobs[2] );
	outputShapeBlobs[0] = CDnnBlob::CreateBlob( inputShapeBlobs[1]->GetMathEngine(), outputDesc.GetDataType(), outputDesc );
	onnxOneHotImpl( *inputShapeBlobs[0], *inputShapeBlobs[2], *outputShapeBlobs[0] );
}

void COnnxOneHotLayer::RunOnce()
{
	if( inputShapeBlobs[0] == nullptr ) {
		onnxOneHotImpl( *inputBlobs[0], *inputShapeBlobs[2], *outputBlobs[0] );
	}
}

} // namespace NeoML
