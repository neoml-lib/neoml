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

#include <NeoML/Dnn/Layers/Onnx/OnnxCastLayer.h>

namespace NeoML {

static void onnxCastImpl( const CDnnBlob& input, CDnnBlob& output )
{
	IMathEngine& mathEngine = input.GetMathEngine();
	if( output.GetDataType() == input.GetDataType() ) {
		output.CopyFrom( &input );
	} else if( input.GetDataType() == CT_Int ) {
		mathEngine.VectorConvert( input.GetData<int>(), output.GetData(), input.GetDataSize() );
	} else {
		mathEngine.VectorConvert( input.GetData(), output.GetData<int>(), input.GetDataSize() );
	}
}

//---------------------------------------------------------------------------------------------------------------------

COnnxCastLayer::COnnxCastLayer( IMathEngine& mathEngine ) :
	COnnxLayerBase( mathEngine, "OnnxCastLayer" ),
	outputType( CT_Float )
{
}

static const int OnnxCastLayerVersion = 0;

void COnnxCastLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( OnnxCastLayerVersion );
	COnnxLayerBase::Serialize( archive );

	int outputTypeInt = static_cast<int>( outputType );
	archive.Serialize( outputTypeInt );
	outputType = static_cast<TBlobType>( outputTypeInt );
}

void COnnxCastLayer::SetOutputType( TBlobType type )
{
	if( outputType == type ) {
		return;
	}

	outputType = type;
	ForceReshape();
}

void COnnxCastLayer::CalculateShapes()
{
	CheckArchitecture( inputDescs.Size() == 1, GetPath(), "CCastLayer must have 1 input" );
	CheckArchitecture( outputDescs.Size() == 1, GetPath(), "CCastLayer must have 1 output" );

	if( inputShapeBlobs[0] == nullptr ) {
		outputDescs[0] = inputDescs[0];
		outputDescs[0].SetDataType( outputType );
		EnableInPlace( inputDescs[0].GetDataType() == outputDescs[0].GetDataType() && InputsMayBeOverwritten() );
		return;
	}

	outputShapeBlobs[0] = CDnnBlob::CreateBlob( inputShapeBlobs[0]->GetMathEngine(), outputType,
		inputShapeBlobs[0]->GetDesc() );
	onnxCastImpl( *inputShapeBlobs[0], *outputShapeBlobs[0] );
}

void COnnxCastLayer::RunOnce()
{
	if( inputShapeBlobs[0] == nullptr ) {
		onnxCastImpl( *inputBlobs[0], *outputBlobs[0] );
	}
}

} // namespace NeoML
