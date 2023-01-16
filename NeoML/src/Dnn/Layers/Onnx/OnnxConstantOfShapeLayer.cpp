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

#include <NeoML/Dnn/Layers/Onnx/OnnxConstantOfShapeLayer.h>

namespace NeoML {

static const int OnnxConstantOfShapeLayerVersion = 0;

COnnxConstantOfShapeLayer::COnnxConstantOfShapeLayer( IMathEngine& mathEngine ) :
	COnnxLayerBase( mathEngine, "OnnxConstantOfShapeLayer" )
{
	value = CDnnBlob::CreateVector( GetSingleThreadCpuMathEngine(), CT_Float, 1 );
	value->Clear();
}

void COnnxConstantOfShapeLayer::SetValue( const CDnnBlob& blob )
{
	NeoAssert( blob.GetDataSize() == 1 );
	if( blob.GetDataType() != value->GetDataType() ) {
		value = CDnnBlob::CreateVector( GetSingleThreadCpuMathEngine(), blob.GetDataType(), 1 );
	}
	value->CopyFrom( &blob );
}

void COnnxConstantOfShapeLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( OnnxConstantOfShapeLayerVersion );
	COnnxLayerBase::Serialize( archive );
	SerializeBlob( GetSingleThreadCpuMathEngine(), archive, value );
}

void COnnxConstantOfShapeLayer::CalculateShapes()
{
	CheckArchitecture( GetInputCount() == 1, GetPath(), "Layer must have 2 input" );
	CheckArchitecture( GetOutputCount() == 1, GetPath(), "Layer must have 1 output" );
	CheckArchitecture( inputShapeBlobs[0] != nullptr, GetPath(), "Input must contain shape" );
	CheckArchitecture( inputShapeBlobs[0]->GetDataSize() <= BD_Count, GetPath(), "Shape contains too many dims" );

	CBlobDesc desc( value->GetDataType() );
	CDnnBlobBuffer<int> buff( *inputShapeBlobs[0], TDnnBlobBufferAccess::Read );
	for( int i = 0; i < buff.Size(); ++i ) {
		desc.SetDimSize( i, buff[i] );
	}
	outputDescs[0] = desc;
}

void COnnxConstantOfShapeLayer::RunOnce()
{
	if( value->GetDataType() == CT_Float ) {
		outputBlobs[0]->Fill( value->GetData().GetValue() );
	} else {
		outputBlobs[0]->Fill<int>( value->GetData<int>().GetValue() );
	}
}

} // namespace NeoML
