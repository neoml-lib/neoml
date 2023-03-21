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

#include "../common.h"
#pragma hdrstop

#include "TransposeOperator.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CTransposeOperator::CTransposeOperator( const onnx::NodeProto& transpose, int opsetVersion ) :
	CLayerOperator( transpose, opsetVersion )
{
	// v1 - original
	// v13 - bfloat16 is supported
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

void CTransposeOperator::AddLayers( const CTensorArray& inputs, CDnn& /* dnn */, CTensorArray& outputs ) const
{
	CheckNoNullInputs( inputs );

	const int dimCount = inputs[0]->DimCount();

	CFastArray<int, 8> perm;
	GetAttribute( "perm", perm );
	if( perm.IsEmpty() ) {
		// Default value is reverse order
		perm.SetBufferSize( dimCount );
		for( int i = 0; i < dimCount; ++i ) {
			perm.Add( dimCount - 1 - i );
		}
	}

	// Working only with layout (converters will be added by next layers when needed)
	const CTensorLayout& inputLayout = inputs[0]->Layout();
	CTensorLayout outputLayout;
	outputLayout.SetBufferSize( dimCount );

	for( int i = 0; i < dimCount; ++i ) {
		outputLayout.Add( inputLayout[perm[i]] );
	}

	static_assert( static_cast<int>( TTensorType::Count ) == 3, "TTensorType::Count != 3" );
	if( inputs[0]->Type() == TTensorType::Data ) {
		outputs.Add( new CDataTensor( outputLayout,
			*dynamic_cast<const CDataTensor*>( inputs[0].Ptr() )->Data() ) );
	} else if( inputs[0]->Type() == TTensorType::User ) {
		outputs.Add( new CUserTensor( outputLayout,
			dynamic_cast<const CUserTensor*>( inputs[0].Ptr() )->LayerOutput() ) );
	} else if( inputs[0]->Type() == TTensorType::Shape ) {
		const CShapeTensor& shapeTensor = dynamic_cast<const CShapeTensor&>( *inputs[0] );
		CTensorShape outputShape;
		outputShape.SetBufferSize( perm.Size() );
		for( int i = 0; i < dimCount; ++i ) {
			outputShape.Add( shapeTensor.Shape()[perm[i]] );
		}
		outputs.Add( new CShapeTensor( outputLayout, outputShape, shapeTensor.LayerOutput() ) );
	}
}

} // namespace NeoOnnx
