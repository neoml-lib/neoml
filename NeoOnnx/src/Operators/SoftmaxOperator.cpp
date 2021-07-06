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

#include "SoftmaxOperator.h"
#include "NeoOnnxCheck.h"
#include "TensorUtils.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CSoftmaxOperator::CSoftmaxOperator( const onnx::NodeProto& softmax, int opsetVersion ) :
	CLayerOperator( softmax, opsetVersion ),
	axis( 1 )
{
	// The differences between versions are in negative axis support
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	// Negative axis index supported since v11
	GetAttribute( "axis", axis );
	CheckOnnxProtocol( axis >= 0 || opsetVersion >= 11, "negative axis index", *this );
	CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

void CSoftmaxOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CheckOnnxProtocol( inputs[0] != nullptr, "input can't be optional", *this );
	NeoAssert( !inputs[0]->IsCalculated() );

	const int dimCount = inputs[0]->DimCount();
	CheckNeoOnnxSupport( axis <= 3, "more than 3 batch dimensions", *this );
	CheckNeoOnnxSupport( dimCount - axis + 1 <= 4, "more than 4 object  dimensions", *this );

	CTensorLayout compatibleLayout = getCompatibleLayout( dimCount, axis, inputs[0]->Layout() );
	CPtr<const CUserTensor> input = dynamic_cast<const CUserTensor*>( ConvertTensor( *inputs[0], compatibleLayout ).Ptr() );

	CPtr<CSoftmaxLayer> softmax = new CSoftmaxLayer( dnn.GetMathEngine() );
	softmax->SetName( Name() );
	softmax->Connect( 0, *input->Layer(), input->OutputIndex() );
	dnn.AddLayer( *softmax );

	outputs.Add( new CUserTensor( input->Shape(), input->Layout(), CLayerOutput( softmax, 0 ) ) );
}

CTensorLayout CSoftmaxOperator::getCompatibleLayout( int dimCount, int axis, const CTensorLayout& inputLayout ) const
{
	// Check whether input layout is compatible with softmax
	bool isCompatible = true;
	for( int i = 0; i < inputLayout.Size(); ++i ) {
		if( ( i < axis && inputLayout[i] >= BD_Height ) // object dimension before axis
			|| ( i >= axis && inputLayout[i] < BD_Height ) ) // batch dimension after axis
		{
			isCompatible = false;
			break;
		}
	}

	if( isCompatible ) {
		return inputLayout;
	}

	CTensorLayout compatibleLayout;
	compatibleLayout.SetBufferSize( dimCount );
	for( int i = 0; i < dimCount; ++i ) {
		if( i < axis ) {
			compatibleLayout[i] = static_cast<TBlobDim>( i );
		} else {
			compatibleLayout[i] = static_cast<TBlobDim>( BD_Height + i - axis );
		}
	}
	return compatibleLayout;
}

} // namespace NeoOnnx
