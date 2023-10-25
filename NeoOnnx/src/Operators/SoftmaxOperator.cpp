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

class CSoftmaxLayoutValidator : public ITensorLayoutValidator {
public:
	CSoftmaxLayoutValidator( int opsetVersion, int axis ) : opsetVersion( opsetVersion ), axis( axis ) {}

	bool operator()( const CTensorLayout& layout ) const override;

private:
	int opsetVersion;
	int axis;
};

bool CSoftmaxLayoutValidator::operator()( const CTensorLayout& layout ) const
{
	if( opsetVersion < 13 ) {
		for( int i = 0; i < layout.Size(); ++i ) {
			if( ( i < axis && layout[i] >= BD_Height ) // object dimension before axis
				|| ( i >= axis && layout[i] < BD_Height ) ) // batch dimension after axis
			{
				return false;
			}
		}
		return true;
	}

	if( layout[axis] == BD_ListSize ) {
		// Check compatibility with CSoftmaxLayer::NA_ListSize
		// In this case no axes after ListSize may be used
		for( const TBlobDim dim : layout ) {
			if( dim > BD_ListSize ) {
				return false;
			}
		}
	}

	return layout[axis] == BD_Channels || layout[axis] == BD_BatchLength;
}

CSoftmaxOperator::CSoftmaxOperator( const onnx::NodeProto& softmax, int opsetVersion ) :
	CLayerOperator( softmax, opsetVersion )
{
	// The differences between versions are in negative axis support
	// v13 - bfloat16 is supported, default axis value is changed, axis behavior is changed
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
}

void CSoftmaxOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CheckNoNullInputs( inputs );
	CheckNoShapeInputs( inputs );

	const int dimCount = inputs[0]->DimCount();
	// The default axis value has been changed in opset v13
	int axis = OpsetVersion >= 13 ? -1 : 1;
	GetAttribute( "axis", axis );
	if( axis < 0 ) {
		axis += dimCount;
	}
	CheckNeoOnnxSupport( axis <= 3, "more than 3 batch dimensions", *this );
	CheckNeoOnnxSupport( dimCount - axis + 1 <= 4, "more than 4 object  dimensions", *this );

	CPtr<const CUserTensor> input = AsUserTensor(
		*ConvertTensor( *inputs[0], CSoftmaxLayoutValidator( OpsetVersion, axis ) ), Name() + "_Source", dnn );

	CPtr<CSoftmaxLayer> softmax = new CSoftmaxLayer( dnn.GetMathEngine() );
	softmax->SetName( Name() );
	softmax->Connect( 0, *input->Layer(), input->OutputIndex() );
	dnn.AddLayer( *softmax );

	if( OpsetVersion >= 13 ) {
		switch( input->Layout()[axis] ) {
			case BD_Channels:
				softmax->SetNormalizationArea( CSoftmaxLayer::NA_Channel );
				break;
			case BD_BatchLength:
				softmax->SetNormalizationArea( CSoftmaxLayer::NA_BatchLength );
				break;
			case BD_ListSize:
				softmax->SetNormalizationArea( CSoftmaxLayer::NA_ListSize );
				break;
			default:
				NeoAssert( false );
		}
	}

	CBaseLayer* outLayer = softmax.Ptr();
	if( Type() == "LogSoftmax" ) {
		CPtr<CLogLayer> log = new CLogLayer( dnn.GetMathEngine() );
		log->SetName( Name() + "_Log" );
		log->Connect( *softmax );
		dnn.AddLayer( *log );
		outLayer = log.Ptr();
	}

	outputs.Add( new CUserTensor( input->Layout(), CLayerOutput( outLayer, 0 ) ) );
}

} // namespace NeoOnnx
