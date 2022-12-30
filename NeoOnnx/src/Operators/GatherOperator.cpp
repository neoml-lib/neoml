/* Copyright © 2017-2021 ABBYY Production LLC

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

#include "GatherOperator.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CGatherOperator::CGatherOperator( const onnx::NodeProto& gather, int opsetVersion ) :
	CLayerOperator( gather, opsetVersion ),
	axisAttr( 0 )
{
	// v1 - original
	// v11 - negative indices support is added
	// v13 - half data types are supported
	CheckNeoOnnxSupport( OpsetVersion >= 1 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 2, "operator must have 2 inputs", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );

	GetAttribute( "axis", axisAttr );
}

void CGatherOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CheckNoNullInputs( inputs );
	CheckNoShapeInputs( inputs );

	CObjectArray<const CUserTensor> convertedInputs;
	for( int i = 0; i < inputs.Size(); ++i ) {
		convertedInputs.Add( AsUserTensor( *inputs[i], Name() + "_Source" + Str( i ), dnn ).Ptr() );
	}

	addImageToPixelLayer( *convertedInputs[0], *convertedInputs[1], dnn, outputs );
}

void CGatherOperator::addImageToPixelLayer( const CUserTensor& data, const CUserTensor& indices,
	CDnn& dnn, CTensorArray& outputs ) const
{
	const int axis = axisAttr < 0 ? axisAttr + data.DimCount() : axisAttr;
	CheckOnnxProtocol( axis >= 0 && axis < data.DimCount(), "axis out of range", *this );

	CheckNeoOnnxSupport( data.DimCount() == 2 && indices.DimCount() == 1,
		"Gather supports only 2-dimensional data with 1-dimensional indices", *this );

	// Prepare input and indices for CImageToPixelLayer
	CTensorLayout inputLayout = axis == 0 ? CTensorLayout( { BD_Height, BD_Channels } )
		: CTensorLayout( { BD_Channels, BD_Height } );
	CPtr<const CUserTensor> preparedData = ConvertTensor( data, inputLayout );
	CPtr<const CUserTensor> preparedIndices = ConvertTensor( indices, CTensorLayout( { BD_Height } ) );

	CPtr<CImageToPixelLayer> imageLayer = new CImageToPixelLayer( dnn.GetMathEngine() );
	imageLayer->SetName( Name() );
	imageLayer->Connect( 0, *preparedData->Layer(), preparedData->OutputIndex() );
	imageLayer->Connect( 1, *preparedIndices->Layer(), preparedIndices->OutputIndex() );
	dnn.AddLayer( *imageLayer );

	CTensorLayout outputLayout = axis == 0 ? CTensorLayout( { BD_ListSize, BD_Channels } )
		: CTensorLayout( { BD_Channels, BD_ListSize } );
	outputs.Add( new CUserTensor( outputLayout, CLayerOutput( imageLayer.Ptr(), 0 ) ) );
}

} // namespace NeoOnnx
