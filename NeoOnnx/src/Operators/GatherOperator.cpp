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

#include "../common.h"
#pragma hdrstop

#include "GatherOperator.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

#include <NeoML/Dnn/Layers/Onnx/OnnxGatherLayer.h>

using namespace NeoML;

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
	const int axis = axisAttr < 0 ? axisAttr + inputs[0]->DimCount() : axisAttr;
	CheckOnnxProtocol( axis >= 0 && axis < inputs[0]->DimCount(), "axis out of range", *this );
	CheckNeoOnnxSupport( inputs[0]->DimCount() + inputs[1]->DimCount() - 1 <= BD_Count,
		"Too much dimensions", *this );

	CPtr<const CTensorBase> indices = ConvertTensor( *inputs[1], CTensorLayout::IOLayout( inputs[1]->DimCount() ) );

	CPtr<const CTensorBase> data = inputs[0];
	CTensorLayout dataLayout;
	for( int i = 0; i < data->DimCount(); ++i ) {
		dataLayout.Add( static_cast<TBlobDim>( max( 0, indices->DimCount() - 1 ) + i ) );
	}
	if( axis != 0 ) {
		std::swap( dataLayout[0], dataLayout[axis] );
	}
	data = ConvertTensor( *data, dataLayout );

	CLayerOutput dataOutput;
	CLayerOutput indicesOutput;
	CTensorShape outputShape;
	if( HasUserInput( inputs ) ) {
		dataOutput = AsUserTensor( *data, Name() + "_Data", dnn )->LayerOutput();
		indicesOutput = AsUserTensor( *indices, Name() + "_Indices", dnn )->LayerOutput();
	} else {
		CPtr<const CShapeTensor> dataShapeTensor = AsShapeTensor( *data, Name() + "_Data", dnn );
		dataOutput = dataShapeTensor->LayerOutput();
		CPtr<const CShapeTensor> indicesShapeTensor = AsShapeTensor( *indices, Name() + "_Indices", dnn);
		indicesOutput = indicesShapeTensor->LayerOutput();
		getOutputShape( axis, dataShapeTensor->Shape(), indicesShapeTensor->Shape(), outputShape );
	}

	CPtr<COnnxGatherLayer> gatherLayer = new COnnxGatherLayer( dnn.GetMathEngine() );
	gatherLayer->SetName( Name() );
	gatherLayer->SetGatherDim( dataLayout[axis] );
	gatherLayer->Connect( 0, *dataOutput.Layer, dataOutput.OutputIndex );
	gatherLayer->Connect( 1, *indicesOutput.Layer, indicesOutput.OutputIndex );
	dnn.AddLayer( *gatherLayer );

	CTensorLayout outputLayout = getOutputLayout( axis, dataLayout, indices->Layout() );
	if( HasUserInput( inputs ) ) {
		outputs.Add( new CUserTensor( outputLayout, CLayerOutput( gatherLayer, 0 ) ) );
	} else {
		outputs.Add( new CShapeTensor( outputLayout, outputShape, CLayerOutput( gatherLayer, 0 ) ) );
	}
}

// Output layout based on data and indices layouts
CTensorLayout CGatherOperator::getOutputLayout( int axis, const CTensorLayout& dataLayout,
	const CTensorLayout& indicesLayout ) const
{
	CTensorLayout outputLayout;
	for( int i = 0; i < axis; ++i ) {
		outputLayout.Add( dataLayout[i] );
	}
	outputLayout.Add( indicesLayout );
	for( int i = axis + 1; i < dataLayout.Size(); ++i ) {
		outputLayout.Add( dataLayout[i] );
	}
	return outputLayout;
}

// Output shape based on data and indices shapes
void CGatherOperator::getOutputShape( int axis, const CTensorShape& dataShape,
	const CTensorShape& indicesShape, CTensorShape& outputShape ) const
{
	for( int i = 0; i < axis; ++i ) {
		outputShape.Add( dataShape[i] );
	}
	outputShape.Add( indicesShape );
	for( int i = axis + 1; i < dataShape.Size(); ++i ) {
		outputShape.Add( dataShape[i] );
	}
}

} // namespace NeoOnnx
