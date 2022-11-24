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

// Checks that operator can be emulated by CSubSequenceLayer
static bool canBeEmulatedBySubSequence( const CTensorBase& indices )
{
	if( !indices.IsCalculated() ) {
		return false;
	}

	for( int i = 0; i < indices.DimCount(); ++i ) {
		if( indices.Shape()[i] != 1 ) {
			return false;
		}
	}

	return true;
}

// Converts tensor to onnx-compatible layout
static CPtr<const CUserTensor> convertToOnnx( const CUserTensor& data )
{
	if( !IsTransposedLayout( data.Layout() ) ) {
		return &data;
	}

	return ConvertTensor( data, CTensorLayout( data.DimCount() ) );
}

// Returns the shape after the image to pixel layer
static void getImageToPixelShape( const CTensorShape& dataShape, int axis, const CTensorShape& indicesShape,
	CTensorShape& resultShape )
{
	NeoAssert( dataShape.Size() > axis && axis >= 0 );
	const int resultDimCount = indicesShape.Size() + dataShape.Size() - 1;
	indicesShape.CopyTo( resultShape );
	resultShape.SetBufferSize( resultDimCount );
	for( int i = 0; i < dataShape.Size() - 1; ++i ) {
		resultShape.Add( i < axis ? dataShape[i] : dataShape[i + 1] );
	}
	NeoAssert( resultShape.Size() == resultDimCount );
}

// Transforms layer output into layout, expected by onnx
static CPtr<const CUserTensor> transformOutput( const CBaseLayer& layer, const CTensorShape& resultShape, CDnn& dnn )
{
	CTensorLayout resultLayout( resultShape.Size() );
	CPtr<CTransformLayer> transform = new CTransformLayer( dnn.GetMathEngine() );
	transform->SetName( CString( layer.GetName() ) + "_transformOutput" );
	NeoAssert( layer.GetDnn() != nullptr );
	for( TBlobDim dim = BD_BatchLength; dim < BD_Count; ++dim ) {
		const int dimIndex = resultLayout.Find( dim );
		transform->SetDimensionRule( dim, CTransformLayer::O_SetSize,
			dimIndex == NotFound ? 1 : resultShape[dimIndex] );
	}
	transform->Connect( layer );
	dnn.AddLayer( *transform );
	return new CUserTensor( resultShape, resultLayout, CLayerOutput( transform, 0 ) );
}

//---------------------------------------------------------------------------------------------------------------------

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
	CheckOnnxProtocol( inputs[0] != nullptr, "input can't be optional", *this );
	CheckOnnxProtocol( inputs[1] != nullptr, "input can't be optional", *this );

	CPtr<const CUserTensor> data = AsUserTensor( *inputs[0], Name() + "_Source_Data", dnn );

	if( canBeEmulatedBySubSequence( *inputs[1] ) ) {
		addSubSequenceLayer( *data, dynamic_cast<const CDataTensor&>( *inputs[1] ), dnn, outputs );
	} else {
		CPtr<const CUserTensor> indices = AsUserTensor( *inputs[1], Name() + "_Source_Indices", dnn );
		addImageToPixelLayer( *data, *indices, dnn, outputs );
	}
}

void CGatherOperator::addImageToPixelLayer( const CUserTensor& data, const CUserTensor& indices,
	CDnn& dnn, CTensorArray& outputs ) const
{
	const int axis = axisAttr < 0 ? axisAttr + data.DimCount() : axisAttr;
	CheckOnnxProtocol( axis >= 0 && axis < data.DimCount(), "axis out of range", *this );

	// Prepare data blob
	CPtr<const CUserTensor> currData = &data;
	// Step 1: move axis to BD_BatchLength with rest of dims ordered in onnx-compatible way
	CTensorLayout requiredLayout = data.Layout();
	for( int i = 0; i < requiredLayout.Size(); ++i ) {
		requiredLayout[i] = i == axis ? BD_BatchLength : static_cast<TBlobDim>( i + 1 );
	}
	currData = ConvertTensor( *currData, requiredLayout );
	// CImageLookupLayer gathers pixel of BD_Channels size from the set of BD_Height x BD_Width size
	// Step 2: move BD_BatchLength to BD_Height and the rest of the blob to the BD_Channels
	CPtr<CTransformLayer> transformToImage = new CTransformLayer( dnn.GetMathEngine() );
	transformToImage->SetName( Name() + "_TransformToImage" );
	for( TBlobDim dim = BD_BatchLength; dim < BD_Count; ++dim ) {
		if( dim == BD_Height ) {
			transformToImage->SetDimensionRule( dim, CTransformLayer::O_SetSize, data.Shape()[axis] );
		} else if( dim == BD_Channels ) {
			transformToImage->SetDimensionRule( dim, CTransformLayer::O_Remainder, 1 );
		} else {
			transformToImage->SetDimensionRule( dim, CTransformLayer::O_SetSize, 1 );
		}
	}
	transformToImage->Connect( 0, *currData->Layer(), currData->OutputIndex() );
	dnn.AddLayer( *transformToImage );

	// Prepare indices blob
	// CImageLookupLayer expects indices of single image to be in BD_Channels
	CPtr<const CUserTensor> currIndices = convertToOnnx( indices );
	CPtr<CTransformLayer> transformIndices = new CTransformLayer( dnn.GetMathEngine() );
	transformIndices->SetName( Name() + "_TransformIndices" );
	for( TBlobDim dim = BD_BatchLength; dim < BD_Channels; ++dim ) {
		transformIndices->SetDimensionRule( dim, CTransformLayer::O_SetSize, 1 );
	}
	transformIndices->SetDimensionRule( BD_Channels, CTransformLayer::O_Remainder, 1 );
	transformIndices->Connect( 0, *currIndices->Layer(), currIndices->OutputIndex() );
	dnn.AddLayer( *transformIndices );

	// Perform gather
	CPtr<CImageToPixelLayer> imageToPixel = new CImageToPixelLayer( dnn.GetMathEngine() );
	imageToPixel->SetName( Name() );
	imageToPixel->Connect( 0, *transformToImage );
	imageToPixel->Connect( 1, *transformIndices );
	dnn.AddLayer( *imageToPixel );

	// All the indices dims are compressed in BD_ListSize (in onnx-compatible order)
	// All the data dims are compressed in BD_Channels (in onnx-compatible order)
	// Unpack them into onnx user tensor
	CTensorShape imageToPixelShape;
	getImageToPixelShape( data.Shape(), axis, indices.Shape(), imageToPixelShape );
	CPtr<const CUserTensor> currOutput = transformOutput( *imageToPixel, imageToPixelShape, dnn ).Ptr();

	// currOutput contains output with dims in the following order:
	// indices_dim_0, ... , indices_dim_q-1, data_dim_0, ... , data_dim_r-1
	// Now we have to transpose dims in the following way:
	// data_dim_0, ... , data_dim_axis-1, indices_dim_0, ... , indices_dim_q-1, data_dim_axis+1, ... , data_dim_r-1
	// For optimization purposes here we just reinterpret current tensor (by changing shape and layout)
	CTensorShape outputShape;
	outputShape.SetBufferSize( data.DimCount() + indices.DimCount() - 1 );
	CTensorLayout outputLayout;
	outputLayout.SetBufferSize( data.DimCount() + indices.DimCount() - 1 );
	for( int dataDim = 0; dataDim < axis; ++dataDim ) {
		outputShape.Add( currOutput->Shape()[indices.DimCount() + dataDim] );
		outputLayout.Add( currOutput->Layout()[indices.DimCount() + dataDim] );
	}
	for( int indicesDim = 0; indicesDim < indices.DimCount(); ++indicesDim ) {
		outputShape.Add( currOutput->Shape()[indicesDim] );
		outputLayout.Add( currOutput->Layout()[indicesDim] );
	}
	for( int dataDim = axis + 1; dataDim < data.DimCount(); ++dataDim ) {
		outputShape.Add( currOutput->Shape()[indices.DimCount() + dataDim - 1] );
		outputLayout.Add( currOutput->Layout()[indices.DimCount() + dataDim - 1] );
	}
	outputs.Add( new CUserTensor( outputShape, outputLayout, currOutput->LayerOutput() ) );
}

void CGatherOperator::addSubSequenceLayer( const CUserTensor& data, const CDataTensor& indices,
	CDnn& dnn, CTensorArray& outputs ) const
{
	CPtr<const CUserTensor> currData = &data;

	const int axis = axisAttr < 0 ? axisAttr + data.DimCount() : axisAttr;
	CheckOnnxProtocol( axis >= 0 && axis < currData->DimCount(), "axis out of range", *this );

	// For the proper work of CSubSequenceLayer first axis of data must be in BD_BatchLength
	if( currData->Layout()[axis] != BD_BatchLength ) {
		CTensorLayout newLayout = currData->Layout();
		const int batchLenIndex = newLayout.Find( BD_BatchLength );
		if( batchLenIndex != NotFound ) {
			swap( newLayout[axis], newLayout[batchLenIndex] );
		} else {
			newLayout[axis] = BD_BatchLength;
		}
		currData = ConvertTensor( *currData, newLayout );
	}

	CheckOnnxProtocol( indices.Data()->GetDataType() == CT_Int, "non-integer indices", *this );
	const int gatherIndex = indices.Data()->GetData<int>().GetValue();

	CPtr<CSubSequenceLayer> subSeqLayer = new CSubSequenceLayer( dnn.GetMathEngine() );
	subSeqLayer->SetName( Name() );
	subSeqLayer->SetStartPos( gatherIndex );
	subSeqLayer->SetLength( 1 );
	subSeqLayer->Connect( 0, *currData->Layer(), currData->OutputIndex() );
	dnn.AddLayer( *subSeqLayer );

	const int outputDimCount = currData->DimCount() + indices.DimCount() - 1;
	CTensorShape outputShape;
	CTensorLayout outputLayout;

	for( int i = 0; i < axis; ++i ) {
		outputShape.Add( currData->Shape()[i] );
		outputLayout.Add( currData->Layout()[i] );
	}

	TBlobDim unusedDim = BD_BatchLength;
	for( int i = 0; i < indices.DimCount(); ++i ) {
		outputShape.Add( indices.Shape()[i] );
		while( currData->Layout().Find( unusedDim ) != NotFound ) {
			++unusedDim;
		}
		outputLayout.Add( unusedDim );
		++unusedDim;
	}

	for( int i = axis + 1; i < currData->DimCount(); ++i ) {
		outputShape.Add( currData->Shape()[i] );
		outputLayout.Add( currData->Layout()[i] );
	}

	outputs.Add( new CUserTensor( outputShape, outputLayout, CLayerOutput( subSeqLayer.Ptr(), 0 ) ) );
}

} // namespace NeoOnnx
