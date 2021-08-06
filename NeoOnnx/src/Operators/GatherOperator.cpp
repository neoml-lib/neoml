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

// Converts tensor to onnx-compatible layout
static CPtr<const CTensorBase> convertToOnnx( const CTensorBase& data )
{
	if( !IsTransposedLayout( data.Layout() ) ) {
		return &data;
	}

	return ConvertTensor( data, CTensorLayout( data.DimCount() ) );
}

// Returns input data as a blob where:
// - BD_BatchLength - number of objects to gather from in original order
// - others - objects data in onnx-compatible order
static CPtr<const CDnnBlob> prepareDataBlob( const CDataTensor& data, int axis )
{
	NeoAssert( data.DimCount() > axis && axis >= 0 );

	// Make sure that data is in onnx-compatible order
	CPtr<const CDataTensor> onnxData( dynamic_cast<const CDataTensor*>( convertToOnnx( data ).Ptr() ) );

	// Move axis dimension to the BD_BatchLength (the order of the rest of the dimensions must stay the same)
	CPtr<const CDnnBlob> preparedBlob = data.Data();
	int axisDim = static_cast<int>( onnxData->Layout()[axis] );
	while( axisDim != static_cast<int>( BD_BatchLength ) ) {
		preparedBlob = preparedBlob->GetTransposed( axisDim - 1, axisDim );
		axisDim -= 1;
	}

	NeoAssert( preparedBlob->GetBatchLength() == data.Shape()[axis] );
	return preparedBlob;
}

// Gathers result from data based on indices
// Axis used in gathering must be BD_BatchLength
template<class T>
static void gather( const CGatherOperator& op, const CDnnBlob& data, const CDnnBlob& indices, CDnnBlob& result )
{
	const int objectCount = data.GetBatchLength();
	const int objectSize = data.GetDataSize() / objectCount;
	const int indicesSize = indices.GetDataSize();
	NeoAssert( result.GetDataSize() == indicesSize * objectSize );
	IMathEngine& mathEngine = data.GetMathEngine();

	CTypedMemoryHandle<const T> dataPtr = data.GetData<T>();
	const int* indicesBuff = const_cast<CDnnBlob&>( indices ).GetBuffer<int>( 0, indices.GetDataSize(), true);
	CTypedMemoryHandle<T> resultPtr = result.GetData<T>();

	try {
		for( int i = 0; i < indicesSize; ++i ) {
			const int index = indicesBuff[i] < 0 ? indicesBuff[i] + objectCount : indicesBuff[i];
			CheckOnnxProtocol( index >= 0 && index < objectCount, "index outside of bounds", op );
			mathEngine.VectorCopy( resultPtr, dataPtr + index * objectSize, objectSize );
			resultPtr += objectSize;
		}
	} catch( ... ) {
		const_cast<CDnnBlob&>( indices ).ReleaseBuffer( const_cast<int*>( indicesBuff ), false );
	}
	const_cast<CDnnBlob&>( indices ).ReleaseBuffer( const_cast<int*>( indicesBuff ), false );
}

// Returns the shape of the result of gather operator
static void getResultShape( const CTensorShape& dataShape, int axis, const CTensorShape& indicesShape,
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

// Returns the blob desc of the result of gather operator
static CBlobDesc getResultBlobDesc( const CTensorShape& resultShape, const CTensorLayout& resultLayout, TBlobType dataType )
{
	NeoAssert( resultShape.Size() == resultLayout.Size() );
	CBlobDesc resultDesc( dataType );
	for( int i = 0; i < resultShape.Size(); ++i ) {
		resultDesc.SetDimSize( resultLayout[i], resultShape[i] );
	}
	return resultDesc;
}

// Prepares indices user tensor for the lookup layer
static CLayerOutput prepareLookupIndices( const CUserTensor& indices, const CString& opName, CDnn& dnn )
{
	// Lookup with one table expects the indices in the following format:
	// - BD_Channels == 1
	// - Other dims == Number of objects to lookup from table
	// - (additional) indices should be compatible with onnx

	// Step 1: make sure that indices are compatible with onnx
	CPtr<const CUserTensor> currIndices = dynamic_cast<const CUserTensor*>( convertToOnnx( indices ).Ptr() );

	// Step 2: transform into BD_Channels == 1 if needed
	const int channelsDimIndex = currIndices->Layout().Find( BD_Channels );
	if( channelsDimIndex == NotFound || currIndices->Shape()[channelsDimIndex] == 1 ) {
		return currIndices->LayerOutput();
	}
	CPtr<CTransformLayer> transform = new CTransformLayer( dnn.GetMathEngine() );
	transform->SetName( opName + "_transformIndices" );
	transform->SetDimensionRule( BD_Channels, CTransformLayer::O_SetSize, 1 );
	transform->SetDimensionRule( BD_Depth, CTransformLayer::O_Remainder, 1 );
	transform->Connect( 0, *indices.Layer(), indices.OutputIndex() );
	dnn.AddLayer( *transform );
	return CLayerOutput( transform.Ptr(), 0 );
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

void CGatherOperator::ProcessTensors( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	// This is a stub for a specific case: integer 1-dimensional data
	CheckOnnxProtocol( inputs[0] != nullptr, "input can't be optional", *this );
	CheckOnnxProtocol( inputs[1] != nullptr, "input can't be optional", *this );

	if( inputs[0]->IsCalculated() && inputs[1]->IsCalculated() ) {
		const CDataTensor* data = dynamic_cast<const CDataTensor*>( inputs[0].Ptr() );
		NeoAssert( data != nullptr );
		const CDataTensor* indices = dynamic_cast<const CDataTensor*>( inputs[1].Ptr() );
		NeoAssert( indices != nullptr );
		CheckOnnxProtocol( indices->Data()->GetDataType() == CT_Int, "indices must be integer", *this );
		processDataTensors( *data, *indices, outputs );
	} else {
		CLayerOperator::ProcessTensorsImpl( CUserInputMask( 1 ), inputs, dnn, outputs );
	}
}

void CGatherOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	// nullptrs were checked in ProcessTensors
	NeoAssert( inputs[0] != nullptr && inputs[1] != nullptr );

	if( inputs[0]->IsCalculated() ) {
		NeoAssert( !inputs[1]->IsCalculated() );
		addLookupLayer( dynamic_cast<const CDataTensor&>( *inputs[0] ),
			dynamic_cast<const CUserTensor&>( *inputs[1] ), dnn, outputs );
	} else {
		if( inputs[1]->IsCalculated() ) {
			addImageToPixelLayer( dynamic_cast<const CUserTensor&>( *inputs[0] ),
				*AsUserTensor( dynamic_cast<const CDataTensor&>( *inputs[1] ), Name() + "_IndexSource", dnn ),
				dnn, outputs );
		} else {
			addImageToPixelLayer( dynamic_cast<const CUserTensor&>( *inputs[0] ),
				dynamic_cast<const CUserTensor&>( *inputs[1] ), dnn, outputs );
		}
	}
}

// Calculates operator results when both input tensors were calculated during conversion
void CGatherOperator::processDataTensors( const CDataTensor& data, const CDataTensor& indices,
	CTensorArray& outputs ) const
{
	const int axis = axisAttr < 0 ? axisAttr + data.DimCount() : axisAttr;
	CheckOnnxProtocol( axis >= 0 && axis < data.DimCount(), "axis out of range", *this );
	CPtr<const CDnnBlob> dataBlob = prepareDataBlob( data, axis );
	CPtr<const CDnnBlob> indicesBlob = dynamic_cast<const CDataTensor*>( convertToOnnx( indices ).Ptr() )->Data();

	const int resultDimCount = indices.DimCount() + data.DimCount() - 1;
	CTensorLayout resultLayout( resultDimCount );
	CTensorShape resultShape;
	getResultShape( data.Shape(), axis, indices.Shape(), resultShape );

	TBlobType dataType = dataBlob->GetDataType();
	CPtr<CDnnBlob> resultBlob = CDnnBlob::CreateBlob( dataBlob->GetMathEngine(), dataType,
		getResultBlobDesc( resultShape, resultLayout, dataType ) );
	if( dataType == CT_Float ) {
		gather<float>( *this, *dataBlob, *indicesBlob, *resultBlob );
	} else {
		gather<int>( *this, *dataBlob, *indicesBlob, *resultBlob );
	}
	outputs.Add( new CDataTensor( resultShape, resultLayout, *resultBlob ) );
}

void CGatherOperator::addLookupLayer( const CDataTensor& data, const CUserTensor& indices, CDnn& dnn, CTensorArray& outputs ) const
{
	const int axis = axisAttr < 0 ? axisAttr + data.DimCount() : axisAttr;
	CheckOnnxProtocol( axis >= 0 && axis < data.DimCount(), "axis out of range", *this );
	CPtr<const CDnnBlob> dataBlob = prepareDataBlob( data, axis );
	CheckNeoOnnxSupport( dataBlob->GetDataType() == CT_Float, "non-float lookup", *this );

	// Prepare lookup table
	if( dataBlob->GetObjectCount() != dataBlob->GetBatchLength() ) {
		CPtr<CDnnBlob> lookupData = dataBlob->GetCopy();
		CBlobDesc lookupDesc( CT_Float );
		// GetObjectCount == number of objects to gather from
		lookupDesc.SetDimSize( BD_BatchWidth, dataBlob->GetBatchLength() );
		// GetObjectSize == size of each object
		lookupDesc.SetDimSize( BD_Channels, dataBlob->GetDataSize() / dataBlob->GetBatchLength() );
		lookupData->ReinterpretDimensions( lookupDesc );
		dataBlob = lookupData.Ptr();
	}

	// Prepare indices
	CLayerOutput preparedIndices = prepareLookupIndices( indices, Name(), dnn );

	CPtr<CMultichannelLookupLayer> lookup = new CMultichannelLookupLayer( dnn.GetMathEngine() );
	lookup->SetName( Name() );
	lookup->SetDimensions( { CLookupDimension( dataBlob->GetObjectCount(), dataBlob->GetObjectSize() ) } );
	lookup->SetEmbeddings( const_cast<CDnnBlob*>( dataBlob.Ptr() ), 0 );
	lookup->Connect( 0, *preparedIndices.Layer, preparedIndices.OutputIndex );
	dnn.AddLayer( *lookup );

	CTensorShape resultShape;
	getResultShape( data.Shape(), axis, indices.Shape(), resultShape );
	outputs.Add( transformOutput( *lookup, resultShape, dnn ).Ptr() );
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
	currData = dynamic_cast<const CUserTensor*>( ConvertTensor( *currData, requiredLayout ).Ptr() );
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
	CPtr<const CUserTensor> currIndices = dynamic_cast<const CUserTensor*>( convertToOnnx( indices ).Ptr() );
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
	// Unpack them into onnx output tensor
	CTensorShape resultShape;
	getResultShape( data.Shape(), axis, indices.Shape(), resultShape );
	outputs.Add( transformOutput( *imageToPixel, resultShape, dnn ).Ptr() );
}

} // namespace NeoOnnx
