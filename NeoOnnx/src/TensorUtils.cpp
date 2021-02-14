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

#include "common.h"
#pragma hdrstop

#include "NeoOnnxCheck.h"
#include "TensorUtils.h"

#include "onnx.pb.h"

namespace NeoOnnx {

// Gets NeoML blob type from onnx tensor's data type
TBlobType GetBlobType( const onnx::TensorProto_DataType& onnxDataType )
{
	switch( onnxDataType ) {
		case onnx::TensorProto::FLOAT:
		case onnx::TensorProto::DOUBLE:
			return CT_Float;
		case onnx::TensorProto::BOOL:
		case onnx::TensorProto::INT8:
		case onnx::TensorProto::UINT8:
		case onnx::TensorProto::INT16:
		case onnx::TensorProto::UINT16:
		case onnx::TensorProto::INT32:
		case onnx::TensorProto::UINT32:
		// Sometimes Constant operator's value is stored in int64 (even if it can be stored in 32-bit integer)
		// That's why we allow here to use int64 and will cast it later to 32-bit
		case onnx::TensorProto::INT64:
		case onnx::TensorProto::UINT64:
			return CT_Int;
		case onnx::TensorProto::FLOAT16:
		case onnx::TensorProto::BFLOAT16:
		case onnx::TensorProto::COMPLEX64:
		case onnx::TensorProto::COMPLEX128:
		case onnx::TensorProto::UNDEFINED:
		default:
			CheckNeoOnnxInternal( false, "tensor type is not supported" );
	}
	return CT_Invalid;
}

//---------------------------------------------------------------------------------------------------------------------

static CString getUniqueName( const CDnn& dnn, const CString& prefix )
{
	int currIndex = dnn.GetLayerCount();
	CString currName = prefix + Str( currIndex );
	while( dnn.HasLayer( currName ) ) {
		++currIndex;
		currName = prefix + Str( currIndex );
	}
	return currName;
}

static bool isEqual( const CDimOrder& first, const CDimOrder& second )
{
	CheckNeoOnnxInternal( first.Size() == second.Size(), "Layouts have different dimension count" );

	for( int dimIndex = 0; dimIndex < first.Size(); ++dimIndex ) {
		if( first[dimIndex] != second[dimIndex] ) {
			return false;
		}
	}

	return true;
}

static void getDimOrder( const CTensorShape& shape, const CTensorLayout& layout, CDimOrder& order )
{
	if( layout.DimType == DT_NeoML ) {
		layout.OnnxOrder.CopyTo( order );
	} else {
		order.SetSize( shape.Size() );
		for( int dim = 0; dim < order.Size(); ++dim ) {
			order[dim] = static_cast<TBlobDim>( dim );
		}
	}
	CheckNeoOnnxInternal( order.Size() == shape.Size(), "Dimension number mismatch" );
}

// Reinterprets dimensions of data (without reordering)
// Specification for CDataTensor
template<bool IsDataTensor=true>
CPtr<const CTensorBase> reinterpretDimensions( const CTensorBase& input, const CTensorLayout& outputLayout )
{
	CheckNeoOnnxInternal( input.IsCalculated(), "Can't reinterpret dimensions of CUserTensor" );

	CDimOrder inputOrder;
	getDimOrder( input.Shape(), input.Layout(), inputOrder );

	CDimOrder outputOrder;
	getDimOrder( input.Shape(), outputLayout, outputOrder );

	if( isEqual( inputOrder, outputOrder ) ) {
		return &input;
	}

	const CDataTensor& inputDataTensor = dynamic_cast<const CDataTensor&>( input );
	const CDnnBlob* inputBlob = inputDataTensor.Data();

	const CBlobDesc& inputDesc = inputBlob->GetDesc();
	CBlobDesc outputDesc;
	outputDesc.SetDataType( inputBlob->GetDataType() );
	for( int dimIndex = 0; dimIndex < inputOrder.Size(); ++dimIndex ) {
		outputDesc.SetDimSize( outputOrder[dimIndex], inputDesc.DimSize( inputOrder[dimIndex] ) );
	}

	IMathEngine& mathEngine = inputBlob->GetMathEngine();
	CPtr<CDnnBlob> outputBlob = CDnnBlob::CreateBlob( mathEngine, inputBlob->GetDataType(), outputDesc );
	if( inputBlob->GetDataType() == CT_Float ) {
		mathEngine.VectorCopy( outputBlob->GetData(), inputBlob->GetData(), inputBlob->GetDataSize() );
	} else {
		CheckNeoOnnxInternal( inputBlob->GetDataType() == CT_Int, "Unknown blob data type" );
		mathEngine.VectorCopy( outputBlob->GetData<int>(), inputBlob->GetData<int>(), inputBlob->GetDataSize() );
	}

	return new CDataTensor( input.Shape(), outputLayout, *outputBlob );
}

// Specification for CUserTensor
template<>
CPtr<const CTensorBase> reinterpretDimensions<false>( const CTensorBase& input, const CTensorLayout& outputLayout )
{
	CheckNeoOnnxInternal( !input.IsCalculated(), "Can't reinterpret dimensions of CDataTensor" );
	const CTensorShape& inputShape = input.Shape();

	CDimOrder inputOrder;
	getDimOrder( inputShape, input.Layout(), inputOrder );

	CDimOrder outputOrder;
	getDimOrder( inputShape, outputLayout, outputOrder );

	if( isEqual( inputOrder, outputOrder ) ) {
		return &input;
	}

	const CUserTensor& inputUserTensor = dynamic_cast<const CUserTensor&>( input );
	CLayerOutput currLayerOutput = inputUserTensor.LayerOutput();
	CheckNeoOnnxInternal( currLayerOutput.Layer != nullptr && currLayerOutput.OutputIndex != NotFound,
		"Uninitialized layer output" );
	CheckNeoOnnxInternal( currLayerOutput.Layer->GetDnn() != nullptr, "Layer doesn't belong to net" );

	CDnn& dnn = *( currLayerOutput.Layer->GetDnn() );
	IMathEngine& mathEngine = dnn.GetMathEngine();

	CPtr<CTransformLayer> transformLayer = new CTransformLayer( mathEngine );
	transformLayer->SetName( getUniqueName( dnn, "transform_" ) );
	for( TBlobDim dim = BD_BatchLength; dim < BD_Count; ++dim ) {
		const int dimIndex = outputOrder.Find( dim );
		if( dimIndex == NotFound ) {
			transformLayer->SetDimensionRule( dim, CTransformLayer::CDimensionRule( CTransformLayer::O_SetSize, 1 ) );
		} else {
			transformLayer->SetDimensionRule( dim, CTransformLayer::CDimensionRule( CTransformLayer::O_SetSize, inputShape[dimIndex] ) );
		}
	}

	dnn.AddLayer( *transformLayer );
	transformLayer->Connect( 0, *currLayerOutput.Layer, currLayerOutput.OutputIndex );
	currLayerOutput.Layer = transformLayer;
	currLayerOutput.OutputIndex = 0;

	return new CUserTensor( inputShape, outputLayout, currLayerOutput );
}

// Swaps 2 dimensions of given tensor
// Tensor must be of DT_NeoML dim type
// Specification for CDataTensor
template<bool IsDataTensor=true>
CPtr<const CTensorBase> swapDimensions( const CTensorBase& input, TBlobDim firstDim, int firstDimIndex,
	TBlobDim secondDim, int secondDimIndex )
{
	CheckNeoOnnxInternal( input.Layout().DimType == DT_NeoML, "Swapping dimensions is possible only from NeoML dim type" );
	CheckNeoOnnxInternal( input.IsCalculated(), "Can't swap dimensions of CUserTensor" );

	CPtr<const CDnnBlob> outputBlob = dynamic_cast<const CDataTensor&>( input ).Data()->GetTransposed( firstDim, secondDim );
	
	CDimOrder outputOrder;
	input.Layout().OnnxOrder.CopyTo( outputOrder );
	swap( outputOrder[firstDimIndex], outputOrder[secondDimIndex] );

	return new CDataTensor( input.Shape(), CTensorLayout( outputOrder ), *outputBlob );
}

// Specification for CDataTensor
template<>
CPtr<const CTensorBase> swapDimensions<false>( const CTensorBase& input, TBlobDim firstDim, int firstDimIndex,
	TBlobDim secondDim, int secondDimIndex )
{
	CheckNeoOnnxInternal( input.Layout().DimType == DT_NeoML, "Swapping dimensions is possible only from NeoML dim type" );
	CheckNeoOnnxInternal( !input.IsCalculated(), "Can't swap dimensions of CDataTensor" );

	CLayerOutput currLayerOutput = dynamic_cast<const CUserTensor&>( input ).LayerOutput();
	CheckNeoOnnxInternal( currLayerOutput.Layer != nullptr && currLayerOutput.OutputIndex != NotFound,
		"Uninitialized layer output" );
	CheckNeoOnnxInternal( currLayerOutput.Layer->GetDnn() != nullptr, "Layer doesn't belong to net" );

	CDnn& dnn = *( currLayerOutput.Layer->GetDnn() );
	IMathEngine& mathEngine = dnn.GetMathEngine();

	CPtr<CTransposeLayer> transposeLayer = new CTransposeLayer( mathEngine );
	transposeLayer->SetName( getUniqueName( dnn, "transpose_" ) );
	transposeLayer->SetTransposedDimensions( firstDim, secondDim );
	transposeLayer->Connect( 0, *currLayerOutput.Layer, currLayerOutput.OutputIndex );
	dnn.AddLayer( *transposeLayer );
	currLayerOutput.Layer = transposeLayer;
	currLayerOutput.OutputIndex = 0;

	CDimOrder outputOrder;
	input.Layout().OnnxOrder.CopyTo( outputOrder );
	swap( outputOrder[firstDimIndex], outputOrder[secondDimIndex] );

	return new CUserTensor( input.Shape(), CTensorLayout( outputOrder ), currLayerOutput );
}

// Converts input blob from NeoML dim order to Onnx
template<bool IsDataTensor>
CPtr<const CTensorBase> convertFromNeoMLToOnnx( const CTensorBase& input )
{
	CheckNeoOnnxInternal( input.Layout().DimType == DT_NeoML, "input tensor must have DT_NeoML dim type" );

	CDimOrder neoMLOrder;
	getDimOrder( input.Shape(), input.Layout(), neoMLOrder );
	neoMLOrder.QuickSort<Ascending<TBlobDim>>();

	// Step 1: reorder the dimensions (if needed)
	CPtr<const CTensorBase> currTensor = &input;

	CDimOrder outputOrder;
	input.Layout().OnnxOrder.CopyTo( outputOrder );

	for( int dimIndex = 0; dimIndex < outputOrder.Size(); ++dimIndex ) {
		if( currTensor->Layout().OnnxOrder[dimIndex] != outputOrder[dimIndex] ) {
			const int swapDimIndex = currTensor->Layout().OnnxOrder.Find( outputOrder[dimIndex], dimIndex );
			currTensor = swapDimensions<IsDataTensor>( *currTensor, neoMLOrder[dimIndex], dimIndex,
				neoMLOrder[swapDimIndex], swapDimIndex );
			// TODO: Delete after debug
			CheckNeoOnnxInternal( currTensor->Layout().OnnxOrder[dimIndex] == outputOrder[dimIndex], 
				"Something wrong..." );
		}
	}
	
	// Step 2: rename the dimensions into first N CDnnBlob dims
	return reinterpretDimensions<IsDataTensor>( *currTensor, CTensorLayout() );
}

// Converts input blob from Onnx dim order to Onnx
template<bool IsDataTensor>
CPtr<const CTensorBase> convertFromOnnxToNeoML( const CTensorBase& input, const CDimOrder& outputOnnxOrder )
{
	CheckNeoOnnxInternal( input.Layout().DimType == DT_Onnx, "input tensor must have DT_Onnx dim type" );

	CDimOrder neoMLOrder;
	outputOnnxOrder.CopyTo( neoMLOrder );
	neoMLOrder.QuickSort<Ascending<TBlobDim>>();

	// Step 1: rename the dimensions (into DT_NeoML)
	CPtr<const CTensorBase> currTensor = reinterpretDimensions<IsDataTensor>( input, CTensorLayout( neoMLOrder ) );

	// Step 2: reorder the dimensions
	for( int dimIndex = 0; dimIndex < neoMLOrder.Size(); ++dimIndex ) {
		if( currTensor->Layout().OnnxOrder[dimIndex] != outputOnnxOrder[dimIndex] ) {
			const int swapDimIndex = currTensor->Layout().OnnxOrder.Find( outputOnnxOrder[dimIndex] );
			currTensor = swapDimensions<IsDataTensor>( *currTensor, neoMLOrder[dimIndex], dimIndex,
				neoMLOrder[swapDimIndex], swapDimIndex );
			// TODO: Delete after debug
			CheckNeoOnnxInternal( currTensor->Layout().OnnxOrder[dimIndex] == outputOnnxOrder[dimIndex], 
				"Something wrong..." );
		}
	}

	return currTensor;
}

// Converts to different layout of the same dim type
template<bool IsDataTensor>
CPtr<const CTensorBase> convertWithinSameDimType( const CTensorBase& input, const CTensorLayout& outputLayout )
{
	const CTensorLayout& inputLayout = input.Layout();
	CheckNeoOnnxInternal( inputLayout.DimType == outputLayout.DimType, "Layouts have different dim type" );
	static_assert( DT_Count == 2, "DT_Count != 2" );

	if( inputLayout.DimType == DT_Onnx ) {
		// No changes in data needed
		return &input;
	}

	// Since this moment, both layouts definitely have DT_NeoML dim type

	if( isEqual( inputLayout.OnnxOrder, inputLayout.OnnxOrder ) ) {
		// No changes in data needed
		return &input;
	}

	// TODO: write more effective conversion inside of NeoML format
	CPtr<const CTensorBase> onnxTensor = convertFromNeoMLToOnnx<IsDataTensor>( input );
	return convertFromOnnxToNeoML<IsDataTensor>( *onnxTensor, outputLayout.OnnxOrder );
}

// Converts tensor to given layout
template<bool IsDataTensor>
CPtr<const CTensorBase> convert( const CTensorBase& inputTensor, const CTensorLayout& outputLayout )
{
	const CTensorLayout& inputLayout = inputTensor.Layout();
	if( inputLayout.DimType == outputLayout.DimType ) {
		return convertWithinSameDimType<IsDataTensor>( inputTensor, outputLayout );
	}

	if( inputLayout.DimType == DT_NeoML ) {
		return convertFromNeoMLToOnnx<IsDataTensor>( inputTensor );
	} else {
		return convertFromOnnxToNeoML<IsDataTensor>( inputTensor, outputLayout.OnnxOrder );
	}
}

CPtr<const CTensorBase> ConvertTensor( const CTensorBase& inputTensor, const CTensorLayout& outputLayout )
{
	const CTensorLayout& inputLayout = inputTensor.Layout();

	if( inputTensor.IsCalculated() ) {
		return convert<true>( inputTensor, outputLayout );
	} else {
		return convert<false>( inputTensor, outputLayout );
	}
	// To satisfy compilers' warnings
	return nullptr;
}

} // namespace NeoOnnx
