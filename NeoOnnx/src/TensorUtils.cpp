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

// Renames dimensions of data blob (without any reordering in memory)
CPtr<const CDnnBlob> renameDimensions( const CDnnBlob& input, const CTensorShape& shape, const CTensorLayout& outputLayout )
{
	CheckNeoOnnxInternal( shape.Size() == outputLayout.Size(), "shape/layout size mismatch" );
	// We have to copy data here because multiple tensors may be connected to the input tensor
	CBlobDesc outputBlobDesc( input.GetDataType() );
	for( int dimIndex = 0; dimIndex < shape.Size(); ++dimIndex ) {
		outputBlobDesc.SetDimSize( outputLayout[dimIndex], shape[dimIndex] );
	}
	IMathEngine& mathEngine = input.GetMathEngine();
	CPtr<CDnnBlob> result = CDnnBlob::CreateBlob( mathEngine, input.GetDataType(), outputBlobDesc );
	mathEngine.VectorCopy( result->GetData(), input.GetData(), input.GetDataSize() );
	return result.Ptr();
}

// Renames dimensions of layer output (without any reordering in memory)
CLayerOutput renameDimensions( const CLayerOutput& input, const CTensorShape& shape, const CTensorLayout& outputLayout )
{
	CheckNeoOnnxInternal( shape.Size() == outputLayout.Size(), "shape/layout size mismatch" );
	CDnn& dnn = *( input.Layer->GetDnn() );
	CPtr<CTransformLayer> transformLayer = new CTransformLayer( dnn.GetMathEngine() );
	transformLayer->SetName( getUniqueName( dnn, "transform_" ) );
	for( TBlobDim dim = BD_BatchLength; dim < BD_Count; ++dim ) {
		const int dimIndex = outputLayout.Find( dim );
		if( dimIndex == NotFound ) {
			transformLayer->SetDimensionRule( dim, CTransformLayer::CDimensionRule( CTransformLayer::O_SetSize, 1 ) );
		} else {
			transformLayer->SetDimensionRule( dim, CTransformLayer::CDimensionRule( CTransformLayer::O_SetSize, shape[dimIndex] ) );
		}
	}
	dnn.AddLayer( *transformLayer );
	transformLayer->Connect( 0, *input.Layer, input.OutputIndex );
	return CLayerOutput( transformLayer.Ptr(), 0 );
}

// Renames dimensions of tensor (without any reordering in memory)
CPtr<const CTensorBase> renameDimensions( const CTensorBase& input, const CTensorLayout& outputLayout )
{
	if( input.IsCalculated() ) {
		CPtr<const CDnnBlob> blob = renameDimensions( *dynamic_cast<const CDataTensor&>( input ).Data(),
			input.Shape(), outputLayout );
		return new CDataTensor( input.Shape(), outputLayout, *blob );
	} else {
		CLayerOutput layerOutput = renameDimensions( dynamic_cast<const CUserTensor&>( input ).LayerOutput(),
			input.Shape(), outputLayout );
		return new CUserTensor( input.Shape(), outputLayout, layerOutput );
	}
}

// Swaps 2 dimensions of data blob
CPtr<const CDnnBlob> swapDimensions( const CDnnBlob& inputBlob, TBlobDim firstDim, TBlobDim secondDim )
{
	CBlobDesc outputDesc = inputBlob.GetDesc();
	const int firstDimSize = outputDesc.DimSize( firstDim );
	const int secondDimSize = outputDesc.DimSize( secondDim );
	outputDesc.SetDimSize( firstDim, secondDimSize );
	outputDesc.SetDimSize( secondDim, firstDimSize );

	IMathEngine& mathEngine = inputBlob.GetMathEngine();
	CPtr<CDnnBlob> outputBlob = CDnnBlob::CreateBlob( mathEngine, inputBlob.GetDataType(), outputDesc );
	outputBlob->TransposeFrom( &inputBlob, firstDim, secondDim );
	return outputBlob.Ptr();
}

// Swasp 2 dimensions of given layer output
CLayerOutput swapDimensions( const CLayerOutput& input, TBlobDim firstDim, TBlobDim secondDim )
{
	CDnn& dnn = *( input.Layer->GetDnn() );
	CPtr<CTransposeLayer> transposeLayer = new CTransposeLayer( dnn.GetMathEngine() );
	transposeLayer->SetName( getUniqueName( dnn, "transpose_" ) );
	transposeLayer->SetTransposedDimensions( firstDim, secondDim );
	dnn.AddLayer( *transposeLayer );
	transposeLayer->Connect( 0, *input.Layer, input.OutputIndex );
	return CLayerOutput( transposeLayer.Ptr(), 0 );
}

// Swaps 2 dimensions of input tensor
CPtr<const CTensorBase> swapDimensions( const CTensorBase& input, TBlobDim firstDim, TBlobDim secondDim )
{
	CTensorLayout outputLayout = input.Layout();
	const int firstDimIndex = outputLayout.Find( firstDim );
	const int secondDimIndex = outputLayout.Find( secondDim );
	CheckNeoOnnxInternal( firstDimIndex != NotFound && secondDimIndex != NotFound,
		"swap of missing dimension" );
	swap( outputLayout[firstDimIndex], outputLayout[secondDimIndex] );

	if( input.IsCalculated() ) {
		CPtr<const CDnnBlob> blob = swapDimensions( *dynamic_cast<const CDataTensor&>( input ).Data(),
			firstDim, secondDim );
		return new CDataTensor( input.Shape(), outputLayout, *blob );
	} else {
		CLayerOutput layerOutput = swapDimensions( dynamic_cast<const CUserTensor&>( input ).LayerOutput(),
			firstDim, secondDim );
		return new CUserTensor( input.Shape(), outputLayout, layerOutput );
	}
}

CPtr<const CTensorBase> ConvertTensor( const CTensorBase& input, const CTensorLayout& outputLayout )
{
	// Trivial case
	if( input.Layout() == outputLayout ) {
		return &input;
	}

	const int dimCount = outputLayout.Size();
	CheckNeoOnnxInternal( input.DimCount() == dimCount,
		"input's dimension count doesn't math outputLayout's" );

	// Step 1: renaming dimensions (if needed)
	// It's possible that input.Layout() and outputLayout use different dimensions
	// Renaming means assigning outputLayout's dimensions to the ones of input.Layout()
	// without data transposing.
	// e.g.
	//     input.Layout() == { BD_Channels, BD_BatchWidth }
	//     outputLayout == { BD_Height, BD_Width }
	// result of renaming:
	//     renamed.Layout == { BD_Width, BD_Height } (transpose will happen on step #2)
	CPtr<const CTensorBase> currentTensor = &input;
	CTensorLayout sortedInputLayout = input.Layout();
	sortedInputLayout.QuickSort<Ascending<TBlobDim>>();
	CTensorLayout sortedOutputLayout = outputLayout;
	sortedOutputLayout.QuickSort<Ascending<TBlobDim>>();
	if( sortedInputLayout != sortedOutputLayout ) {
		// Tensors use different blob dimensions, need to rename
		const CTensorLayout& inputLayout = input.Layout();
		CTensorLayout renamedLayout;
		renamedLayout.SetBufferSize( dimCount );
		for( int dimIndex = 0; dimIndex < dimCount; ++dimIndex ) {
			const int sortedDimIndex = sortedInputLayout.Find( inputLayout[dimIndex] );
			renamedLayout.Add( sortedOutputLayout[sortedDimIndex] );
		}
		currentTensor = renameDimensions( *currentTensor, renamedLayout );
	}

	// Step 2: reordering dimensions
	// NeoML has operations only for swapping 2 dimensions
	// Step 1 guarantees that outputLayout is a permutation of currentTensor.Layout()
	for( int dimIndex = 0; dimIndex < dimCount; ++dimIndex ) {
		TBlobDim inputDim = currentTensor->Layout()[dimIndex];
		TBlobDim outputDim = outputLayout[dimIndex];
		if( inputDim != outputDim ) {
			currentTensor = swapDimensions( *currentTensor, inputDim, outputDim );
		}
	}

	return currentTensor;
}

// --------------------------------------------------------------------------------------------------------------------

CPtr<const CTensorBase> RemoveTensorDims( const CTensorBase& input, const CFastArray<int, 8>& _dims )
{
	CFastArray<int, 8> dims;
	_dims.CopyTo( dims );
	dims.QuickSort<Ascending<int>>();

	CTensorShape outputShape;
	input.Shape().CopyTo( outputShape );
	for( int i = dims.Size() - 1; i >= 0; --i ) {
		outputShape.DeleteAt( dims[i] );
	}

	CTensorLayout outputLayout = input.Layout();
	for( int i = dims.Size() - 1; i >= 0; --i ) {
		outputLayout.DeleteAt( dims[i] );
	}

	if( input.IsCalculated() ) {
		return new CDataTensor( outputShape, outputLayout,
			*( dynamic_cast<const CDataTensor&>( input ).Data() ) );
	} else {
		return new CUserTensor( outputShape, outputLayout,
			dynamic_cast<const CUserTensor&>( input ).LayerOutput() );
	}
	// To satisfy compilers' warnings
	return nullptr;
}

} // namespace NeoOnnx
