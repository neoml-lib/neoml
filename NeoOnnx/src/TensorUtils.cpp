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
			CheckNeoOnnxSupport( false, "tensor type" );
	}
	return CT_Invalid;
}

//---------------------------------------------------------------------------------------------------------------------

// Gets layer name with given prefix which isn't used in dnn
static CString getUniqueLayerName( const CDnn& dnn, const CString& prefix )
{
	int currIndex = dnn.GetLayerCount();
	CString currName = prefix + Str( currIndex );
	while( dnn.HasLayer( currName ) ) {
		++currIndex;
		currName = prefix + Str( currIndex );
	}
	return currName;
}

// Converts tensor in a way that layout[heightDimIndex] is BD_Height and layout[widthDimIndex] is BD_Width.
// If widthDimIndex is NotFound then only height dimension is moved to layout[heightDimIndex]
CPtr<const CUserTensor> convertTensorToHw( const CUserTensor& input, int heightDimIndex, int widthDimIndex )
{
	const CTensorLayout& inputLayout = input.Layout();

	if( inputLayout[heightDimIndex] == BD_Height
		&& ( widthDimIndex == NotFound || inputLayout[widthDimIndex] == static_cast<int>( BD_Width ) ) )
	{
		return &input;
	}

	CTensorLayout newLayout;
	newLayout.SetBufferSize( input.DimCount() );
	for( int i = 0; i < input.DimCount(); ++i ) {
		if( i == heightDimIndex ) {
			newLayout.Add( BD_Height );
		} else if( i == widthDimIndex ) {
			newLayout.Add( BD_Width );
		} else if( widthDimIndex == NotFound ) {
			newLayout.Add( i < static_cast<int>( BD_Width ) ? static_cast<TBlobDim>( i )
				: static_cast<TBlobDim>( i + 1 ) );
		} else {
			newLayout.Add( i < static_cast<int>( BD_Width ) ? static_cast<TBlobDim>( i )
				: static_cast<TBlobDim>( i + 2 ) );
		}
	}

	return dynamic_cast<const CUserTensor*>( ConvertTensor( input, newLayout ).Ptr() );
}

//---------------------------------------------------------------------------------------------------------------------

// Renames dimensions of data blob (without any reordering in memory)
CPtr<const CDnnBlob> renameDimensions( const CDnnBlob& input, const CTensorShape& shape, const CTensorLayout& outputLayout )
{
	NeoAssert( shape.Size() == outputLayout.Size() );
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
	NeoAssert( shape.Size() == outputLayout.Size() );
	CDnn& dnn = *( input.Layer->GetDnn() );
	CPtr<CTransformLayer> transformLayer = new CTransformLayer( dnn.GetMathEngine() );
	transformLayer->SetName( getUniqueLayerName( dnn, "transform_" ) );
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
	transposeLayer->SetName( getUniqueLayerName( dnn, "transpose_" ) );
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
	NeoAssert( firstDimIndex != NotFound && secondDimIndex != NotFound );
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
	NeoAssert( input.DimCount() == dimCount );

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

// --------------------------------------------------------------------------------------------------------------------

void CalculatePadding( const CString& autoPad, const CTensorShape& kernelShape, CFastArray<int, 8>& pads )
{
	const int padDims = static_cast<int>( kernelShape.Size() );
	for( int padDimIndex = 0; padDimIndex < padDims; ++padDimIndex ) {
		const int totalPadSize = kernelShape[padDimIndex] - 1;
		if( autoPad == "SAME_LOWER" ) {
			pads[padDimIndex] = ( totalPadSize + 1 ) / 2;
		} else {
			pads[padDimIndex] = totalPadSize / 2;
		}
		pads[padDims + padDimIndex] = totalPadSize - pads[padDimIndex];
	}
}

// --------------------------------------------------------------------------------------------------------------------

CPtr<const CUserTensor> addImageResizeLayer( CImageResizeLayer& imageResize, CDnn& dnn, const CUserTensor& input,
	int heightDimIndex, int widthDimIndex )
{
	// Add imageResize layer
	CPtr<const CUserTensor> result = convertTensorToHw( input, heightDimIndex, widthDimIndex );
	imageResize.Connect( 0, *result->Layer(), result->OutputIndex() );
	dnn.AddLayer( imageResize );

	// Calculate output shape
	CTensorShape outputShape;
	result->Shape().CopyTo( outputShape );
	outputShape[heightDimIndex] += imageResize.GetDelta( CImageResizeLayer::IS_Top )
		+ imageResize.GetDelta( CImageResizeLayer::IS_Bottom );
	if( widthDimIndex != NotFound ) {
		outputShape[widthDimIndex] += imageResize.GetDelta( CImageResizeLayer::IS_Left )
			+ imageResize.GetDelta( CImageResizeLayer::IS_Right );
	}

	// Construct new CUserTensor which is provided by imageResize layer
	return new CUserTensor( outputShape, result->Layout(), CLayerOutput( &imageResize, 0 ) );
}

CPtr<const CUserTensor> PadUserTensor( const CUserTensor& input, const CFastArray<int, 8>& pads, float padValue )
{
	// Pool and conv operators storing pads only for N-2 tensor dimensions (leaving out batch and channels)
	// On the other side Pad operator is storing pads for every tensor dimension

	// Number of padded dimensions
	const int paddedDims = pads.Size() / 2;
	// Index of first padded dimension
	const int padDimIndex = input.DimCount() - paddedDims;
	// Prefix for padding layer names
	const CString padNamePrefix = input.Layer()->GetName() + CString( "_pad_" );
	// Used network
	CDnn& dnn = *( input.Layer()->GetDnn() );
	// Used mathEngine
	IMathEngine& mathEngine = dnn.GetMathEngine();

	CPtr<const CUserTensor> currData = &input;
	CPtr<CImageResizeLayer> imageResize = nullptr;
	int heightDimIndex = NotFound;
	int widthDimIndex = NotFound;

	for( int i = 0; i < paddedDims; ++i ) {
		if( pads[i] == 0 && pads[i + paddedDims] == 0 ) {
			continue;
		}

		if( imageResize == nullptr ) {
			imageResize = new CImageResizeLayer( mathEngine );
			imageResize->SetName( getUniqueLayerName( dnn, padNamePrefix ) );
			imageResize->SetDefaultValue( padValue );
		}

		if( heightDimIndex == NotFound ) {
			heightDimIndex = padDimIndex + i;
			imageResize->SetDelta( CImageResizeLayer::IS_Top, pads[i] );
			imageResize->SetDelta( CImageResizeLayer::IS_Bottom, pads[paddedDims + i] );
		} else {
			widthDimIndex = padDimIndex + i;
			imageResize->SetDelta( CImageResizeLayer::IS_Left, pads[i] );
			imageResize->SetDelta( CImageResizeLayer::IS_Right, pads[paddedDims + i] );
			currData = addImageResizeLayer( *imageResize, dnn, *currData, heightDimIndex, widthDimIndex );
			imageResize = nullptr;
			heightDimIndex = NotFound;
			widthDimIndex = NotFound;
		}
	}

	// Corner case: we need to expand odd number of dimensions
	// In that case by this moment imageResize != nullptr
	// heightDimIndex will be defined but widthDimIndex will remain NotFound
	if( imageResize != nullptr ) {
		currData = addImageResizeLayer( *imageResize, dnn, *currData, heightDimIndex, widthDimIndex );
	}

	return currData;
}

//---------------------------------------------------------------------------------------------------------------------

// Returns true if shapes are equal
bool areShapesEqual( const CTensorShape& first, const CTensorShape& second )
{
	if( first.Size() != second.Size() ) {
		return false;
	}

	for( int i = 0; i < first.Size(); ++i ) {
		if( first[i] != second[i] ) {
			return false;
		}
	}

	return true;
}

bool BroadcastTensorShape( const CTensorShape& first, const CTensorShape& second, const CBroadcast& broadcast, CTensorShape& result )
{
	if( broadcast.Type == BT_None ) {
		// No broadcast, the shape must match
		if( areShapesEqual( first, second ) ) {
			first.CopyTo( result );
			return true;
		}
		return false;
	}

	int axis = NotFound;
	if( broadcast.Type == BT_Onnx ) {
		axis = broadcast.Axis;
		CheckNeoOnnxSupport( second.Size() <= first.Size(), "second tensor has more dimensions" );
		if( axis < 0 ) {
			axis = abs( first.Size() - second.Size() );
		}
	} else {
		// Numpy-style broadcast is similar to the Onnx-broadcast with axis equal to difference
		// in number of dimensions
		axis = abs( first.Size() - second.Size() );
	}

	// The shape with lesser number of dimensions must be padded with ones
	const CTensorShape& lesserShape = first.Size() <= second.Size() ? first : second;
	const CTensorShape& biggerShape = first.Size() > second.Size() ? first  : second;
	CTensorShape paddedShape;
	paddedShape.Add( 1, axis );
	paddedShape.Add( lesserShape );
	if( paddedShape.Size() > biggerShape.Size() ) {
		// Wrong braodcast parameters (axis value is too big)
		return false;
	}
	NeoAssert( broadcast.Type == BT_Onnx || paddedShape.Size() == biggerShape.Size() );

	// This will add ones only in case of BT_Onnx and axis != abs( first.Size() - second.Size() )
	paddedShape.Add( 1, biggerShape.Size() - paddedShape.Size() );

	result.SetSize( paddedShape.Size() );
	for( int dim = 0; dim < result.Size(); ++dim ) {
		if( paddedShape[dim] == biggerShape[dim] || min( paddedShape[dim], biggerShape[dim] ) == 1 ) {
			result[dim] = max( paddedShape[dim], biggerShape[dim] );
		} else {
			result.DeleteAll();
			return false;
		}
	}

	return true;
}

static CPtr<const CUserTensor> addUpsample2dLayer( CUpsampling2DLayer& upsample, CDnn& dnn, const CUserTensor& input,
	int heightDimIndex, int widthDimIndex )
{
	// Add imageResize layer
	CPtr<const CUserTensor> result = convertTensorToHw( input, heightDimIndex, widthDimIndex );
	upsample.Connect( 0, *result->Layer(), result->OutputIndex() );
	dnn.AddLayer( upsample );

	// Calculate output shape
	CTensorShape outputShape;
	result->Shape().CopyTo( outputShape );

	NeoAssert( outputShape[heightDimIndex] == 1 );
	outputShape[heightDimIndex] = upsample.GetHeightCopyCount();
	if( widthDimIndex != NotFound ) {
		NeoAssert( outputShape[widthDimIndex] == 1 );
		outputShape[widthDimIndex] = upsample.GetWidthCopyCount();
	}

	// Construct new CUserTensor which is provided by imageResize layer
	return new CUserTensor( outputShape, result->Layout(), CLayerOutput( &upsample, 0 ) );
}

// Pads shape with '1' without changing the data
static CPtr<const CUserTensor> padTensorShape( const CUserTensor& input, int dimCount, int axis )
{
	const CTensorShape& inputShape = input.Shape();
	NeoAssert( axis + inputShape.Size() <= dimCount );

	CTensorShape outputShape;
	outputShape.Add( 1, axis );
	outputShape.Add( inputShape );
	outputShape.Add( 1, dimCount - outputShape.Size() );

	const CTensorLayout& inputLayout = input.Layout();

	TBlobDim currDim = BD_BatchLength;
	CTensorLayout outputLayout;
	outputLayout.SetBufferSize( dimCount );
	// Adding unused blob dims to the new layout
	for( int i = 0; i < axis; ++i ) {
		while( inputLayout.Find( currDim ) != NotFound && currDim < BD_Count ) {
			++currDim;
		}
		NeoAssert( currDim != BD_Count );
		outputLayout.Add( currDim );
		++currDim;
	}
	// Copying existing dims
	outputLayout.Add( inputLayout );
	// Adding unused blob dims to the new layout
	for( int i = outputLayout.Size(); i < dimCount; ++i ) {
		while( inputLayout.Find( currDim ) != NotFound && currDim < BD_Count ) {
			++currDim;
		}
		NeoAssert( currDim != BD_Count );
		outputLayout.Add( currDim );
		++currDim;
	}

	return new CUserTensor( outputShape, outputLayout, input.LayerOutput() );
}

// Broadcasts user tensor into outputShape via broadcastInfo
static CPtr<const CUserTensor> broadcastUserTensor( const CUserTensor& input, const CBroadcast& broadcast,
	const CTensorShape& outputShape )
{
	if( areShapesEqual( input.Shape(), outputShape ) ) {
		return &input;
	}

	NeoAssert( broadcast.Type != BT_None );
	NeoAssert( input.DimCount() <= outputShape.Size() );

	// Prefix for upsmaple layer names
	const CString upsampleNamePrefix = input.Layer()->GetName() + CString( "_upsample_" );
	// Used network
	CDnn& dnn = *( input.Layer()->GetDnn() );
	// Used mathEngine
	IMathEngine& mathEngine = dnn.GetMathEngine();

	int axis = outputShape.Size() - input.DimCount();
	if( broadcast.Type == BT_Onnx && broadcast.Axis >= 0 ) {
		axis = broadcast.Axis;
	}

	CPtr<const CUserTensor> currData = padTensorShape( input, outputShape.Size(), axis );
	CPtr<CUpsampling2DLayer> upsample = nullptr;
	int heightDimIndex = NotFound;
	int widthDimIndex = NotFound;
	CTensorShape inputShape;
	currData->Shape().CopyTo( inputShape );

	for( int i = 0; i < inputShape.Size(); ++i ) {
		if( inputShape[i] == outputShape[i] ) {
			continue;
		}
		NeoAssert( inputShape[i] == 1 );

		if( upsample == nullptr ) {
			upsample = new CUpsampling2DLayer( mathEngine );
			upsample->SetName( getUniqueLayerName( dnn, upsampleNamePrefix ) );
		}

		if( heightDimIndex == NotFound ) {
			heightDimIndex = i;
			upsample->SetHeightCopyCount( outputShape[i] );
		} else {
			widthDimIndex = i;
			upsample->SetWidthCopyCount( outputShape[i] );
			currData = addUpsample2dLayer( *upsample, dnn, *currData, heightDimIndex, widthDimIndex );
			upsample = nullptr;
			heightDimIndex = NotFound;
			widthDimIndex = NotFound;
		}
	}

	// Corner case: we need to broadcast odd number of dimensions
	// In that case by this moment upsample != nullptr
	// heightDimIndex will be defined but widthDimIndex will remain NotFound
	if( upsample != nullptr ) {
		upsample->SetWidthCopyCount( 1 ); // Default value is 0 which is invalid
		currData = addUpsample2dLayer( *upsample, dnn, *currData, heightDimIndex, widthDimIndex );
	}

	return currData;
}

// Broadcasts data tensor into outputShape via broadcastInfo
CPtr<const CDataTensor> broadcastDataTensor( const CDataTensor& input, const CBroadcast& broadcast,
	const CTensorShape& outputShape )
{
	if( areShapesEqual( input.Shape(), outputShape ) ) {
		return &input;
	}

	NeoAssert( broadcast.Type != BT_None );

	IMathEngine& mathEngine = input.Data()->GetMathEngine();
	CRandom random( 0x32456 );

	CDnn internalDnn( random, mathEngine );
	CPtr<CSourceLayer> source = new CSourceLayer( mathEngine );
	source->SetBlob( input.Data()->GetCopy() );
	internalDnn.AddLayer( *source );

	CPtr<const CUserTensor> internalInput = new CUserTensor( input.Shape(), input.Layout(), CLayerOutput( source, 0 ) );
	CPtr<const CUserTensor> internalOutput = broadcastUserTensor( *internalInput, broadcast, outputShape );
	NeoPresume( areShapesEqual( internalOutput->Shape(), outputShape ) );

	CPtr<CSinkLayer> sink = new CSinkLayer( mathEngine );
	sink->Connect( 0, *internalOutput->Layer(), internalOutput->OutputIndex() );
	internalDnn.AddLayer( *sink );

	internalDnn.RunOnce();

	return new CDataTensor( outputShape, internalOutput->Layout(), *sink->GetBlob() );
}

CPtr<const CTensorBase> BroadcastTensor( const CTensorBase& input, const CBroadcast& broadcast,
	const CTensorShape& outputShape )
{
	if( input.IsCalculated() ) {
		return broadcastDataTensor( dynamic_cast<const CDataTensor&>( input ), broadcast, outputShape ).Ptr();
	}
	return broadcastUserTensor( dynamic_cast<const CUserTensor&>( input ), broadcast, outputShape ).Ptr();
}

} // namespace NeoOnnx
