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

#include <cfloat>
#include <cmath>

#include "NeoOnnxCheck.h"
#include "TensorUtils.h"
#include <NeoML/Dnn/Layers/Onnx/SourceReshaper.h>

namespace NeoOnnx {

bool IsInteger( float x )
{
	return std::fabs( std::roundf( x ) - x ) < FLT_EPSILON;
}

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

// Gets layer name with the given prefix which isn't used in dnn
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
static CPtr<const CUserTensor> convertTensorToHw( const CUserTensor& input, int heightDimIndex, int widthDimIndex )
{
	const CTensorLayout& inputLayout = input.Layout();

	if( inputLayout[heightDimIndex] == BD_Height
		&& ( widthDimIndex == NotFound || inputLayout[widthDimIndex] == static_cast<int>( BD_Width ) ) )
	{
		return &input;
	}

	CTensorLayout newLayout;
	newLayout.SetBufferSize( input.DimCount() );
	TBlobDim unusedAxis = BD_BatchLength;
	for( int i = 0; i < input.DimCount(); ++i ) {
		if( i == heightDimIndex ) {
			newLayout.Add( BD_Height );
		} else if( i == widthDimIndex ) {
			newLayout.Add( BD_Width );
		} else if( widthDimIndex == NotFound ) {
			newLayout.Add( unusedAxis < BD_Height ? unusedAxis : unusedAxis + 1 );
			++unusedAxis;
		} else {
			newLayout.Add( unusedAxis < BD_Height ? unusedAxis : unusedAxis + 2 );
			++unusedAxis;
		}
	}

	NeoAssert( newLayout[heightDimIndex] == BD_Height );
	NeoAssert( widthDimIndex == -1 || newLayout[widthDimIndex] == BD_Width );
	return ConvertTensor( input, newLayout );
}

//---------------------------------------------------------------------------------------------------------------------

// Renames dimensions of data blob (without any reordering in memory)
static CPtr<const CDnnBlob> renameDimensions( const CDnnBlob& input, const CTensorLayout& inputLayout, const CTensorLayout& outputLayout )
{
	NeoAssert( inputLayout.Size() == outputLayout.Size() );
	// We have to copy data here because multiple tensors may be connected to the input tensor
	CBlobDesc outputBlobDesc( input.GetDataType() );
	for( int dimIndex = 0; dimIndex < inputLayout.Size(); ++dimIndex ) {
		outputBlobDesc.SetDimSize( outputLayout[dimIndex], input.DimSize( inputLayout[dimIndex] ) );
	}
	IMathEngine& mathEngine = input.GetMathEngine();
	CPtr<CDnnBlob> result = CDnnBlob::CreateBlob( mathEngine, input.GetDataType(), outputBlobDesc );
	if( result->GetDataType() == CT_Float ) {
		mathEngine.VectorCopy( result->GetData(), input.GetData(), input.GetDataSize() );
	} else {
		mathEngine.VectorCopy( result->GetData<int>(), input.GetData<int>(), input.GetDataSize() );
	}
	return result.Ptr();
}

// Renames dimensions of layer output (without any reordering in memory)
static CLayerOutput renameDimensions( const CLayerOutput& input, const CTensorLayout& inputLayout, const CTensorLayout& outputLayout )
{
	NeoAssert( inputLayout.Size() == outputLayout.Size() );
	CDnn& dnn = *( input.Layer->GetDnn() );
	CPtr<CTransformLayer> transformLayer = new CTransformLayer( dnn.GetMathEngine() );
	transformLayer->SetName( getUniqueLayerName( dnn, "transform_" ) );
	for( TBlobDim dim = BD_BatchLength; dim < BD_Count; ++dim ) {
		const int dimIndex = outputLayout.Find( dim );
		if( dimIndex == NotFound ) {
			transformLayer->SetDimensionRule( dim, CTransformLayer::CDimensionRule( CTransformLayer::O_SetSize, 1 ) );
		} else {
			transformLayer->SetDimensionRule( dim, CTransformLayer::CDimensionRule( CTransformLayer::O_InputDim, inputLayout[dimIndex] ) );
		}
	}
	dnn.AddLayer( *transformLayer );
	transformLayer->Connect( 0, *input.Layer, input.OutputIndex );
	return CLayerOutput( transformLayer.Ptr(), 0 );
}

// Renames dimensions of tensor (without any reordering in memory)
static CPtr<const CTensorBase> renameDimensions( const CTensorBase& input, const CTensorLayout& outputLayout )
{
	if( input.Layout() == outputLayout ) {
		return &input;
	}

	if( input.Type() == TTensorType::Data ) {
		CPtr<const CDnnBlob> blob = renameDimensions( *dynamic_cast<const CDataTensor&>( input ).Data(),
			input.Layout(), outputLayout );
		return new CDataTensor( outputLayout, *blob );
	}

	CLayerOutput layerOutput = renameDimensions( dynamic_cast<const CUserTensor&>( input ).LayerOutput(),
		input.Layout(), outputLayout );
	return new CUserTensor( outputLayout, layerOutput );
}

// Swaps 2 dimensions of data blob
static CPtr<const CDnnBlob> swapDimensions( const CDnnBlob& inputBlob, TBlobDim firstDim, TBlobDim secondDim )
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

// Swaps 2 dimensions of given layer output
static CLayerOutput swapDimensions( const CLayerOutput& input, TBlobDim firstDim, TBlobDim secondDim )
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
static CPtr<const CTensorBase> swapDimensions( const CTensorBase& input, TBlobDim firstDim, TBlobDim secondDim )
{
	CTensorLayout outputLayout = input.Layout();
	const int firstDimIndex = outputLayout.Find( firstDim );
	const int secondDimIndex = outputLayout.Find( secondDim );
	NeoAssert( firstDimIndex != NotFound || secondDimIndex != NotFound );
	if( firstDimIndex != NotFound && secondDimIndex != NotFound ) {
		swap( outputLayout[firstDimIndex], outputLayout[secondDimIndex] );
	} else if( firstDimIndex != NotFound ) {
		outputLayout[firstDimIndex] = secondDim;
	} else {
		outputLayout[secondDimIndex] = firstDim;
	}

	if( input.Type() == TTensorType::Data ) {
		CPtr<const CDnnBlob> blob = swapDimensions( *dynamic_cast<const CDataTensor&>( input ).Data(),
			firstDim, secondDim );
		return new CDataTensor( outputLayout, *blob );
	}

	CLayerOutput layerOutput = swapDimensions( dynamic_cast<const CUserTensor&>( input ).LayerOutput(),
		firstDim, secondDim );
	return new CUserTensor( outputLayout, layerOutput );
}

// Checks that layout is a channel-last-like (NeoML compatible)
static inline bool isChannelLastLayout( const CTensorLayout& layout )
{
	for( int i = 2; i < layout.Size(); ++i ) {
		if( layout[0] > layout[i] || layout[1] < layout[i]
			|| ( i != 2 && layout[i] < layout[i - 1] ) )
		{
			return false;
		}
	}

	return true;
}

// Checks that layout is a channel-first-like (ONNX compatible)
static inline bool isChannelFirstLayout( const CTensorLayout& layout )
{
	for( int i = 1; i < layout.Size(); ++i ) {
		if( layout[i] < layout[i - 1] ) {
			return false;
		}
	}

	return true;
}

// Converts tensor from channel-first-like layout to channel-last-like layout
static CPtr<const CTensorBase> convertToChannelLast( const CTensorBase& input, const CTensorLayout& outputLayout )
{
	static_assert( BD_Count == 7, "BD_Count != 7" );
	const int dimCount = input.DimCount();
	NeoAssert( dimCount > 2 && dimCount < 7 );
	NeoAssert( isChannelFirstLayout( input.Layout() ) );
	NeoAssert( isChannelLastLayout( outputLayout ) );

	CPtr<const CTensorBase> currInput = &input;
	if( currInput->Layout().Find( BD_Channels ) != NotFound ) {
		// isChannelFirstLayout( input.Layout() ) guarantees that layout[i + 1] > layout[i]
		// The restriction dimCount < 7 guarantees that in this intermediate layout BD_Channels won't be used
		CTensorLayout intermediateLayout = CTensorLayout::IOLayout( dimCount );
		currInput = renameDimensions( *currInput, intermediateLayout );
	}

	NeoAssert( currInput->Layout().Find( BD_Channels ) == NotFound );
	currInput = swapDimensions( *currInput, currInput->Layout()[1], BD_Channels );
	return renameDimensions( *currInput, outputLayout );
}

// Converts tensor from channel-last-like layout to channel-first-like layout
static CPtr<const CTensorBase> convertToChannelFirst( const CTensorBase& input, const CTensorLayout& outputLayout )
{
	static_assert( BD_Count == 7, "BD_Count != 7" );
	const int dimCount = input.DimCount();
	NeoAssert( dimCount > 2 && dimCount < 7 );
	NeoAssert( isChannelLastLayout( input.Layout() ) );
	NeoAssert( isChannelFirstLayout( outputLayout ) );

	CPtr<const CTensorBase> currInput = &input;
	TBlobDim onnxChannelDim = currInput->Layout()[0] + 1;
	if( currInput->Layout()[2] == currInput->Layout()[0] + 1 ) {
		// We have make additional renaming
		CTensorLayout intermediateLayout( dimCount );
		intermediateLayout[0] = BD_BatchLength;
		intermediateLayout[1] = BD_Channels;
		for( int i = 2; i < dimCount; ++i ) {
			intermediateLayout[i] = BD_ListSize + ( i - 2 );
		}
		onnxChannelDim = BD_BatchWidth;
		currInput = renameDimensions( *currInput, intermediateLayout );
	}

	NeoAssert( currInput->Layout().Find( onnxChannelDim ) == NotFound );
	currInput = swapDimensions( *currInput, currInput->Layout()[1], onnxChannelDim );
	return renameDimensions( *currInput, outputLayout );
}

CPtr<const CTensorBase> ConvertTensor( const CTensorBase& input, const CTensorLayout& outputLayout )
{
	// Trivial case
	if( input.Layout() == outputLayout ) {
		return &input;
	}

	const int dimCount = outputLayout.Size();
	NeoAssert( input.DimCount() == dimCount );

	// Special cases for conversions between channel-first (ONNX) and channel-last (NeoML)
	if( dimCount > 2 && dimCount < 7 ) {
		if( isChannelFirstLayout( input.Layout() ) && isChannelLastLayout( outputLayout ) ) {
			return convertToChannelLast( input, outputLayout );
		} else if( isChannelLastLayout( input.Layout() ) && isChannelFirstLayout( outputLayout ) ) {
			return convertToChannelFirst( input, outputLayout );
		}
	}

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
	// Step 1 guarantees that outputLayout is a permutation of currentTensor.Layout()
	// NeoML has operations only for swapping 2 dimensions
	// that's why reordering is implemented as a sequence of swaps
	for( int dimIndex = 0; dimIndex < dimCount; ++dimIndex ) {
		TBlobDim inputDim = currentTensor->Layout()[dimIndex];
		TBlobDim outputDim = outputLayout[dimIndex];
		if( inputDim != outputDim ) {
			currentTensor = swapDimensions( *currentTensor, inputDim, outputDim );
		}
	}

	return currentTensor;
}

CPtr<const CDataTensor> ConvertTensor( const CDataTensor& dataTensor, const CTensorLayout& destLayout )
{
	return dynamic_cast<const CDataTensor*>( ConvertTensor( static_cast<const CTensorBase&>( dataTensor ), destLayout ).Ptr() );
}

CPtr<const CUserTensor> ConvertTensor( const CUserTensor& dataTensor, const CTensorLayout& destLayout )
{
	return dynamic_cast<const CUserTensor*>( ConvertTensor( static_cast<const CTensorBase&>( dataTensor ), destLayout ).Ptr() );
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

// Adds given image resize layer in order to resize heightDimIndex'th and widthDimIndex'th dimensions
// widthDimIndex may be NotFound (that means only heightDimIndex'th dimension should be resized)
static CPtr<const CUserTensor> addImageResizeLayer( CImageResizeLayer& imageResize, CDnn& dnn, const CUserTensor& input,
	int heightDimIndex, int widthDimIndex )
{
	// Add imageResize layer
	CPtr<const CUserTensor> result = convertTensorToHw( input, heightDimIndex, widthDimIndex );
	imageResize.Connect( 0, *result->Layer(), result->OutputIndex() );
	dnn.AddLayer( imageResize );

	// Construct new CUserTensor which is provided by imageResize layer
	return new CUserTensor( result->Layout(), CLayerOutput( &imageResize, 0 ) );
}

CPtr<const CUserTensor> PadUserTensor( const CUserTensor& input, const CFastArray<int, 8>& pads, float padValue )
{
	// Pool and conv operators storing pads only for N-2 tensor dimensions (leaving out batch and channels)
	// On the other side Pad operator is storing pads for every tensor dimension

	// Number of padded dimensions
	const int paddedDims = pads.Size() / 2;
	NeoAssert( pads.Size() == paddedDims * 2 );
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

	// In case of padding odd number of dimensions by this moment imageResize != nullptr
	// and widthDimIndex is equal to NotFound
	if( imageResize != nullptr ) {
		currData = addImageResizeLayer( *imageResize, dnn, *currData, heightDimIndex, widthDimIndex );
	}

	return currData;
}

//---------------------------------------------------------------------------------------------------------------------

CPtr<const CTensorBase> PrepareForBroadcast( const CTensorBase& input, const CBroadcast& broadcast, int outputDims )
{
	int axis = outputDims - input.DimCount();
	if( broadcast.Type == BT_Onnx && broadcast.Axis >= 0 && axis > broadcast.Axis ) {
		axis = broadcast.Axis;
	}
	NeoAssert( axis + input.DimCount() <= outputDims );

	const CTensorLayout& inputLayout = input.Layout();

	TBlobDim currDim = BD_BatchLength;
	CTensorLayout outputLayout;
	outputLayout.SetBufferSize( outputDims );
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
	for( int i = outputLayout.Size(); i < outputDims; ++i ) {
		while( inputLayout.Find( currDim ) != NotFound && currDim < BD_Count ) {
			++currDim;
		}
		NeoAssert( currDim != BD_Count );
		outputLayout.Add( currDim );
		++currDim;
	}

	if( input.Type() == TTensorType::Data ) {
		return new CDataTensor( outputLayout, *dynamic_cast<const CDataTensor&>( input ).Data() );
	}
	return new CUserTensor( outputLayout, dynamic_cast<const CUserTensor&>( input ).LayerOutput() );
}

//---------------------------------------------------------------------------------------------------------------------

CPtr<const CUserTensor> AsUserTensor( const CTensorBase& tensor, const CString& layerName, CDnn& dnn )
{
	static_assert( static_cast<int>( TTensorType::Count ) == 3, "TTensorType::Count != 3" );

	// TODO: Add conversion from Shape to user
	CheckNeoOnnxSupport( tensor.Type() != TTensorType::Shape, "Shape tensor can't be converted to User" );

	if( tensor.Type() == TTensorType::User ) {
		// No conversion needed
		return dynamic_cast<const CUserTensor*>( &tensor );
	}

	const CDataTensor& dataTensor = dynamic_cast<const CDataTensor&>( tensor );
	CPtr<CDataLayer> dataLayer = new CDataLayer( dnn.GetMathEngine() );
	dataLayer->SetBlob( dataTensor.Data()->GetCopy() );
	// Guarantee that serialization won't lead to data loss
	dataLayer->SetName( layerName );
	dnn.AddLayer( *dataLayer );
	return new CUserTensor( dataTensor.Layout(), CLayerOutput( dataLayer, 0 ) );
}

CPtr<const CShapeTensor> AsShapeTensor( const CTensorBase& tensor, const CString& layerName, CDnn& dnn )
{
	if( tensor.Type() == TTensorType::Shape ) {
		return CheckCast<const CShapeTensor>( &tensor );
	}

	CheckNeoOnnxSupport( tensor.Type() != TTensorType::User, "User tensor can't be converted to Shape" );

	CPtr<const CDataTensor> dataTensor = CheckCast<const CDataTensor>( &tensor );
	dataTensor = ConvertTensor( *dataTensor, CTensorLayout::IOLayout( dataTensor->DimCount() ) );

	CTensorShape resultShape;
	for( int dimIndex = 0; dimIndex < dataTensor->DimCount(); ++dimIndex ) {
		resultShape.Add( dataTensor->DimSize( dimIndex ) );
	}

	CPtr<CSourceReshaper> source = new CSourceReshaper( dnn.GetMathEngine() );
	source->SetName( layerName );
	source->Blob() = dataTensor->Data()->GetCopy();
	dnn.AddLayer( *source );
	return new CShapeTensor( resultShape, CLayerOutput( source.Ptr(), 0 ) );
}

} // namespace NeoOnnx
