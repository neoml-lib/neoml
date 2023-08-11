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
#include <NeoML/Dnn/Layers/Onnx/OnnxSourceHelper.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxTransformHelper.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxTransposeHelper.h>
#include <NeoML/Dnn/Layers/Onnx/OnnxShapeToBlobLayer.h>

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
	NeoPresume( inputLayout.Size() == outputLayout.Size() );
	CDnn& dnn = *( input.Layer->GetDnn() );
	CPtr<COnnxTransformHelper> transformLayer = new COnnxTransformHelper( dnn.GetMathEngine(), inputLayout, outputLayout );
	transformLayer->SetName( getUniqueLayerName( dnn, "transform_" ) );
	for( int dimIndex = 0; dimIndex < outputLayout.Size(); ++dimIndex ) {
		transformLayer->SetRule( inputLayout[dimIndex], outputLayout[dimIndex] );
	}
	dnn.AddLayer( *transformLayer );
	transformLayer->Connect( 0, *input.Layer, input.OutputIndex );
	return CLayerOutput( transformLayer.Ptr(), 0 );
}

// Renames dimensions of tensor (without any reordering in memory)
static CPtr<const CTensorBase> renameDimensions( const CTensorBase& input, const CTensorLayoutRename& rename )
{
	if( rename.From == rename.To ) {
		return &input;
	}

	CTensorLayout outputLayout;
	outputLayout.SetBufferSize( input.DimCount() );
	for( int dimIndex = 0; dimIndex < input.DimCount(); ++dimIndex ) {
		const int sortedDimIndex = rename.From.Find( input.Layout()[dimIndex] );
		outputLayout.Add( rename.To[sortedDimIndex] );
	}

	if( input.Type() == TTensorType::Data ) {
		CPtr<const CDnnBlob> blob = renameDimensions( *dynamic_cast<const CDataTensor&>( input ).Data(),
			input.Layout(), outputLayout );
		return new CDataTensor( outputLayout, *blob );
	}

	CLayerOutput layerOutput = input.Type() == TTensorType::User
		? dynamic_cast<const CUserTensor&>( input ).LayerOutput()
		: dynamic_cast<const CShapeTensor&>( input ).LayerOutput();
	layerOutput = renameDimensions( layerOutput, input.Layout(), outputLayout );

	if( input.Type() == TTensorType::User ) {
		return new CUserTensor( outputLayout, layerOutput );
	}

	return new CShapeTensor( outputLayout, dynamic_cast<const CShapeTensor&>( input ).Shape(), layerOutput );
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
static CLayerOutput swapDimensions( const CLayerOutput& input, TBlobDim firstDim, TBlobDim secondDim,
	const CTensorLayout& inputLayout, const CTensorLayout& outputLayout )
{
	CDnn& dnn = *( input.Layer->GetDnn() );
	CPtr<COnnxTransposeHelper> transposeLayer = new COnnxTransposeHelper( dnn.GetMathEngine(),
		inputLayout, outputLayout );
	transposeLayer->SetName( getUniqueLayerName( dnn, "transpose_" ) );
	transposeLayer->SetDims( firstDim, secondDim );
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

	CLayerOutput layerOutput = input.Type() == TTensorType::User
		? dynamic_cast<const CUserTensor&>( input ).LayerOutput()
		: dynamic_cast<const CShapeTensor&>( input ).LayerOutput();
	layerOutput = swapDimensions( layerOutput, firstDim, secondDim, input.Layout(), outputLayout );

	if( input.Type() == TTensorType::User ) {
		return new CUserTensor( outputLayout, layerOutput );
	}

	return new CShapeTensor( outputLayout, dynamic_cast<const CShapeTensor&>( input ).Shape(), layerOutput );
}

CPtr<const CTensorBase> ConvertTensor( const CTensorBase& input, const ITensorLayoutValidator& validator )
{
	CTensorLayoutConversion conversion;
	( void ) FindConversion( input.Layout(), validator, conversion );

	CPtr<const CTensorBase> currentTensor = &input;

	// Step 1: renaming dimensions before transpositions (if needed)
	// Renaming means assigning outputLayout's dimensions to the ones of input.Layout()
	// without data transpositions
	currentTensor = renameDimensions( *currentTensor, conversion.PreTransposeRename );

	// Step 2: reordering dimensions (apply transpositions)
	for( const CTensorLayoutTranspose& transpose : conversion.Transposes ) {
		currentTensor = swapDimensions( *currentTensor, transpose.First, transpose.Second );
	}

	// Step 3: renaming dimensions after transpositions (if needed)
	return renameDimensions( *currentTensor, conversion.PostTransposeRename );
}

CPtr<const CUserTensor> ConvertTensor( const CUserTensor& userTensor, const ITensorLayoutValidator& validator )
{
	return dynamic_cast<const CUserTensor*>( ConvertTensor( static_cast<const CTensorBase&>( userTensor ), validator ).Ptr() );
}

CPtr<const CTensorBase> ConvertTensor( const CTensorBase& input, const CTensorLayout& outputLayout )
{
	// Trivial case
	if( input.Layout() == outputLayout ) {
		return &input;
	}

	NeoAssert( input.DimCount() == outputLayout.Size() );

	CTensorLayoutMatchValidator validator( outputLayout );
	CPtr<const CTensorBase> output = ConvertTensor( input, validator );
	NeoAssert( output->Layout() == outputLayout );
	return output;
}

CPtr<const CDataTensor> ConvertTensor( const CDataTensor& dataTensor, const CTensorLayout& destLayout )
{
	return dynamic_cast<const CDataTensor*>( ConvertTensor( static_cast<const CTensorBase&>( dataTensor ), destLayout ).Ptr() );
}

CPtr<const CUserTensor> ConvertTensor( const CUserTensor& userTensor, const CTensorLayout& destLayout )
{
	return dynamic_cast<const CUserTensor*>( ConvertTensor( static_cast<const CTensorBase&>( userTensor ), destLayout ).Ptr() );
}

CPtr<const CShapeTensor> ConvertTensor( const CShapeTensor& shapeTensor, const CTensorLayout& destLayout )
{
	return dynamic_cast<const CShapeTensor*>( ConvertTensor( static_cast<const CTensorBase&>( shapeTensor ), destLayout ).Ptr() );
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

CPtr<const CUserTensor> PadUserTensor( const CUserTensor& input, const CFastArray<int, 8>& pads,
	TBlobResizePadding padding, float padValue )
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
			imageResize->SetPadding( padding );
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

// Returns true if shapes are equal
static bool areShapesEqual( const CTensorShape& first, const CTensorShape& second )
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

	// The shape with lesser number of dimensions must be padded
	const CTensorShape& lesserShape = first.Size() <= second.Size() ? first : second;
	const CTensorShape& biggerShape = first.Size() > second.Size() ? first  : second;
	CTensorShape paddedShape;
	paddedShape.Add( 1, axis );
	paddedShape.Add( lesserShape );
	if( paddedShape.Size() > biggerShape.Size() ) {
		// Wrong broadcast parameters (axis value is too big)
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

CPtr<const CTensorBase> PrepareForBroadcast( const CTensorBase& input, const CBroadcast& broadcast, int outputDims )
{
	const bool isShapeTensor = input.Type() == TTensorType::Shape;

	int axis = outputDims - input.DimCount();
	if( broadcast.Type == BT_Onnx && broadcast.Axis >= 0 && axis > broadcast.Axis ) {
		axis = broadcast.Axis;
	}

	CTensorShape outputShape;
	if( isShapeTensor ) {
		const CTensorShape& inputShape = dynamic_cast<const CShapeTensor&>( input ).Shape();
		NeoAssert( axis + inputShape.Size() <= outputDims );
		outputShape.Add( 1, axis );
		outputShape.Add( inputShape );
		outputShape.Add( 1, outputDims - outputShape.Size() );
	}

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
	} else if( isShapeTensor ) {
		return new CShapeTensor( outputLayout, outputShape,
			dynamic_cast<const CShapeTensor&>( input ).LayerOutput() );
	}
	return new CUserTensor( outputLayout, dynamic_cast<const CUserTensor&>( input ).LayerOutput() );
}

//---------------------------------------------------------------------------------------------------------------------

CPtr<const CUserTensor> AsUserTensor( const CTensorBase& tensor, const CString& layerName, CDnn& dnn )
{
	static_assert( static_cast<int>( TTensorType::Count ) == 3, "TTensorType::Count != 3" );

	if( tensor.Type() == TTensorType::User ) {
		// No conversion needed
		return dynamic_cast<const CUserTensor*>( &tensor );
	}

	if( tensor.Type() == TTensorType::Shape ) {
		// Convert shape to usual blob via special layer
		CPtr<COnnxShapeToBlobLayer> conversionLayer = new COnnxShapeToBlobLayer( dnn.GetMathEngine() );
		conversionLayer->SetName( layerName );
		const CShapeTensor& input = dynamic_cast<const CShapeTensor&>( tensor );
		conversionLayer->Connect( 0, *input.Layer(), input.OutputIndex() );
		dnn.AddLayer( *conversionLayer );
		return new CUserTensor( input.Layout(), CLayerOutput( conversionLayer, 0 ) );
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

	CTensorShape resultShape;
	for( int dimIndex = 0; dimIndex < dataTensor->DimCount(); ++dimIndex ) {
		resultShape.Add( dataTensor->DimSize( dimIndex ) );
	}

	CPtr<COnnxSourceHelper> source = new COnnxSourceHelper( dnn.GetMathEngine() );
	source->SetName( layerName );
	source->Blob() = dataTensor->Data()->GetCopy();
	dnn.AddLayer( *source );
	return new CShapeTensor( dataTensor->Layout(), resultShape, CLayerOutput( source.Ptr(), 0 ) );
}

template<class T>
static CPtr<const CShapeTensor> asShapeTensor( const CFastArray<T, 8>& data, const CString& layerName, CDnn& dnn )
{
	CPtr<COnnxSourceHelper> source = new COnnxSourceHelper( dnn.GetMathEngine() );
	source->SetName( layerName );
	source->Blob() = CDnnBlob::CreateTensor( dnn.GetMathEngine(), CBlobType<T>::GetType(), { data.Size() } );
	source->Blob()->CopyFrom( data.GetPtr() );
	dnn.AddLayer( *source );
	return new CShapeTensor( CTensorLayout::IOLayout( 1 ), { data.Size() },
		CLayerOutput( source.Ptr(), 0 ) );
}

CPtr<const CShapeTensor> AsShapeTensor( const CFastArray<int, 8>& data, const CString& layerName, CDnn& dnn )
{
	return asShapeTensor( data, layerName, dnn );
}

CPtr<const CShapeTensor> AsShapeTensor( const CFastArray<float, 8>& data, const CString& layerName, CDnn& dnn )
{
	return asShapeTensor( data, layerName, dnn );
}

//---------------------------------------------------------------------------------------------------------------------

void GetTensorShape( const CTensorBase& tensor, CTensorShape& shape )
{
	switch( tensor.Type() ) {
		case TTensorType::Shape:
			dynamic_cast<const CShapeTensor&>( tensor ).Shape().CopyTo( shape );
			return;
		case TTensorType::Data:
		{
			const CDataTensor& dataTensor = dynamic_cast<const CDataTensor&>( tensor );
			shape.SetSize( tensor.DimCount() );
			for( int dimIndex = 0; dimIndex < tensor.DimCount(); ++dimIndex ) {
				shape[dimIndex] = dataTensor.DimSize( dimIndex );
			}
			return;
		}
		case TTensorType::User:
		default:
			CheckNeoOnnxSupport( false, "Can't extract tensor shape" );
	}
}

//---------------------------------------------------------------------------------------------------------------------

// Encodes layout into 32-bit integer
static int tensorLayoutHash( const CTensorLayout& layout )
{
	static_assert( static_cast<int>( BD_Count ) <= 8, "BD_Count > 8" );
	int result = 0;
	for( int i = 0; i < layout.Size(); ++i ) {
		result |= static_cast<int>( layout[i] ) << ( 3 * i );
	}
	return result;
}

class CLayoutConversionBfs {
public:
	explicit CLayoutConversionBfs( const ITensorLayoutValidator& validator ) :
		validator( validator ), queueResultIndex( NotFound ) {}

	CTensorLayout Find( const CTensorLayout& inputLayout, CTensorLayoutConversion& conversion );

private:
	struct CBfsEntry {
		CBfsEntry() = default;

		CBfsEntry( const CTensorLayoutRename& rename, const CTensorLayout& outputLayout ) : OutputLayout( outputLayout )
			{ Conversion.PreTransposeRename = rename; }

		CTensorLayoutConversion Conversion;
		CTensorLayout OutputLayout;
	};

	const ITensorLayoutValidator& validator;

	CHashTable<int> visited;
	CArray<CBfsEntry> queue;
	int queueResultIndex;
	
	void setOptimum( int index ) { queueResultIndex = index; }
	bool isOptimumFound() const { return queueResultIndex != NotFound; }
	int optimum() const { return queueResultIndex; }
	void addInitialRenames( const CTensorLayout& inputLayout );
	void addTranspose( const CBfsEntry& entry, const CTensorLayoutTranspose& transpose );
	void bruteForceRenames( const CBfsEntry& entry, bool addToQueue );
};

CTensorLayout CLayoutConversionBfs::Find( const CTensorLayout& inputLayout, CTensorLayoutConversion& conversion )
{
	setOptimum( NotFound );
	addInitialRenames( inputLayout );

	// layout of size 1 must be converted without transposes
	NeoAssert( isOptimumFound() || inputLayout.Size() > 1 );

	int queueIndex = 0;

	while( !isOptimumFound() ) {
		NeoAssert( queueIndex < queue.Size() );

		const CBfsEntry entry = queue[queueIndex];

		TBlobDim secondLastAxis = BD_BatchLength;
		TBlobDim lastAxis = BD_BatchLength;
		for( TBlobDim dim : entry.OutputLayout ) {
			if( dim > lastAxis ) {
				secondLastAxis = lastAxis;
				lastAxis = dim;
			} else if( dim > secondLastAxis ) {
				secondLastAxis = dim;
			}
		}

		// Add all transposes which are equivalent to transposition of a batch of 2 dimensional tensors (B x H x W)
		// 1. Transpose last 2 used axes
		// B x H x W is (D_0...D_n-3) x D_n-2 x D_n-1
		CTensorLayoutTranspose transpose( secondLastAxis, lastAxis );
		addTranspose( entry, transpose );
		// 2. Transpose non-last used axis with unused axis after last axis
		// B x H x W is (D_0...D_i-1) x D_i x (D_i+1...D_n-1)
		for( TBlobDim first : entry.OutputLayout ) {
			transpose.First = first;
			if( transpose.First == lastAxis ) {
				continue;
			}
			for( transpose.Second = lastAxis + 1; transpose.Second != BD_Count; ++transpose.Second ) {
				addTranspose( entry, transpose );
			}
		}
		// 3. Transpose last used axis with unused axis less than last axis
		// B x H x W is (D_0...D_i-1) x (D_i...D_n-2) x D_n-1
		transpose.Second = lastAxis;
		for( transpose.First = BD_BatchLength; transpose.First != transpose.Second; ++transpose.First ) {
			if( entry.OutputLayout.Find( transpose.First ) != NotFound ) {
				continue;
			}
			addTranspose( entry, transpose );
		}

		// Add all other possible transposes
		for( transpose.First = BD_BatchLength; transpose.First != BD_Channels; ++transpose.First ) {
			for( transpose.Second = transpose.First + 1; transpose.Second != BD_Count; ++transpose.Second ) {
				addTranspose( entry, transpose );
			}
		}

		queueIndex++;
	}

	NeoAssert( isOptimumFound() );
	const int optimumIndex = optimum();
	conversion = queue[optimumIndex].Conversion;
	return queue[optimumIndex].OutputLayout;
}

// Adds all possible renames of input layout to the queue
void CLayoutConversionBfs::addInitialRenames( const CTensorLayout& inputLayout )
{
	visited.Add( tensorLayoutHash( inputLayout ) );
	queue.SetSize( 1 );
	queue[0].OutputLayout = inputLayout;

	// Corner case - input layout is a valid layout
	if( validator( queue[0].OutputLayout ) ) {
		setOptimum( 0 );
		return;
	}

	const CBfsEntry firstEntry = queue[0];
	bruteForceRenames( firstEntry, true );
}

// Adds transposition to current bfs entry
// Also brute-forces post-transpose renames
void CLayoutConversionBfs::addTranspose( const CBfsEntry& entry, const CTensorLayoutTranspose& transpose )
{
	if( isOptimumFound() ) {
		return;
	}

	CBfsEntry transposedEntry = entry;
	transposedEntry.Conversion.Transposes.Add( transpose );

	const int firstIndex = entry.OutputLayout.Find( transpose.First );
	const int secondIndex = entry.OutputLayout.Find( transpose.Second );
	if( firstIndex == NotFound && secondIndex == NotFound ) {
		return;
	} else if( firstIndex == NotFound ) {
		transposedEntry.OutputLayout[secondIndex] = transpose.First;
	} else if( secondIndex == NotFound ) {
		transposedEntry.OutputLayout[firstIndex] =  transpose.Second;
	} else {
		std::swap( transposedEntry.OutputLayout[firstIndex], transposedEntry.OutputLayout[secondIndex] );
	}

	if( validator( transposedEntry.OutputLayout ) ) {
		queue.Add( transposedEntry );
		setOptimum( queue.Size() - 1 );
		return;
	}

	const int hash = tensorLayoutHash( transposedEntry.OutputLayout );
	if( !visited.Has( hash ) ) {
		queue.Add( transposedEntry );
		visited.Add( hash );
	}

	bruteForceRenames( transposedEntry, false );
}

// Brute-forces all possible renames for current bfs entry
void CLayoutConversionBfs::bruteForceRenames( const CBfsEntry& entry, bool addToQueue )
{
	if( isOptimumFound() ) {
		return;
	}

	CBfsEntry bruteForceEntry = entry;
	const bool isPostRename = ( !entry.Conversion.PreTransposeRename.From.IsEmpty()
		|| !entry.Conversion.Transposes.IsEmpty() );
	const CTensorLayout& inputLayout = entry.OutputLayout; // Initial layout, it won't be changed during brute-forced
	// Rename which will be changed during brute-force
	CTensorLayoutRename& rename = isPostRename ? bruteForceEntry.Conversion.PostTransposeRename
		: bruteForceEntry.Conversion.PreTransposeRename;
	// Layout which will be changed during brute-force
	CTensorLayout& outputLayout = bruteForceEntry.OutputLayout;

	inputLayout.CopyTo( rename.From );
	rename.From.QuickSort<Ascending<TBlobDim>>();
	rename.To.SetSize( inputLayout.Size() );

	static_assert( static_cast<int>( BD_Count ) == 7, "BD_Count != 7" );

	auto bruteForce = [&] ( int sortedAxisIndex, auto&& bruteForce ) -> void
	{
		NeoPresume( !isOptimumFound() );

		const bool isLastAxis = sortedAxisIndex == outputLayout.Size() - 1;
		const int axisIndex = inputLayout.Find( rename.From[sortedAxisIndex] );
		const int minValue = sortedAxisIndex == 0 ? 0 : static_cast<int>( rename.To[sortedAxisIndex - 1] ) + 1;
		const int maxValue = static_cast<int>( BD_Count ) - ( outputLayout.Size() - sortedAxisIndex );

		for( int value = minValue; value <= maxValue; ++value ) {
			rename.To[sortedAxisIndex] = static_cast<TBlobDim>( value );
			outputLayout[axisIndex] = static_cast<TBlobDim>( value );
			if( isLastAxis ) {
				const bool isValid = validator( outputLayout );
				const int hash = ( addToQueue || isValid ) ? tensorLayoutHash( outputLayout ) : 0;
				NeoPresume( !isValid || !visited.Has( hash ) ); // if layout is valid then it shouldn't be visited before
				if( isValid || ( addToQueue && !visited.Has( hash ) ) ) {
					visited.Add( hash );
					queue.Add( bruteForceEntry );
				}
				if( isValid ) {
					setOptimum( queue.Size() - 1 );
					return;
				}
			} else {
				bruteForce( sortedAxisIndex + 1, bruteForce );
				if( isOptimumFound() ) {
					return;
				}
			}
		}
	};

	bruteForce( 0, bruteForce );
}

//---------------------------------------------------------------------------------------------------------------------

CTensorLayout FindConversion( const CTensorLayout& inputLayout, const ITensorLayoutValidator& validator,
	CTensorLayoutConversion& conversion )
{
	return CLayoutConversionBfs( validator ).Find( inputLayout, conversion );
}

} // namespace NeoOnnx
