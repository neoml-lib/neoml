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

#include "PadOperator.h"
#include "NeoOnnxCheck.h"
#include "TensorUtils.h"

#include "onnx.pb.h"

namespace NeoOnnx {

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

// Generate unique layer name for dnn
static CString getUniqueLayerName( const CString& prefix, const CDnn& dnn )
{
	int currIndex = dnn.GetLayerCount();
	CString currName = prefix + Str( currIndex );
	while( dnn.HasLayer( currName ) ) {
		++currIndex;
		currName = prefix + Str( currIndex );
	}
	return currName;
}

// Converts tensor prior to imageResizeLayer
CPtr<const CUserTensor> convertTensorBeforeImageResize( const CUserTensor& input, int heightDimIndex, int widthDimIndex )
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

CPtr<const CUserTensor> addImageResizeLayer( CImageResizeLayer& imageResize, CDnn& dnn, const CUserTensor& input,
	int heightDimIndex, int widthDimIndex )
{
	// Add imageResize layer
	CPtr<const CUserTensor> result = convertTensorBeforeImageResize( input, heightDimIndex, widthDimIndex );
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
			imageResize->SetName( getUniqueLayerName( padNamePrefix, dnn ) );
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

CPadOperator::CPadOperator( const onnx::NodeProto& pad, int opsetVersion ) :
	CLayerOperator( pad, opsetVersion ),
	mode( Attributes.GetOptionalString( "mode", "constant" ) ),
	value( 0.f )
{
	// In v1 pads are provided by 'paddings' attribute and pad value is provided by 'value' attribute 
	// In v2 pads are provided by 'pads' attribute and pad value is provided by 'value' attribute 
	// In v11 pads and pad value are provided by additional inputs
	if( opsetVersion < 11 ) {
		CheckOnnxProtocol( InputCount() == 1, "operator must have 1 input", *this );
		Attributes.GetRequiredIntArray( opsetVersion == 1 ? "paddings" : "pads", pads );
		value = Attributes.GetOptionalFloat( "value", 0.f );
	} else {
		CheckOnnxProtocol( InputCount() == 2 || InputCount() == 3, "operator must have 2 or 3 inputs", *this );
	}

	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );
	CheckNeoOnnxSupport( mode == "constant", "Pad with non-constant mode", *this );
}

void CPadOperator::AddLayers( const CObjectArray<const CTensorBase>& inputs,
	CDnn& /* dnn */, CObjectArray<const CTensorBase>& outputs )
{
	if( OpsetVersion >= 11 ) {
		CheckNeoOnnxSupport( inputs[1]->IsCalculated(), "user-provided pad sizes", *this );
		const CDnnBlob& padsBlob = *( dynamic_cast<const CDataTensor*>( inputs[1].Ptr() )->Data() );
		CheckOnnxProtocol( padsBlob.GetDataType() == CT_Int, "non-integer pad sizes", *this );
		pads.SetSize( padsBlob.GetDataSize() );
		padsBlob.CopyTo( pads.GetPtr() );
		if( InputCount() == 3 ) {
			CheckNeoOnnxSupport( inputs[2]->IsCalculated(), "user-provided pad value", *this );
			const CDnnBlob& valueBlob = *( dynamic_cast<const CDataTensor*>( inputs[2].Ptr() )->Data() );
			if( valueBlob.GetDataType() == CT_Float ) {
				value = valueBlob.GetData<float>().GetValue();
			} else {
				value = static_cast<float>( valueBlob.GetData<int>().GetValue() );
			}
		}
	}

	outputs[0] = PadUserTensor( dynamic_cast<const CUserTensor&>( *inputs[0] ), pads, value ).Ptr();
}

} // namespace NeoOnnx
