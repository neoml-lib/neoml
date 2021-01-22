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

#include <common.h>
#pragma hdrstop

#include <NeoML/Dnn/Layers/ImageAndPixelConversionLayer.h>
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

// The common part for two layers

static const int defaultPseudoImageForConvertationSize = 128;

// Checks the validity of the indices blob
static void checkIndicesBlob( const CBlobDesc& indices )
{
	// Check the type of data in the blob
	NeoAssert( indices.GetDataType() == CT_Int );
	// The blob size should be 1 x BatchWidth x Height.
	NeoAssert( indices.BatchLength() == 1 );
	NeoAssert( indices.GeometricalSize() == 1 );
}

// Initialize the blob with index offsets
static void initializeShift( CDnnBlob& shift, int imageHeight, int imageWidth )
{
	CArray<int> buffer;
	buffer.SetSize( shift.GetDataSize() );
	for( int i = 0; i < buffer.Size(); ++i ) {
		buffer[i] = i * imageHeight * imageWidth;
	}
	shift.CopyFrom<int>( buffer.GetPtr() );
}

// Calculates the shifted indices to use the set of images from which 
// the pixels should be extracted as one lookup table of batchSize x imageHeight x imageWidth size
static void shiftIndices( IMathEngine& mathEngine, CDnnBlob& indices, CDnnBlob& shift, CDnnBlob& result )
{
	mathEngine.AddVectorToMatrixColumns( indices.GetData<int>(), result.GetData<int>(),
		indices.GetBatchWidth(), indices.GetObjectSize(), shift.GetData<int>() );
}

// Converts the set of pixels and their indices into images
static void convertPixelToImage( IMathEngine& mathEngine, CDnnBlob& pixel, CDnnBlob& shiftedIndices, CDnnBlob& result )
{
	const int batchSize = pixel.GetBatchWidth();
	const int objectsCount = pixel.GetListSize();
	const int featuresCount = pixel.GetChannelsCount();

	const int imageHeight = result.GetHeight();
	const int imageWidth = result.GetWidth();

	mathEngine.MatrixSpreadRows( pixel.GetData(), batchSize * objectsCount, featuresCount,
		result.GetData(), batchSize * imageHeight * imageWidth, shiftedIndices.GetData<int>(),
		CConstFloatHandle() );
}

// Converts images and indices into pixel sets
static void convertImageToPixel( IMathEngine& mathEngine, CDnnBlob& images, CDnnBlob& shiftedIndices, CDnnBlob& result )
{
	const int batchSize = images.GetBatchWidth();
	const int featuresCount = images.GetChannelsCount();

	const int imageHeight = images.GetHeight();
	const int imageWidth = images.GetWidth();

	CLookupDimension lookupDim;
	lookupDim.VectorCount = batchSize * imageHeight * imageWidth;
	lookupDim.VectorSize = featuresCount;

	CConstFloatHandle lookupTable = images.GetData();

	result.Fill( 0.f );
	mathEngine.VectorMultichannelLookupAndCopy( shiftedIndices.GetDataSize(), 1,
		shiftedIndices.GetData<int>(), &lookupTable, &lookupDim, 1, result.GetData(), featuresCount );
}

// ====================================================================================================================

CPixelToImageLayer::CPixelToImageLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CCnnPixelToImageLayer", false ),
	imageHeight( defaultPseudoImageForConvertationSize ),
	imageWidth( defaultPseudoImageForConvertationSize )
{
}

static const int PixelToImageLayerVersion = 2000;

void CPixelToImageLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( PixelToImageLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize( archive );

	archive.Serialize( imageHeight );
	archive.Serialize( imageWidth );
}

void CPixelToImageLayer::Reshape()
{
	// Check the inputs
	CheckInputs();

	// The layer needs two inputs
	NeoAssert( GetInputCount() == 2 );

	// Check the indices
	checkIndicesBlob( inputDescs[1] );

	// Check the input data type
	NeoAssert( inputDescs[0].GetDataType() == CT_Float );

	// The minimum output image size
	NeoAssert( imageHeight >= 1 );
	NeoAssert( imageWidth >= 1 );

	// The layer may not be used in a recurrent sub-network
	NeoAssert( inputDescs[0].BatchLength() == 1 );

	// The data should be represented as a matrix
	NeoAssert( inputDescs[0].Depth() == 1 );

	// Checks the data validity
	NeoAssert( inputDescs[0].BatchWidth() == inputDescs[1].BatchWidth() );
	NeoAssert( inputDescs[0].ListSize() == inputDescs[1].ObjectSize() );

	// Calculate and set the output blob size
	const int outputBatchWidth = inputDescs[0].BatchWidth();
	const int outputHeight = imageHeight;
	const int outputWidth = imageWidth;
	const int outputChannelsCount = inputDescs[0].Channels();
	outputDescs[0] = CBlobDesc( CT_Float );
	outputDescs[0].SetDimSize( BD_BatchWidth, outputBatchWidth );
	outputDescs[0].SetDimSize( BD_Height, outputHeight );
	outputDescs[0].SetDimSize( BD_Width, outputWidth );
	outputDescs[0].SetDimSize( BD_Channels, outputChannelsCount );

	// Initialize the vector that should be added to the indices matrix columns
	// After which the images set may be interpreted as a big lookup table of batchSize x imageHeight x imageWidth size
	shift = CDnnBlob::CreateVector( MathEngine(), CT_Int, outputBatchWidth );
	initializeShift( *shift, imageHeight, imageWidth );
	shiftedIndices = CDnnBlob::CreateBlob( MathEngine(), CT_Int, inputDescs[1] );
	RegisterRuntimeBlob( shiftedIndices );
}

void CPixelToImageLayer::RunOnce()
{
	// The layer needs two inputs and one output
	// Additional dimension checks are performed in Reshape()
	NeoAssert( inputBlobs.Size() == 2 );
	NeoAssert( outputBlobs.Size() == 1 );

	// Initialize the offset indices vector
	shiftIndices( MathEngine(), *inputBlobs[1], *shift, *shiftedIndices );

	convertPixelToImage( MathEngine(), *inputBlobs[0], *shiftedIndices, *outputBlobs[0] );
}

void CPixelToImageLayer::BackwardOnce()
{
	// Get the outputDiffBlob.
	// Use it to find the inputDiffBlobs for which RunOnce returns the outputDiffBlob
	// Similar to calculating gradients in tensorflow.scatter_nd
	// (Python\Python36\Lib\site-packages\tensorflow\python\ops\array_grad.py),
	// simply invert the data positioning according to the existing indices set

	// The layer needs two inputs and one output
	// Additional dimension checks are performed in Reshape()
	NeoAssert( inputBlobs.Size() == 2 );
	NeoAssert( inputDiffBlobs.Size() == 2 );
	NeoAssert( outputDiffBlobs.Size() == 1 );

	convertImageToPixel( MathEngine(), *outputDiffBlobs[0], *shiftedIndices, *inputDiffBlobs[0] );
}

void CPixelToImageLayer::SetImageHeight( int newHeight )
{
	NeoAssert( newHeight > 0 );
	imageHeight = newHeight;
}

void CPixelToImageLayer::SetImageWidth( int newWidth )
{
	NeoAssert( newWidth > 0 );
	imageWidth = newWidth;
}

CLayerWrapper<CPixelToImageLayer> PixelToImage( int imageHeight, int imageWidth )
{
	return CLayerWrapper<CPixelToImageLayer>( "PixelToImage", [=]( CPixelToImageLayer* result ) {
		result->SetImageHeight( imageHeight );
		result->SetImageWidth( imageWidth );
	} );
}

// ====================================================================================================================

CImageToPixelLayer::CImageToPixelLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CCnnImageToPixelLayer", false )
{
}

void CImageToPixelLayer::Reshape()
{
	// Check the inputs
	CheckInputs();

	// The layer needs two inputs
	NeoAssert( GetInputCount() == 2 );

	// Check the indices
	checkIndicesBlob( inputDescs[1] );

	// Check the input data type
	NeoAssert( inputDescs[0].GetDataType() == CT_Float );

	// The layer may not be used in a recurrent sub-network
	NeoAssert( inputDescs[0].BatchLength() == 1 );

	// The index data should be represented as a matrix
	NeoAssert( inputDescs[0].Depth() == 1 );

	// Check the data validity
	NeoAssert( inputDescs[0].BatchWidth() == inputDescs[1].BatchWidth() );

	NeoAssert( inputDescs[0].Depth() == 1 );

	// Calculate and set the output blob size
	const int outputBatchWidth = inputDescs[0].BatchWidth();
	outputDescs[0] = CBlobDesc( CT_Float );
	outputDescs[0].SetDimSize( BD_BatchWidth, outputBatchWidth );
	outputDescs[0].SetDimSize( BD_ListSize, inputDescs[1].ObjectSize() );
	outputDescs[0].SetDimSize( BD_Channels, inputDescs[0].Channels() );

	// Initialize the vector that should be added to the indices matrix columns
	// After which the images set may be interpreted as a big lookup table of batchSize x imageHeight x imageWidth size
	shift = CDnnBlob::CreateVector( MathEngine(), CT_Int, outputBatchWidth );
	initializeShift( *shift, inputDescs[0].Height(), inputDescs[0].Width() );
	shiftedIndices = CDnnBlob::CreateBlob( MathEngine(), CT_Int, inputDescs[1] );
	RegisterRuntimeBlob( shiftedIndices );
}

void CImageToPixelLayer::RunOnce()
{
	// The layer needs two inputs and one output
	// Additional dimension checks are performed in Reshape()
	NeoAssert( inputBlobs.Size() == 2 );
	NeoAssert( outputBlobs.Size() == 1 );

	// Initialize the offset indices vector
	shiftIndices( MathEngine(), *inputBlobs[1], *shift, *shiftedIndices );

	convertImageToPixel( MathEngine(), *inputBlobs[0], *shiftedIndices, *outputBlobs[0] );
}

void CImageToPixelLayer::BackwardOnce()
{
	// Get the outputDiffBlob
	// Use it to find the inputDiffBlobs for which RunOnce returns the outputDiffBlob
	// Similar to calculating gradients in tensorflow.scatter_nd
	// (Python\Python36\Lib\site-packages\tensorflow\python\ops\array_grad.py),
	// simply invert the data positioning according to the existing indices set

	// The layer needs two inputs and one output
	// Additional dimension checks are performed in Reshape()
	NeoAssert( inputBlobs.Size() == 2 );
	NeoAssert( inputDiffBlobs.Size() == 2 );
	NeoAssert( outputDiffBlobs.Size() == 1 );

	convertPixelToImage( MathEngine(), *outputDiffBlobs[0], *shiftedIndices, *inputDiffBlobs[0] );
}

static const int ImageToPixelLayerVersion = 2000;

void CImageToPixelLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( ImageToPixelLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize( archive );
}

CLayerWrapper<CImageToPixelLayer> ImageToPixel()
{
	return CLayerWrapper<CImageToPixelLayer>( "ImageToPixel" );
}

} // namespace NeoML
