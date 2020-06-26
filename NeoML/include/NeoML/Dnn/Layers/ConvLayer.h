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

#pragma once

#include <NeoML/NeoMLDefs.h>
#include <NeoML/Dnn/Layers/BatchNormalizationLayer.h>
#include <NeoML/Dnn/Dnn.h>

namespace NeoML {

// CBaseConvLayer is a base class for convolution layers
class NEOML_API CBaseConvLayer : public CBaseLayer {
public:
	void Serialize( CArchive& archive ) override;

	// Filter size
	int GetFilterHeight() const { return filterHeight; }
	void SetFilterHeight( int filterHeight );
	int GetFilterWidth() const { return filterWidth; }
	void SetFilterWidth( int filterWidth );
	// Filter stride, vertical and horizontal
	int GetStrideHeight() const { return strideHeight; }
	void SetStrideHeight( int _strideHeight );
	int GetStrideWidth() const { return strideWidth; }
	void SetStrideWidth( int _strideWidth );

	// Image padding (additional zero-filled columns or rows at the image edges)
	// The padding size is the same for the opposite edges
	int GetPaddingHeight() const { return paddingHeight; }
	void SetPaddingHeight( int _paddingHeight );
	int GetPaddingWidth() const { return paddingWidth; }
	void SetPaddingWidth( int _paddingWidth );

	// The filter dilation
	int GetDilationHeight() const { return dilationHeight; }
	void SetDilationHeight( int newDilationHeight );
	int GetDilationWidth() const { return dilationWidth; }
	void SetDilationWidth( int newDilationWidth );

	// The number of filters per input channel
	int GetFilterCount() const { return filterCount; }
	void SetFilterCount( int _filterCount );

	// Returns or sets the filter data (the blobs are copied)
	// A filter blob has the FilterCount * FilterHeight * FilterWidth * FilterDepth * InputChannelsCount dimensions
	// (or InputChannelsCount * FilterHeight * FilterWidth * FilterDepth * FilterCount dimensions for transposed filters)
	// May be null if the filter has not been initialized (or must be reset)
	virtual CPtr<CDnnBlob> GetFilterData() const;
	virtual void SetFilterData(const CPtr<CDnnBlob>& newFilter);

	// Returns true for transposed filters
	virtual bool IsFilterTransposed() const { return false; }

	// Indicates if the free term should be set to zero ("no bias")
	bool IsZeroFreeTerm() const { return isZeroFreeTerm; }
	void SetZeroFreeTerm(bool _isZeroFreeTerm) { isZeroFreeTerm  = _isZeroFreeTerm; }

	// Returns or sets the free term (the blobs are copied) 
	// The blob should be of FilterCount size
	// May be null if the free term has not been initialized (or must be reset)
	virtual CPtr<CDnnBlob> GetFreeTermData() const;
	virtual void SetFreeTermData(const CPtr<CDnnBlob>& newFreeTerms);

	// Applies the batch normalization parameters to the internal parameters of the layer
	// The layer will then return the same output 
	// that was previously returned by the combination of this layer with batch normalization
	void ApplyBatchNormalization(CBatchNormalizationLayer& batchNorm);

protected:
	CBaseConvLayer( IMathEngine& mathEngine, const char* name );
	~CBaseConvLayer();

	int filterHeight;			// filter height
	int filterWidth;			// filter width
	int strideHeight;			// vertical filter stride
	int strideWidth;			// horizontal filter stride
	int filterCount;			// the number of filters per channel
	int paddingHeight;			// vertical padding size
	int paddingWidth;			// horizontal padding size
	int dilationHeight;			// vertical dilation
	int dilationWidth;			// horizontal dilation

	bool isZeroFreeTerm; // indicates if free term is set to zero

	// The filter; the pointer is valid only when the desired parameters are known: either set externally or are filled in on reshape
	CPtr<CDnnBlob>& Filter() { return paramBlobs[0]; }
	CPtr<CDnnBlob>& FreeTerms() { return paramBlobs[1]; }	// free terms matrix

	CPtr<CDnnBlob>& FilterDiff() { return paramDiffBlobs[0]; }
	CPtr<CDnnBlob>& FreeTermsDiff() { return paramDiffBlobs[1]; }

	const CPtr<CDnnBlob>& Filter() const { return paramBlobs[0]; }
	const CPtr<CDnnBlob>& FreeTerms() const { return paramBlobs[1]; }	// free terms matrix

	void FilterLayerParams( float threshold ) override;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////

// CConvLayer implements a convolution layer
class NEOML_API CConvLayer : public CBaseConvLayer {
	NEOML_DNN_LAYER( CConvLayer )
public:
	explicit CConvLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

protected:
	virtual ~CConvLayer();

	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;
	void LearnOnce() override;

private:
	CConvolutionDesc* convDesc; // the convolution descriptor

	void calcOutputBlobSize(int& outputHeight, int& outputWidth) const;
	void initConvDesc();
	void destroyConvDesc();
};

///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
// CRleConvLayer implements a layer that performs convolution on an image in RLE raster format
// The input blob instead of float data contains CRleStroke objects
// CDnnRleImageSentinel is the end-of-row signal
// The images in the blob are organized by batches: in one CRleImage object the data is separated into rows by the end-of-row signals instead of being put into different rows in blob dimensions (as it is done in Image::CRleStroke)
// The Channels dimension should be 1
class NEOML_API CRleConvLayer : public CBaseConvLayer {
	NEOML_DNN_LAYER( CRleConvLayer )
public:
	explicit CRleConvLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	void SetFilterData( const CPtr<CDnnBlob>& newFilter ) override;
	void SetFreeTermData( const CPtr<CDnnBlob>& newFreeTerm ) override;

	// The pixel value for strokes
	float GetStrokeValue() const { return strokeValue; }
	void SetStrokeValue(float _strokeValue) { strokeValue = _strokeValue; }

	// The pixel value for non-strokes
	float GetNonStrokeValue() const { return nonStrokeValue; }
	void SetNonStrokeValue(float _nonStrokeValue) { nonStrokeValue = _nonStrokeValue; }

protected:
	virtual ~CRleConvLayer();

	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;
	void LearnOnce() override;

private:
	float strokeValue;		// the pixel value for strokes
	float nonStrokeValue;	// the pixel value for strokes
	CRleConvolutionDesc* convDesc; // RLE convolution descriptor

	void calcOutputBlobSize(int& outputHeight, int& outputWidth) const;
	void initConvDesc();
	void destroyConvDesc();
};

} // namespace NeoML
