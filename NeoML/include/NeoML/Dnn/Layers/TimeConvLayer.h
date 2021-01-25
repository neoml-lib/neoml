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
#include <NeoML/Dnn/Dnn.h>

namespace NeoML {

// CTimeConvLayer performs one-dimensional convolution of sequences (over the BatchLength dimension)
class NEOML_API CTimeConvLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CTimeConvLayer )
public:
	explicit CTimeConvLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	// Layer parameters
	// The number of filters for convolution
	int GetFilterCount() const { return filterCount; }
	void SetFilterCount( int _filterCount );
	// The filter size
	int GetFilterSize() const { return filterSize; }
	void SetFilterSize( int _filterSize );
	// The filter stride
	int GetStride() const { return stride; }
	void SetStride( int _stride );
	// Padding at the beginning of BD_BatchLength
	void SetPaddingFront( int paddingFront );
	int GetPaddingFront() const { return paddingFront; }
	// Padding at the end of BD_BatchLength
	void SetPaddingBack( int paddingBack );
	int GetPaddingBack() const { return paddingBack; }
	// The filter dilation
	int GetDilation() const { return dilation; }
	void SetDilation( int _dilation );

	// Returns or sets the filter data (the blobs are copied)
	// A filter blob has the FilterCount * FilterSize * 1 * InputChannelsCount dimensions
	// May be null if the filter has not been initialized (or must be reset)
	virtual CPtr<CDnnBlob> GetFilterData() const;
	virtual void SetFilterData( const CPtr<CDnnBlob>& newFilter );

	// Returns or sets the free term (the blobs are copied)
	// A free term blob has the FilterCount size
	// May be null if the free term has not been initialized (or must be reset)
	virtual CPtr<CDnnBlob> GetFreeTermData() const;
	virtual void SetFreeTermData( const CPtr<CDnnBlob>& newFreeTerms );

	// DEPRECATED
	// Gets padding from both sides
	// assert if paddingFront != paddingBack
	int GetPadding() const;
	// Sets padding in both sides of BD_BatchLength
	void SetPadding( int _padding );
protected:
	virtual ~CTimeConvLayer() { destroyDesc(); }

	// CBaseLayer methods
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;
	void LearnOnce() override;
	void FilterLayerParams( float threshold ) override;
	void GetFreeTermParameterIndexes( CArray<int>& indexes ) const override;

private:
	CTimeConvolutionDesc* desc;
	// The number of filters
	int filterCount;
	// The filter size
	int filterSize;
	// The filter stride
	int stride;
	// padding
	int paddingFront; // padding at the beginning of BD_BatchLength
	int paddingBack; // padding at the end of BD_BatchLength
	// The filter dilation
	int dilation;

	void initDesc();
	void destroyDesc();

	// Auxiliary methods for easy access to the parameters
	CPtr<CDnnBlob>& filter() { return paramBlobs[0]; }
	CPtr<CDnnBlob>& freeTerms() { return paramBlobs[1]; }

	CPtr<CDnnBlob>& filterDiff() { return paramDiffBlobs[0]; }
	CPtr<CDnnBlob>& freeTermsDiff() { return paramDiffBlobs[1]; }

	const CPtr<CDnnBlob>& filter() const { return paramBlobs[0]; }
	const CPtr<CDnnBlob>& freeTerms() const { return paramBlobs[1]; }
};

//=================================================================================================

inline void CTimeConvLayer::SetFilterCount( int _filterCount )
{
	NeoAssert( _filterCount > 0 );
	if( filterCount != _filterCount ) {
		filterCount = _filterCount;
		ForceReshape();
	}
}

inline void CTimeConvLayer::SetFilterSize( int _filterSize )
{
	NeoAssert( _filterSize > 0 );
	if( filterSize != _filterSize ) {
		filterSize = _filterSize;
		ForceReshape();
	}
}

inline void CTimeConvLayer::SetStride( int _stride )
{
	NeoAssert( _stride > 0 );
	if( stride != _stride ) {
		stride = _stride;
		ForceReshape();
	}
}

inline void CTimeConvLayer::SetPaddingFront( int _paddingFront )
{
	NeoAssert( _paddingFront >= 0 );
	if( paddingFront != _paddingFront ) {
		paddingFront = _paddingFront;
		ForceReshape();
	}
}

inline void CTimeConvLayer::SetPaddingBack( int _paddingBack )
{
	NeoAssert( _paddingBack >= 0 );
	if( paddingBack != _paddingBack ) {
		paddingBack = _paddingBack;
		ForceReshape();
	}
}

inline int CTimeConvLayer::GetPadding() const
{
	NeoAssert( paddingFront == paddingBack );
	return paddingFront;
}

inline void CTimeConvLayer::SetPadding( int _padding )
{
	NeoAssert( _padding >= 0 );
	if( paddingFront != _padding || paddingBack != _padding ) {
		paddingFront = _padding;
		paddingBack = _padding;
		ForceReshape();
	}
}

inline void CTimeConvLayer::SetDilation( int _dilation )
{
	NeoAssert( _dilation > 0 );
	if( dilation != _dilation ) {
		dilation = _dilation;
		ForceReshape();
	}
}

NEOML_API CLayerWrapper<CTimeConvLayer> TimeConv( int filterCount,
	int filterSize, int padding = 0, int stride = 1, int dilation = 1 );

} // namespace NeoML
