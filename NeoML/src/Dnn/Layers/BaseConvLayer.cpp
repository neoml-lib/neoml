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

#include <NeoML/Dnn/Layers/ConvLayer.h>
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

CBaseConvLayer::CBaseConvLayer( IMathEngine& mathEngine, const char* name ) :
	CBaseLayer( mathEngine, name, true ),
	filterHeight( 1 ),
	filterWidth( 1 ),
	strideHeight( 1 ),
	strideWidth( 1 ),
	filterCount( 1 ),
	paddingHeight( 0 ),
	paddingWidth( 0 ),
	dilationHeight( 1 ),
	dilationWidth( 1 ),
	isZeroFreeTerm( false ),
	activation( AF_None )
{
	paramBlobs.SetSize(2);
}

CBaseConvLayer::~CBaseConvLayer()
{
}

void CBaseConvLayer::SetFilterHeight( int _filterHeight )
{
	filterHeight = _filterHeight;
	ForceReshape();
}

void CBaseConvLayer::SetFilterWidth( int _filterWidth )
{
	filterWidth = _filterWidth;
	ForceReshape();
}

void CBaseConvLayer::SetStrideHeight( int _strideHeight )
{
	strideHeight = _strideHeight;
	ForceReshape();
}

void CBaseConvLayer::SetStrideWidth( int _strideWidth )
{
	strideWidth = _strideWidth;
	ForceReshape();
}

void CBaseConvLayer::SetPaddingHeight( int _paddingHeight )
{
	paddingHeight = _paddingHeight;
	ForceReshape();
}

void CBaseConvLayer::SetPaddingWidth( int _paddingWidth )
{
	paddingWidth = _paddingWidth;
	ForceReshape();
}

void CBaseConvLayer::SetDilationHeight( int newDilationHeight )
{
	dilationHeight = newDilationHeight;
	ForceReshape();
}

void CBaseConvLayer::SetDilationWidth( int newDilationWidth )
{
	dilationWidth = newDilationWidth;
	ForceReshape();
}

void CBaseConvLayer::SetFilterCount( int _filterCount )
{
	filterCount = _filterCount;
	ForceReshape();
}

void CBaseConvLayer::SetActivation( const CActivationInfo& newActivation )
{
	if( activation.Type == newActivation.Type && activation.Param1 == newActivation.Param1
		&& activation.Param2 == newActivation.Param2 )
	{
		return;
	}

	activation = newActivation;
	ForceReshape();
}

CPtr<CDnnBlob> CBaseConvLayer::GetFilterData() const
{
	if( Filter() == 0 ) {
		return 0;
	}

	return Filter()->GetCopy();
}

void CBaseConvLayer::SetFilterData(const CPtr<CDnnBlob>& newFilter)
{
	if(newFilter == 0) {
		NeoAssert(Filter() == 0 || GetDnn() == 0);
		Filter() = 0;
	} else if(Filter() != 0 && GetDnn() != 0) {
		NeoAssert(Filter()->HasEqualDimensions(newFilter));
		Filter()->CopyFrom(newFilter);
	} else {
		Filter() = newFilter->GetCopy();
	}
}

CPtr<CDnnBlob> CBaseConvLayer::GetFreeTermData() const
{
	if(FreeTerms() == 0) {
		return 0;
	}

	return FreeTerms()->GetCopy();
}

void CBaseConvLayer::SetFreeTermData(const CPtr<CDnnBlob>& newFreeTerms)
{
	if(newFreeTerms == 0) {
		NeoAssert(FreeTerms() == 0 || GetDnn() == 0);
		FreeTerms() = 0;
	} else {
		if(FreeTerms() != 0 && GetDnn() != 0) {
			NeoAssert(FreeTerms()->GetDataSize() == newFreeTerms->GetDataSize());

			FreeTerms()->CopyFrom(newFreeTerms);
		} else {
			FreeTerms() = newFreeTerms->GetCopy();
		}
	}
}

void CBaseConvLayer::ApplyBatchNormalization(CBatchNormalizationLayer& batchNorm)
{
	CPtr<CDnnBlob> params = batchNorm.GetFinalParams();
	if(params.Ptr() == 0 || Filter().Ptr() == 0) {
		return;
	}
	NeoAssert(params->GetObjectSize() == filterCount);
	CConstFloatHandle gamma = params->GetObjectData( 0 );
	CConstFloatHandle beta = params->GetObjectData( 1 );

	// Because the inheriting classes may have different filter structure, 
	// use the external representation of the filter and free term
	CPtr<CDnnBlob> newFilter = GetFilterData();
	CPtr<CDnnBlob> newFreeTerm = GetFreeTermData();

	CFloatHandle filterData = newFilter->GetData();
	CFloatHandle freeTermData = newFreeTerm->GetData();

	MathEngine().VectorEltwiseMultiply(freeTermData, gamma, freeTermData, filterCount);
	MathEngine().VectorAdd(freeTermData, beta, freeTermData, filterCount);

	if(IsFilterTransposed()) {
		MathEngine().MultiplyMatrixByDiagMatrix(filterData,
			newFilter->GetGeometricalSize() * newFilter->GetBatchWidth(), filterCount, gamma,
			filterData, newFilter->GetDataSize());
	} else {
		MathEngine().MultiplyDiagMatrixByMatrix(gamma, filterCount, filterData, newFilter->GetObjectSize(),
			filterData, newFilter->GetDataSize());
	}

	SetFilterData(newFilter);
	SetFreeTermData(newFreeTerm);
}

void CBaseConvLayer::FilterLayerParams( float threshold )
{
	for( int blobIndex = 0; blobIndex < paramBlobs.Size(); ++blobIndex ) {
		if( paramBlobs[blobIndex] != 0 ) {
			MathEngine().FilterSmallValues( paramBlobs[blobIndex]->GetData(),
				paramBlobs[blobIndex]->GetDataSize(), threshold );
		}
	}
}

static const int BaseConvLayerVersion = 2001;

void CBaseConvLayer::Serialize( CArchive& archive )
{
	const int version = archive.SerializeVersion( BaseConvLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize( archive );

	archive.Serialize( filterHeight );
	archive.Serialize( filterWidth );
	archive.Serialize( strideHeight );
	archive.Serialize( strideWidth );
	archive.Serialize( filterCount );
	archive.Serialize( paddingHeight );
	archive.Serialize( paddingWidth );
	archive.Serialize( dilationHeight );
	archive.Serialize( dilationWidth );
	archive.Serialize( isZeroFreeTerm );

	if( archive.IsLoading() ) {
		// Convert the free terms blob into a new tensor with the length in the first dimension not Channels
		CDnnBlob* freeTerms = FreeTerms();
		if( freeTerms != 0 && freeTerms->DimSize(0) != freeTerms->GetDataSize() ) {
			NeoAssert( freeTerms->GetChannelsCount() == freeTerms->GetDataSize() );
			CBlobDesc desc( CT_Float );
			desc.SetDimSize( 0, freeTerms->GetDataSize() );
			freeTerms->ReinterpretDimensions( desc );
		}
	}

	if( version >= 2001 ) {
		int activationTypeInt = static_cast<int>( activation.Type );
		archive.Serialize( activationTypeInt );
		activation.Type = static_cast<TActivationFunction>( activationTypeInt );
		archive.Serialize( activation.Param1 );
		archive.Serialize( activation.Param2 );
	} else if( archive.IsLoading() ) {
		activation = AF_None;
	}
}

} // namespace NeoML
