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

#include <NeoML/TraditionalML/CommonCluster.h>
#include <NeoML/TraditionalML/Clustering.h>

namespace NeoML {

// Precision of a double value
const double Precision = 1e-15;

CCommonCluster::CCommonCluster( const CClusterCenter& _center, const CParams& _params ) :
	params( _params ),
	center( _center ),
	isCenterDirty( false ),
	sumWeight( 0 )
{
	sum.Add( 0.0, _center.Mean.Size() );
	sumSquare.Add( 0.0, _center.Mean.Size() );
}

CCommonCluster::CCommonCluster( const CCommonCluster& first, const CCommonCluster& second ) :
	params( first.params ),
	center( first.center ),
	isCenterDirty( false ),
	sumWeight( first.sumWeight + second.sumWeight )
{
	NeoAssert( first.sum.Size() == second.sum.Size() );
	NeoAssert( first.sumSquare.Size() == second.sumSquare.Size() );

	elements.Add( first.elements );
	elements.Add( second.elements );

	for( int i = 0; i < first.sum.Size(); i++ ) {
		sum.Add( first.sum[i] + second.sum[i] );
		sumSquare.Add( first.sumSquare[i] + second.sumSquare[i] );
	}

	RecalcCenter();
}

void CCommonCluster::Add( int dataIndex, const CSparseFloatVectorDesc& desc, double weight )
{
	NeoAssert( dataIndex >= 0 );

	elements.Add( dataIndex );

	sumWeight += weight;

	for( int i = 0; i < desc.Size; i++ ) {
		sum[desc.Indexes == nullptr ? i : desc.Indexes[i]] += desc.Values[i] * weight;
		sumSquare[desc.Indexes == nullptr ? i : desc.Indexes[i]] += desc.Values[i] * desc.Values[i] * weight;
	}

	isCenterDirty = true;
}

void CCommonCluster::Reset()
{
	elements.DeleteAll();

	sumWeight = 0;
	for( int i = 0; i < sum.Size(); i++ ) {
		sum[i] = 0;
		sumSquare[i] = 0;
	}

	isCenterDirty = true;
}

void CCommonCluster::RecalcCenter()
{
	for( int i = 0; i < center.Mean.Size(); i++ ) {
		center.Mean.SetAt( i, static_cast<float>( sum[i] / sumWeight ) );

		double variance = params.DefaultVariance;
		if( sumWeight >= params.MinElementCountForVariance ) {
			variance = ( sumSquare[i] / sumWeight ) - ( sum[i] * sum[i] / sumWeight / sumWeight );
		}
		if( variance < Precision ) {
			variance = Precision;
		}
		center.Disp.SetAt( i, static_cast<float>( variance ) );
	}

	center.Norm = DotProduct( center.Mean, center.Mean );
}

} // namespace NeoML
