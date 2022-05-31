/* Copyright © 2017-2022 ABBYY Production LLC

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

#include <NeoML/Dnn/Layers/InterpolationLayer.h>

namespace NeoML {

static inline int nontrivialInterpolationDims( const CBlobDesc& input, const CBlobDesc& output )
{
	int result = 0;
	for( TBlobDim dim = BD_BatchLength; dim < BD_Count; ++dim ) {
		if( input.DimSize( dim ) != output.DimSize( dim ) ) {
			result++;
		}
	}
	return result;
}

// --------------------------------------------------------------------------------------------------------------------

static const int InterpolationLayerRuleVersion = 1;

void CInterpolationLayer::CRule::Serialize( CArchive& archive )
{
	archive.SerializeVersion( InterpolationLayerRuleVersion );
	int typeInt = static_cast<int>( Type );
	archive.Serialize( typeInt );
	Type = static_cast<TRuleType>( typeInt );
	switch( Type ) {
		case TRuleType::None:
			break;
		case TRuleType::Resize:
			archive.Serialize( NewSize );
			break;
		case TRuleType::Scale:
			archive.Serialize( ScaleCoeff );
			break;
		default:
			NeoAssert( false );
	}
}

// --------------------------------------------------------------------------------------------------------------------

CInterpolationLayer::CInterpolationLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CInterpolationLayer", false ),
	coords( TInterpolationCoords::Asymmetric ),
	round( TInterpolationRound::None )
{
	rules.SetSize( BD_Count );
}

static const int InterpolationLayerVersion = 1;

void CInterpolationLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( InterpolationLayerVersion );
	CBaseLayer::Serialize( archive );
	int coordsInt = static_cast<int>( coords );
	archive.Serialize( coordsInt );
	coords = static_cast<TInterpolationCoords>( coordsInt );
	int roundInt = static_cast<int>( round );
	archive.Serialize( roundInt );
	round = static_cast<TInterpolationRound>( roundInt );
	archive.Serialize( rules );
}

void CInterpolationLayer::Reshape()
{
	CheckInput1();
	CheckOutputs();
	CheckArchitecture( GetInputCount() == 1, GetName(), "CInterpolationLayer must have 1 input" );
	CheckArchitecture( GetOutputCount() == 1, GetName(), "CInterpolationLayer must have 1 output" );
	CheckArchitecture( inputDescs[0].GetDataType() == CT_Float, GetName(), "CInterpolationLayer supports only float data" );

	outputDescs[0] = inputDescs[0];

	for( int dim = 0; dim < rules.Size(); ++dim ) {
		switch( rules[dim].Type ) {
			case TRuleType::None:
				break;
			case TRuleType::Resize:
				outputDescs[0].SetDimSize( dim, rules[dim].NewSize );
				break;
			case TRuleType::Scale:
				outputDescs[0].SetDimSize( dim, static_cast<int>( outputDescs[0].DimSize( dim ) * rules[dim].ScaleCoeff ) );
				break;
			default:
				NeoAssert( false );
		}
		CheckArchitecture( outputDescs[0].DimSize( dim ) > 0, GetName(), "Zero or negative dim size" );
	}
}

void CInterpolationLayer::RunOnce()
{
	int nontrivialDims = nontrivialInterpolationDims( inputBlobs[0]->GetDesc(), outputBlobs[0]->GetDesc() );
	if( nontrivialDims == 0 ) {
		outputBlobs[0]->CopyFrom( inputBlobs[0].Ptr() );
	} else if( nontrivialDims == 1 ) {
		int objectCount = 1;
		for( int i = 0; i < static_cast<int>( BD_Count ); ++i ) {
			const int oldSize = inputBlobs[0]->DimSize( i );
			const int newSize = outputBlobs[0]->DimSize( i );
			if( oldSize != newSize ) {
				const float scale = rules[i].Type == TRuleType::Scale ? rules[i].ScaleCoeff
					: static_cast<float>( newSize ) / oldSize;
				const int objectSize = inputBlobs[0]->GetDataSize() / ( objectCount * oldSize );
				MathEngine().LinearInterpolation( inputBlobs[0]->GetData(), outputBlobs[0]->GetData(), coords,
					round, objectCount, oldSize, objectSize, scale );
				break;
			}
			objectCount *= oldSize;
		}
	} else {
		int objectCount = 1;
		int objectSize = inputBlobs[0]->GetDataSize();
		const int halfBuffer = max( outputBlobs[0]->GetDataSize(), inputBlobs[0]->GetDataSize() );
		CFloatHandleStackVar buffer( MathEngine(), 2 * halfBuffer );
		CConstFloatHandle currInput = inputBlobs[0]->GetData();
		CFloatHandle currOutput = nontrivialDims % 2 == 0 ? buffer.GetHandle() : buffer.GetHandle() + halfBuffer;
		for( int i = 0; i < static_cast<int>( BD_Count ); ++i ) {
			const int oldSize = inputBlobs[0]->DimSize( i );
			const int newSize = outputBlobs[0]->DimSize( i );
			objectSize /= oldSize;
			if( oldSize != newSize ) {
				--nontrivialDims;
				const float scale = rules[i].Type == TRuleType::Scale ? rules[i].ScaleCoeff
					: static_cast<float>( newSize ) / oldSize;
				MathEngine().LinearInterpolation( currInput, currOutput, coords, round,
					objectCount, oldSize, objectSize, scale );
				currInput = currOutput;
				currOutput = nontrivialDims % 2 == 0 ? buffer.GetHandle() : buffer.GetHandle() + halfBuffer;
			}
			objectCount *= newSize;
		}
		MathEngine().VectorCopy( outputBlobs[0]->GetData(), currInput, outputBlobs[0]->GetDataSize() );
	}
}

void CInterpolationLayer::BackwardOnce()
{
	NeoAssert( false );
}

} // namespace NeoML
