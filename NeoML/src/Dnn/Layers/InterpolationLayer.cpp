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

#include <memory>

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
	round( TInterpolationRound::None ),
	nontrivialDims( 0 ),
	upsamplingHeightCopyCount( 0 ),
	upsamplingWidthCopyCount( 0 )
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
	CheckOutputs();
	CheckLayerArchitecture( GetOutputCount() == 1, "CInterpolationLayer must have 1 output" );
	CheckLayerArchitecture( inputDescs[0].GetDataType() == CT_Float, "CInterpolationLayer supports only float data" );

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
		CheckLayerArchitecture( outputDescs[0].DimSize( dim ) > 0, "Zero or negative dim size" );
	}

	nontrivialDims = nontrivialInterpolationDims( inputDescs[0], outputDescs[0] );
	upsamplingHeightCopyCount = 0; // Set as "not using Upsampling2DForward"

	if( coords == TInterpolationCoords::Asymmetric && round == TInterpolationRound::Floor
		&& nontrivialDims > 0 && nontrivialDims < 3 )
	{
		tryUpsampling();
	}
}

void CInterpolationLayer::RunOnce()
{
	if( nontrivialDims == 0 ) {
		outputBlobs[0]->CopyFrom( inputBlobs[0].Ptr() );
	} else if( upsamplingHeightCopyCount > 0 ) {
		MathEngine().Upsampling2DForward( upsamplingInputDesc, inputBlobs[0]->GetData(), upsamplingHeightCopyCount,
			upsamplingWidthCopyCount, upsamplingOutputDesc, outputBlobs[0]->GetData() );
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
	} else if( nontrivialDims == 2 ) {
		int objectCount = 1;
		int objectSize = inputBlobs[0]->GetDataSize();
		std::unique_ptr<CFloatHandleStackVar> buff;
		for( int i = 0; i < static_cast<int>( BD_Count ); ++i ) {
			const int oldSize = inputBlobs[0]->DimSize( i );
			const int newSize = outputBlobs[0]->DimSize( i );
			objectSize /= oldSize;
			if( oldSize != newSize ) {
				const float scale = rules[i].Type == TRuleType::Scale ? rules[i].ScaleCoeff
					: static_cast<float>( newSize ) / oldSize;
				if( buff == nullptr ) {
					// First of 2 interpolations: interpolate from inputBlobs[0] to intermediate buffer
					buff.reset( new CFloatHandleStackVar( MathEngine(), inputBlobs[0]->GetDataSize() / oldSize * newSize ) );
					MathEngine().LinearInterpolation( inputBlobs[0]->GetData(), buff->GetHandle(), coords, round,
						objectCount, oldSize, objectSize, scale );
				} else {
					// Second of 2 interpolations: interpolate from intermediate buffer to outputBlobs[0]
					MathEngine().LinearInterpolation( buff->GetHandle(), outputBlobs[0]->GetData(), coords, round,
						objectCount, oldSize, objectSize, scale );
					break;
				}
			}
			objectCount *= newSize;
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
				MathEngine().LinearInterpolation( currInput,
					nontrivialDims != 0 ? currOutput : outputBlobs[0]->GetData(),
					coords, round, objectCount, oldSize, objectSize, scale );
				currInput = currOutput;
				currOutput = nontrivialDims % 2 == 0 ? buffer.GetHandle() : buffer.GetHandle() + halfBuffer;
			}
			objectCount *= newSize;
		}
	}
}

void CInterpolationLayer::BackwardOnce()
{
	NeoAssert( false );
}

// Tries to interpret this layer as Upsampling2D
void CInterpolationLayer::tryUpsampling()
{
	CFastArray<TBlobDim, 2> usedDims;
	for( TBlobDim dim = BD_BatchLength; dim != BD_Count; ++dim ) {
		if( outputDescs[0].DimSize( dim ) % inputDescs[0].DimSize( dim ) != 0 ) {
			// It can't be emulated by Upsampling2d
			return;
		}
		if( outputDescs[0].DimSize( dim ) != inputDescs[0].DimSize( dim ) ) {
			usedDims.Add( dim );
		}
	}

	NeoPresume( usedDims.Size() == 1 || usedDims.Size() == 2 );
	if( usedDims.Size() == 2 ) {
		// Upsampling2D is possible only if there's no data between upsampled dims
		for( TBlobDim dim = usedDims[0] + 1; dim != usedDims[1]; ++dim ) {
			if( outputDescs[0].DimSize( dim ) != 1 ) {
				return;
			}
		}
	}

	if( usedDims.Size() == 2 ) {
		upsamplingHeightCopyCount = outputDescs[0].DimSize( usedDims[0] ) / inputDescs[0].DimSize( usedDims[0] );
		upsamplingWidthCopyCount = outputDescs[0].DimSize( usedDims[1] ) / inputDescs[0].DimSize( usedDims[1] );
	} else {
		upsamplingHeightCopyCount = 1;
		upsamplingWidthCopyCount = outputDescs[0].DimSize( usedDims[0] ) / inputDescs[0].DimSize( usedDims[0] );
	}

	int objectCount = 1;
	for( TBlobDim dim = BD_BatchLength; dim != usedDims[0]; ++dim ) {
		objectCount *= inputDescs[0].DimSize( dim );
	}

	int pixelSize = 1;
	for( TBlobDim dim = usedDims.Last() + 1; dim != BD_Count; ++dim ) {
		pixelSize *= inputDescs[0].DimSize( dim );
	}

	upsamplingInputDesc = CBlobDesc( CT_Float );
	upsamplingInputDesc.SetDimSize( BD_BatchWidth, objectCount );
	upsamplingInputDesc.SetDimSize( BD_Height, usedDims.Size() == 1 ? 1 : inputDescs[0].DimSize( usedDims[0] ) );
	upsamplingInputDesc.SetDimSize( BD_Width,  inputDescs[0].DimSize( usedDims.Last() ) );
	upsamplingInputDesc.SetDimSize( BD_Channels, pixelSize );
	PRESUME_EXPR( upsamplingInputDesc.BlobSize() == inputDescs[0].BlobSize() );

	upsamplingOutputDesc = upsamplingInputDesc;
	upsamplingOutputDesc.SetDimSize( BD_Height, upsamplingOutputDesc.Height() * upsamplingHeightCopyCount );
	upsamplingOutputDesc.SetDimSize( BD_Width, upsamplingOutputDesc.Width() * upsamplingWidthCopyCount );
	PRESUME_EXPR( upsamplingOutputDesc.BlobSize() == outputDescs[0].BlobSize() );
}

} // namespace NeoML
