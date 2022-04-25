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

static inline int nontrivialInterpolationDims( const CArray<int>& scales )
{
	int result = 0;
	for( int i = 0; i < scales.Size(); ++i ) {
		result += scales[i] > 1 ? 1 : 0;
	}
	return result;
}

// --------------------------------------------------------------------------------------------------------------------

CInterpolationLayer::CInterpolationLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CInterpolationLayer", false )
{
	scales.Add( 1, static_cast<int>( BD_Count ) );
}

static const int InterpolationLayerVersion = 0;

void CInterpolationLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( InterpolationLayerVersion );
	CBaseLayer::Serialize( archive );
	archive.Serialize( scales );
}

void CInterpolationLayer::SetScale( TBlobDim dim, int scale )
{
	NeoAssert( dim >= BD_BatchLength && dim < BD_Count );
	NeoAssert( scale >= 1 );
	scales[dim] = scale;
}

int CInterpolationLayer::GetScale( TBlobDim dim ) const
{
	NeoAssert( dim >= BD_BatchLength && dim < BD_Count );
	return scales[dim];
}

void CInterpolationLayer::Reshape()
{
	CheckInput1();
	CheckOutputs();
	CheckArchitecture( GetInputCount() == 1, GetName(), "CInterpolationLayer must have 1 input" );
	CheckArchitecture( GetOutputCount() == 1, GetName(), "CInterpolationLayer must have 1 output" );
	CheckArchitecture( inputDescs[0].GetDataType() == CT_Float, GetName(), "CInterpolationLayer supports only float data" );

	outputDescs[0] = inputDescs[0];

	for( int dim = 0; dim < scales.Size(); ++dim ) {
		outputDescs[0].SetDimSize( dim, inputDescs[0].DimSize( dim ) * scales[dim] );
	}
}

void CInterpolationLayer::RunOnce()
{
	int nontrivialDims = nontrivialInterpolationDims( scales );
	if( nontrivialDims == 0 ) {
		outputBlobs[0]->CopyFrom( inputBlobs[0].Ptr() );
	} else if( nontrivialDims == 1 ) {
		int objectCount = 1;
		for( int i = 0; i < scales.Size(); ++i ) {
			if( scales[i] > 1 ) {
				const int scaledAxis = inputBlobs[0]->DimSize( i );
				const int objectSize = inputBlobs[0]->GetDataSize() / ( objectCount / scaledAxis );
				MathEngine().LinearInterpolation( inputBlobs[0]->GetData(), outputBlobs[0]->GetData(),
					objectCount, scaledAxis, objectSize, scales[i] );
				break;
			}
			objectCount *= inputBlobs[0]->DimSize( i );
		}
	} else {
		int objectCount = 1;
		int objectSize = inputBlobs[0]->GetDataSize();
		CFloatHandleStackVar buffer( MathEngine(), outputBlobs[0]->GetDataSize() );
		CConstFloatHandle currInput = inputBlobs[0]->GetData();
		CFloatHandle currOutput = nontrivialDims % 2 == 0 ? buffer.GetHandle() : outputBlobs[0]->GetData();
		for( int i = 0; i < scales.Size(); ++i ) {
			const int scaledAxis = inputBlobs[0]->DimSize( i );
			objectSize /= scaledAxis;
			if( scales[i] > 1 ) {
				--nontrivialDims;
				MathEngine().LinearInterpolation( currInput, currOutput, objectCount, scaledAxis, objectSize, scales[i] );
				currInput = currOutput;
				currOutput = nontrivialDims % 2 == 0 ? buffer.GetHandle() : outputBlobs[0]->GetData();
			}
			objectCount *= scaledAxis * scales[i];
		}
	}
}

void CInterpolationLayer::BackwardOnce()
{
	int nontrivialDims = nontrivialInterpolationDims( scales );
	if( nontrivialDims == 0 ) {
		inputDiffBlobs[0]->CopyFrom( outputDiffBlobs[0].Ptr() );
	} else if( nontrivialDims == 1 ) {
		int objectCount = 1;
		for( int i = 0; i < scales.Size(); ++i ) {
			if( scales[i] > 1 ) {
				const int scaledAxis = inputDiffBlobs[0]->DimSize( i );
				const int objectSize = inputDiffBlobs[0]->GetDataSize() / ( objectCount / scaledAxis );
				MathEngine().LinearInterpolationBackward( outputDiffBlobs[0]->GetData(), inputDiffBlobs[0]->GetData(),
					objectCount, scaledAxis, objectSize, scales[i] );
				break;
			}
			objectCount *= inputDiffBlobs[0]->DimSize( i );
		}
	} else {
		CFloatHandleStackVar buffer( MathEngine(), outputDiffBlobs[0]->GetDataSize() );
		CConstFloatHandle currOutputDiff = outputDiffBlobs[0]->GetData();
		CFloatHandle currInputDiff = nontrivialDims % 2 == 0 ? buffer.GetHandle() : inputDiffBlobs[0]->GetData();
		int objectSize = 1;
		for( int i = scales.Size() - 1; i >= 0; --i ) {
			const int scaledAxis = inputDiffBlobs[0]->DimSize( i );
			if( scales[i] > 1 ) {
				--nontrivialDims;
				MathEngine().LinearInterpolationBackward( currOutputDiff, currInputDiff,
					outputDiffBlobs[0]->GetDataSize() / ( scales[i] * scaledAxis * objectSize ), scaledAxis, objectSize, scales[i] );
				currOutputDiff = currInputDiff;
				currInputDiff = nontrivialDims % 2 == 0 ? buffer.GetHandle() : inputDiffBlobs[0]->GetData();
			}
			objectSize *= inputDiffBlobs[0]->DimSize( i );
		}
	}
}

} // namespace NeoML
