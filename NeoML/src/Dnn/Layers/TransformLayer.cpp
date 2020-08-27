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

#include <NeoML/Dnn/Layers/TransformLayer.h>

namespace NeoML {

CTransformLayer::CDimensionRule::CDimensionRule() :
	Operation( O_Multiply ),
	Parameter(1)
{
}

CTransformLayer::CDimensionRule::CDimensionRule( TOperation op, int param ) :
	Operation( op ),
	Parameter( param )
{
	NeoAssert( Operation == O_Remainder || param > 0 );
}

bool CTransformLayer::CDimensionRule::operator==( const CDimensionRule& other ) const
{
	return Operation == other.Operation && Parameter == other.Parameter;
}

// Applies the transformation
int CTransformLayer::CDimensionRule::Transform( int input ) const
{
	switch( Operation ) {
		case O_Remainder:
			return 1;
		case O_SetSize:
			return Parameter;
		case O_Multiply:
			return input * Parameter;
			break;
		case O_Divide:
			NeoAssert( input % Parameter == 0 );
			return input / Parameter;
			break;
		default:
			NeoAssert( false );
	}
	return NotFound;
}

/////////////////////////////////////////////////////////////////////////////////////////

CTransformLayer::CTransformLayer( IMathEngine& mathEngine ) :
	CBaseInPlaceLayer( mathEngine, "CCnnTransformLayer", false )
{
}

CTransformLayer::~CTransformLayer()
{
}

void CTransformLayer::SetDimensionRule( TBlobDim dim, const CDimensionRule& rule )
{
	if( rules[dim] == rule ) {
		return;
	}
	rules[dim] = rule;
	ForceReshape();
}

void CTransformLayer::SetDimensionRule( TBlobDim dim, TOperation op, int param )
{
	CDimensionRule newRule( op, param );
	if( rules[dim] == newRule ) {
		return;
	}
	rules[dim] = newRule;
	ForceReshape();
}

void CTransformLayer::OnReshaped()
{
	CheckInput1();

	CheckArchitecture( !GetDnn()->IsRecurrentMode(), GetName(), "can't be used inside of recurrent layers" );
	CheckArchitecture( inputDescs[0].GetDataType() == CT_Float || !IsBackwardPerformed(), GetName(),
		"Integer blobs can't be backpropagated" );

	outputDescs[0] = inputDescs[0];
	// The first pass: calculate everything except O_Remainder
	int remainder = inputDescs[0].BlobSize();
	TBlobDim remainderDim = TBlobDim(-1);
	for( TBlobDim d = TBlobDim(0); d < BD_Count; ++d ) {
		if(rules[d].Operation == O_Remainder) {
			NeoAssert(remainderDim < 0);
			remainderDim = d;
		}
		int outputDimSize = rules[d].Transform(inputDescs[0].DimSize(d));
		outputDescs[0].SetDimSize(d, outputDimSize);
		NeoAssert(remainder % outputDimSize == 0);
		remainder /= outputDimSize;
	}
	// Set the remainder
	if(remainderDim >= 0) {
		outputDescs[0].SetDimSize(remainderDim, remainder);
	}
	NeoAssert(outputDescs[0].BlobSize() == inputDescs[0].BlobSize());

	inputDesc = inputDescs[0];
	outputDesc = outputDescs[0];
}

static const int TransformLayerVersion = 2001;

void CTransformLayer::Serialize( CArchive& archive )
{
	int version = archive.SerializeVersion( TransformLayerVersion, CDnn::ArchiveMinSupportedVersion );
	if( version > 2000 ) {
		CBaseInPlaceLayer::Serialize( archive );
	} else {
		CBaseLayer::Serialize( archive );
	}

	if( archive.IsStoring() ) {
		archive.WriteSmallValue( 0 );
	} else if( archive.IsLoading() ) {
		archive.ReadSmallValue();
	} else {
		NeoAssert( false );
	}

	for( int i = 0; i < BD_Count; i++ ) {
		CDimensionRule& rule = rules[i];
		archive.SerializeEnum( rule.Operation );
		archive.SerializeSmallValue( rule.Parameter );
	}
}

void CTransformLayer::RunOnce()
{
	if( inputBlobs[0]->GetDataType() == CT_Float && inputBlobs[0]->GetData() != outputBlobs[0]->GetData() ) {
		MathEngine().VectorCopy( outputBlobs[0]->GetData(), inputBlobs[0]->GetData(), outputBlobs[0]->GetDataSize() );
	} else if( inputBlobs[0]->GetDataType() == CT_Int && inputBlobs[0]->GetData<int>() != outputBlobs[0]->GetData<int>() ) {
		MathEngine().VectorCopy( outputBlobs[0]->GetData<int>(), inputBlobs[0]->GetData<int>(), outputBlobs[0]->GetDataSize() );
	} else {
		outputBlobs[0]->ReinterpretDimensions( outputDesc );
	}
}

void CTransformLayer::BackwardOnce()
{
	if( inputDiffBlobs[0]->GetData() != outputDiffBlobs[0]->GetData() ) {
		MathEngine().VectorCopy( inputDiffBlobs[0]->GetData(), outputDiffBlobs[0]->GetData(),
			inputDiffBlobs[0]->GetDataSize() );
	} else {
		inputDiffBlobs[0]->ReinterpretDimensions( inputDesc );
	}
}


} // namespace NeoML
