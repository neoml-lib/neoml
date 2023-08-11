/* Copyright © 2017-2023 ABBYY

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

#include <NeoML/Dnn/Layers/GrnLayer.h>

namespace NeoML {

CGrnLayer::CGrnLayer( IMathEngine& mathEngine ) :
	CBaseLayer( mathEngine, "CGrnLayer", false ),
	epsilon( mathEngine ),
	invChannels( mathEngine ),
	one( mathEngine )
{
	paramBlobs.SetSize( PN_Count );
	epsilon.SetValue( 1e-6f ); // default value from the article
	one.SetValue( 1.f );
}

void CGrnLayer::SetEpsilon( float newEpsilon )
{
	NeoAssert( newEpsilon > 0 );
	epsilon.SetValue( newEpsilon );
}

void CGrnLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( 0 );
	CBaseLayer::Serialize( archive );

	float epsilonValue = archive.IsStoring() ? GetEpsilon() : 0.f;

	archive.Serialize( epsilonValue );
	
	if( archive.IsLoading() ) {
		SetEpsilon( epsilonValue );
	}
}

void CGrnLayer::Reshape()
{
	CheckLayerArchitecture( GetInputCount() == 1, "layer must have exactly 1 input" );
	CheckLayerArchitecture( GetOutputCount() == 1, "layer must have exactly 1 output" );

	CBlobDesc paramDesc;
	paramDesc.SetDimSize( BD_Channels, inputDescs[0].Channels() );
	if( scale() == nullptr || scale()->GetDataSize() != paramDesc.BlobSize() ) {
		scale() = CDnnBlob::CreateBlob( MathEngine(), CT_Float, paramDesc );
		scale()->Fill( 1.f );
	}
	if( bias() == nullptr || bias()->GetDataSize() != paramDesc.BlobSize() ) {
		bias() = CDnnBlob::CreateBlob( MathEngine(), CT_Float, paramDesc );
		bias()->Clear();
	}

	invChannels.SetValue( 1.f / inputDescs[0].Channels() );

	inputDescs.CopyTo( outputDescs );
}

void CGrnLayer::RunOnce()
{
	NeoAssert( inputBlobs[0] != outputBlobs[0] );

	const int objectCount = inputBlobs[0]->GetObjectCount();
	const int geometry = inputBlobs[0]->GetGeometricalSize();
	const int channels = inputBlobs[0]->GetChannelsCount();
	const int objectSize = geometry * channels;

	CConstFloatHandle inputData = inputBlobs[0]->GetData();
	CFloatHandle outputData = outputBlobs[0]->GetData();

	CFloatHandleStackVar buff( MathEngine(), objectCount + objectCount * channels );
	CFloatHandle batchBuff = buff.GetHandle();
	CFloatHandle batchChannelBuff = batchBuff + objectCount;

	// Calculate L2
	MathEngine().VectorEltwiseMultiply( inputData, inputData, outputData, objectCount * objectSize );
	MathEngine().SumMatrixRows( objectCount, batchChannelBuff, outputData, geometry, channels );
	MathEngine().VectorSqrt( batchChannelBuff, batchChannelBuff, objectCount * channels );
	
	// Calculate average
	if( objectCount > 1 ) {
		MathEngine().SumMatrixColumns( batchBuff, batchChannelBuff, objectCount, channels );
	} else {
		MathEngine().VectorSum( batchChannelBuff, channels, batchBuff );
	}
	MathEngine().VectorMultiply( batchBuff, batchBuff, objectCount, invChannels );
	MathEngine().VectorAddValue( batchBuff, batchBuff, objectCount, epsilon );
	MathEngine().VectorInv( batchBuff, batchBuff, objectCount );

	if( objectCount > 1 ) {
		MathEngine().MultiplyDiagMatrixByMatrix( batchBuff, objectCount, batchChannelBuff, channels,
			batchChannelBuff, objectCount * channels );
		MathEngine().MultiplyMatrixByDiagMatrix( batchChannelBuff, objectCount, channels,
			scale()->GetData(), batchChannelBuff, objectCount * channels );
	} else {
		MathEngine().VectorMultiply( batchChannelBuff, batchChannelBuff, channels, batchBuff );
		MathEngine().VectorEltwiseMultiply( batchChannelBuff, scale()->GetData(), batchChannelBuff, channels );
	}
	MathEngine().VectorAddValue( batchChannelBuff, batchChannelBuff, objectCount * channels, one );

	// TODO: ineffective on GPU, add function?
	for( int b = 0; b < objectCount; ++b ) {
		MathEngine().MultiplyMatrixByDiagMatrix( inputData + b * objectSize, geometry, channels,
			batchChannelBuff + b * channels, outputData + b * objectSize, objectSize );
	}
	MathEngine().AddVectorToMatrixRows( 1, outputData, outputData, objectCount * geometry,
		channels, bias()->GetData() );
}

void CGrnLayer::setParam( TParamName name, const CPtr<CDnnBlob>& newValue )
{
	if( newValue == nullptr ) {
		NeoAssert( paramBlobs[name] == nullptr || GetDnn() == nullptr );
		paramBlobs[name] = nullptr;
	} else if( paramBlobs[name] != nullptr && GetDnn() != nullptr ) {
		NeoAssert( paramBlobs[name]->GetDataSize() == newValue->GetDataSize() );
		paramBlobs[name]->CopyFrom( newValue );
	} else {
		paramBlobs[name] = newValue->GetCopy();
	}
}

CPtr<CDnnBlob> CGrnLayer::getParam( TParamName name ) const
{
	if( paramBlobs[name] == nullptr ) {
		return nullptr;
	}
	return paramBlobs[name]->GetCopy();
}

} // namespace NeoML
