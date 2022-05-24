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

#include <NeoML/Dnn/Layers/PoolingLayer.h>
#include <NeoMathEngine/NeoMathEngine.h>
#include <float.h>

namespace NeoML {

CPoolingLayer::CPoolingLayer( IMathEngine& mathEngine, const char* name ) :
	CBaseLayer( mathEngine, name, false )
{
	filterHeight = 1;
	filterWidth = 1;
	strideHeight = 1;
	strideWidth = 1;
}

void CPoolingLayer::SetFilterHeight( int _filterHeight )
{
	NeoAssert( _filterHeight > 0 );
	if( filterHeight == _filterHeight ) {
		return;
	}
	filterHeight = _filterHeight;
	ForceReshape();
}

void CPoolingLayer::SetFilterWidth( int _filterWidth )
{
	NeoAssert( _filterWidth > 0 );
	if( filterWidth == _filterWidth ) {
		return;
	}
	filterWidth = _filterWidth;
	ForceReshape();
}

void CPoolingLayer::SetStrideHeight( int _strideHeight )
{
	NeoAssert( _strideHeight > 0 );
	if( strideHeight == _strideHeight ) {
		return;
	}
	strideHeight = _strideHeight;
	ForceReshape();
}

void CPoolingLayer::SetStrideWidth( int _strideWidth )
{
	NeoAssert( _strideWidth > 0 );
	if( strideWidth == _strideWidth ) {
		return;
	}
	strideWidth = _strideWidth;
	ForceReshape();
}

void CPoolingLayer::Reshape()
{
	CheckInputs();
	CheckOutputs();
	CheckArchitecture( GetInputCount() == 1, GetName(), "pooling with multiple inputs" );
	CheckArchitecture( GetOutputCount() == 1, GetName(), "pooling with multiple outputs" );

	outputDescs[0] = inputDescs[0];
	outputDescs[0].SetDimSize( BD_Height, ( inputDescs[0].Height() - filterHeight ) / strideHeight + 1 );
	outputDescs[0].SetDimSize( BD_Width, ( inputDescs[0].Width() - filterWidth ) / strideWidth + 1 );
}

static const int PoolingLayerVersion = 2000;

void CPoolingLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( PoolingLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize( archive );

	archive.Serialize( filterHeight );
	archive.Serialize( filterWidth );
	archive.Serialize( strideHeight );
	archive.Serialize( strideWidth );

	if( archive.IsLoading() ) {
		ForceReshape();
	}
}

///////////////////////////////////////////////////////////////////////////////////
// CMeanPoolingLayer

void CMeanPoolingLayer::RunOnce()
{
	initDesc();

	MathEngine().BlobMeanPooling( *desc, inputBlobs[0]->GetData(),
		outputBlobs[0]->GetData() );
}

void CMeanPoolingLayer::BackwardOnce()
{
	initDesc();

	MathEngine().BlobMeanPoolingBackward( *desc, outputDiffBlobs[0]->GetData(),
		inputDiffBlobs[0]->GetData() );
}

void CMeanPoolingLayer::Reshape()
{
	CPoolingLayer::Reshape();
	destroyDesc();
}

void CMeanPoolingLayer::initDesc()
{
	if( desc == 0 ) {
		NeoPresume( inputBlobs[0] != nullptr || inputDiffBlobs[0] != nullptr );
		NeoPresume( outputBlobs[0] != nullptr || outputDiffBlobs[0] != nullptr );
		desc = MathEngine().InitMeanPooling( inputBlobs[0] != nullptr ? inputBlobs[0]->GetDesc() : inputDiffBlobs[0]->GetDesc(),
			filterHeight, filterWidth, strideHeight, strideWidth,
			outputBlobs[0] != nullptr ? outputBlobs[0]->GetDesc() : outputDiffBlobs[0]->GetDesc() );
	}
}

void CMeanPoolingLayer::destroyDesc()
{
	if( desc != 0 ) {
		delete desc;
		desc = 0;
	}
}

static const int MeanPoolingLayerVersion = 2000;

void CMeanPoolingLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( MeanPoolingLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CPoolingLayer::Serialize( archive );
}

///////////////////////////////////////////////////////////////////////////////////
// CMaxPoolingLayer

void CMaxPoolingLayer::Reshape()
{
	CPoolingLayer::Reshape();
	maxIndices = 0;
	if( IsBackwardPerformed() ) {
		maxIndices = CDnnBlob::CreateBlob( MathEngine(), CT_Int, outputDescs[0] );
		RegisterRuntimeBlob(maxIndices);
	}
	destroyDesc();
}

void CMaxPoolingLayer::RunOnce()
{
	initDesc();

	CIntHandle maxIndicesData;
	if( maxIndices != 0 ) {
		maxIndicesData = maxIndices->GetData<int>();
	}

	MathEngine().BlobMaxPooling( *desc, inputBlobs[0]->GetData(),
		maxIndices != 0 ? &maxIndicesData : 0, outputBlobs[0]->GetData() );
}

void CMaxPoolingLayer::BackwardOnce()
{
	initDesc();

	MathEngine().BlobMaxPoolingBackward( *desc, outputDiffBlobs[0]->GetData(),
		maxIndices->GetData<int>(), inputDiffBlobs[0]->GetData() );
}

void CMaxPoolingLayer::initDesc()
{
	if( desc == 0 ) {
		NeoPresume( inputBlobs[0] != nullptr || inputDiffBlobs[0] != nullptr );
		NeoPresume( outputBlobs[0] != nullptr || outputDiffBlobs[0] != nullptr );
		desc = MathEngine().InitMaxPooling( inputBlobs[0] != nullptr ? inputBlobs[0]->GetDesc() : inputDiffBlobs[0]->GetDesc(),
			filterHeight, filterWidth, strideHeight, strideWidth,
			outputBlobs[0] != nullptr ? outputBlobs[0]->GetDesc() : outputDiffBlobs[0]->GetDesc() );
	}
}

void CMaxPoolingLayer::destroyDesc()
{
	if( desc != 0 ) {
		delete desc;
		desc = 0;
	}
}

static const int MaxPoolingLayerVersion = 2000;

void CMaxPoolingLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( MaxPoolingLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CPoolingLayer::Serialize( archive );
}

CLayerWrapper<CMaxPoolingLayer> MaxPooling(
	int filterHeight, int filterWidth, int strideHeight, int strideWidth )
{
	return CLayerWrapper<CMaxPoolingLayer>( "MaxPooling", [=]( CMaxPoolingLayer* result ) {
		result->SetFilterHeight( filterHeight );
		result->SetFilterWidth( filterWidth );
		result->SetStrideHeight( strideHeight );
		result->SetStrideWidth( strideWidth );
	} );
}

CLayerWrapper<CMeanPoolingLayer> MeanPooling(
	int filterHeight, int filterWidth, int strideHeight, int strideWidth )
{
	return CLayerWrapper<CMeanPoolingLayer>( "MeanPooling", [=]( CMeanPoolingLayer* result ) {
		result->SetFilterHeight( filterHeight );
		result->SetFilterWidth( filterWidth );
		result->SetStrideHeight( strideHeight );
		result->SetStrideWidth( strideWidth );
	} );
}

} // namespace NeoML
