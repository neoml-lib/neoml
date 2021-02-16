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

#include <NeoML/Dnn/Layers/3dPoolingLayer.h>
#include <NeoMathEngine/NeoMathEngine.h>
#include <float.h>

namespace NeoML {

C3dPoolingLayer::C3dPoolingLayer( IMathEngine& mathEngine, const char* name ) :
	CBaseLayer( mathEngine, name, false )
{
	filterHeight = 1;
	filterWidth = 1;
	filterDepth = 1;
	strideHeight = 1;
	strideWidth = 1;
	strideDepth = 1;
}

void C3dPoolingLayer::SetFilterHeight(int _filterHeight)
{
	NeoAssert(GetDnn() == 0);
	filterHeight = _filterHeight;
}

void C3dPoolingLayer::SetFilterWidth(int _filterWidth)
{
	NeoAssert(GetDnn() == 0);
	filterWidth = _filterWidth;
}

void C3dPoolingLayer::SetFilterDepth(int _filterDepth)
{
	NeoAssert(GetDnn() == 0);
	filterDepth = _filterDepth;
}

void C3dPoolingLayer::SetStrideHeight(int _strideHeight)
{
	NeoAssert(GetDnn() == 0);
	strideHeight = _strideHeight;
}

void C3dPoolingLayer::SetStrideWidth(int _strideWidth)
{
	NeoAssert(GetDnn() == 0);
	strideWidth = _strideWidth;
}

void C3dPoolingLayer::SetStrideDepth(int _strideDepth)
{
	NeoAssert(GetDnn() == 0);
	strideDepth = _strideDepth;
}

void C3dPoolingLayer::Reshape()
{
	CheckInputs();
	CheckOutputs();
	CheckArchitecture( GetInputCount() == 1, GetName(), "pooling with multiple inputs" );
	CheckArchitecture( GetOutputCount() == 1, GetName(), "pooling with multiple outputs" );

	outputDescs[0] = inputDescs[0];
	outputDescs[0].SetDimSize(BD_Height, (inputDescs[0].Height() - filterHeight) / strideHeight + 1);
	outputDescs[0].SetDimSize(BD_Width, (inputDescs[0].Width() - filterWidth) / strideWidth + 1);
	outputDescs[0].SetDimSize(BD_Depth, (inputDescs[0].Depth() - filterDepth) / strideDepth + 1);
}

static const int Pooling3dLayerVersion = 2000;

void C3dPoolingLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( Pooling3dLayerVersion, CDnn::ArchiveMinSupportedVersion );
	CBaseLayer::Serialize( archive );

	archive.Serialize( filterHeight );
	archive.Serialize( filterWidth );
	archive.Serialize( filterDepth );
	archive.Serialize( strideHeight );
	archive.Serialize( strideWidth );
	archive.Serialize( strideDepth );
}

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

static const int MaxPooling3dLayerVersion = 2000;

void C3dMaxPoolingLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( MaxPooling3dLayerVersion, CDnn::ArchiveMinSupportedVersion );
	C3dPoolingLayer::Serialize( archive );
}

void C3dMaxPoolingLayer::Reshape()
{
	C3dPoolingLayer::Reshape();
	indexBlob = 0;
	if(IsBackwardPerformed()) {
		indexBlob = CDnnBlob::CreateBlob( MathEngine(), CT_Int, outputDescs[0] );
		RegisterRuntimeBlob(indexBlob);
	}
	destroyDesc();
}

void C3dMaxPoolingLayer::RunOnce()
{
	initDesc();

	CIntHandle indexBlobData;
	if( indexBlob != 0 ) {
		indexBlobData = indexBlob->GetData<int>();
	}

	MathEngine().Blob3dMaxPooling( *desc, inputBlobs[0]->GetData(),
		indexBlob != 0 ? &indexBlobData : 0, outputBlobs[0]->GetData() );
}

void C3dMaxPoolingLayer::BackwardOnce()
{
	initDesc();
	MathEngine().Blob3dMaxPoolingBackward( *desc, outputDiffBlobs[0]->GetData(),
		indexBlob->GetData<int>(), inputDiffBlobs[0]->GetData() );
}

void C3dMaxPoolingLayer::initDesc()
{
	if( desc == 0 ) {
		desc = MathEngine().Init3dMaxPooling( inputBlobs[0]->GetDesc(),
			filterHeight, filterWidth, filterDepth, strideHeight, strideWidth, strideDepth,
			outputBlobs[0]->GetDesc() );
	}
}

void C3dMaxPoolingLayer::destroyDesc()
{
	if( desc != 0 ) {
		delete desc;
		desc = 0;
	}
}

CLayerWrapper<C3dMaxPoolingLayer> Pooling3dMax( int filterHeight, int filterWidth, int filterDepth,
	int strideHeight, int strideWidth, int strideDepth )
{
	return CLayerWrapper<C3dMaxPoolingLayer>( "Pooling3D", [=]( C3dMaxPoolingLayer* result ) {
		result->SetFilterHeight( filterHeight );
		result->SetFilterWidth( filterWidth );
		result->SetFilterDepth( filterDepth );
		result->SetStrideHeight( strideHeight );
		result->SetStrideWidth( strideWidth );
		result->SetStrideDepth( strideDepth );
	} );
}

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

static const int MeanPooling3dLayerVersion = 2000;

void C3dMeanPoolingLayer::Serialize( CArchive& archive )
{
	archive.SerializeVersion( MeanPooling3dLayerVersion, CDnn::ArchiveMinSupportedVersion );
	C3dPoolingLayer::Serialize( archive );
}

void C3dMeanPoolingLayer::Reshape()
{
	destroyDesc();
}

void C3dMeanPoolingLayer::RunOnce()
{
	initDesc();
	MathEngine().Blob3dMeanPooling( *desc, inputBlobs[0]->GetData(),
		outputBlobs[0]->GetData() );
}

void C3dMeanPoolingLayer::BackwardOnce()
{
	initDesc();
	MathEngine().Blob3dMeanPoolingBackward( *desc, outputDiffBlobs[0]->GetData(),
		inputDiffBlobs[0]->GetData() );
}

void C3dMeanPoolingLayer::initDesc()
{
	if( desc == 0 ) {
		desc = MathEngine().Init3dMeanPooling( inputBlobs[0]->GetDesc(),
			filterHeight, filterWidth, filterDepth, strideHeight, strideWidth, strideDepth,
			outputBlobs[0]->GetDesc() );
	}
}

void C3dMeanPoolingLayer::destroyDesc()
{
	if( desc != 0 ) {
		delete desc;
		desc = 0;
	}
}

CLayerWrapper<C3dMeanPoolingLayer> Pooling3dMean( int filterHeight, int filterWidth, int filterDepth,
	int strideHeight, int strideWidth, int strideDepth )
{
	return CLayerWrapper<C3dMeanPoolingLayer>( "Pooling3dMean", [=]( C3dMeanPoolingLayer* result ) {
		result->SetFilterHeight( filterHeight );
		result->SetFilterWidth( filterWidth );
		result->SetFilterDepth( filterDepth );
		result->SetStrideHeight( strideHeight );
		result->SetStrideWidth( strideWidth );
		result->SetStrideDepth( strideDepth );
	} );
}

} // namespace NeoML
