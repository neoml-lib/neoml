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

#include "common.h"
#pragma hdrstop

#include "GlobalPoolNodeBase.h"
#include "GraphCache.h"

#include "onnx.pb.h"

namespace NeoOnnx {

CGlobalPoolNodeBase::CGlobalPoolNodeBase( int nodeIndex, const onnx::NodeProto& onnxNode, int opsetVersion ) :
	COpNode( nodeIndex, onnxNode, opsetVersion )
{}

static const int pool2dDims = ( 1 << static_cast<int>( BD_Height ) ) | ( 1 << static_cast<int>( BD_Width ) );
static const int pool3dDims = pool2dDims | ( 1 << static_cast<int>( BD_Depth ) );
static const int globalPoolDims = pool3dDims;

void CGlobalPoolNodeBase::AddPoolingLayer( TPoolingType poolingType, const CTensorDim& dimsToPool, const CTensorShape& inputShape,
	const CTensorDim& inputDim, CNeoMLLinkCache& neoMLLinks, CDnn& dnn )
{
	// Blob dimensions which must be pooled
	int requiredPoolDims = 0;
	for( int dimIndex = 0; dimIndex < dimsToPool.Size(); ++dimIndex ) {
		requiredPoolDims |= ( 1 << static_cast<int>( dimsToPool[dimIndex] ) );
	}

	// Blob dimensions which can be pooled (required or inputShape[dim] == 1)
	int possiblePoolDims = requiredPoolDims;
	for( int dimIndex = 0; dimIndex < inputDim.Size(); ++dimIndex ) {
		if( inputShape[dimIndex] == 1 ) {
			possiblePoolDims |= ( 1 << static_cast<int>( inputDim[dimIndex] ) );
		}
	}

	// Use global pooling layer if possible
	if( ( globalPoolDims & requiredPoolDims ) == requiredPoolDims && ( globalPoolDims & possiblePoolDims ) == globalPoolDims ) {
		addGlobalPoolingLayer( poolingType, neoMLLinks, dnn );
	} else if( ( pool2dDims & requiredPoolDims ) == requiredPoolDims ) {
		add2dPoolingLayer( poolingType, inputShape, inputDim, requiredPoolDims, neoMLLinks, dnn );
	} else if( ( pool3dDims & requiredPoolDims ) == requiredPoolDims ) {
		add3dPoolingLayer( poolingType, inputShape, inputDim, requiredPoolDims, neoMLLinks, dnn );
	} else {
		CheckNeoOnnxSupport( false, "Pool dimensions are not supported", OnnxNode );
	}
}

void CGlobalPoolNodeBase::addGlobalPoolingLayer( TPoolingType poolingType, CNeoMLLinkCache& neoMLLinks, CDnn& dnn )
{
	CPtr<CBaseLayer> poolingLayer;
	static_assert( PT_Count == 2, "PT_Count != 2" );
	switch( poolingType ) {
		case PT_Max:
			poolingLayer = new CGlobalMaxPoolingLayer( dnn.GetMathEngine() );
			break;
		case PT_Mean:
			poolingLayer = new CGlobalMeanPoolingLayer( dnn.GetMathEngine() );
			break;
		default:
			CheckNeoOnnxInternal( false, "unknown pool type", OnnxNode );
	}

	poolingLayer->SetName( Name );
	poolingLayer->Connect( 0, *neoMLLinks[Input[0]].Layer, neoMLLinks[Input[0]].OutputIndex );
	dnn.AddLayer( *poolingLayer );
	neoMLLinks[Output[0]] = CNeoMLLink( poolingLayer, 0 );
}

void CGlobalPoolNodeBase::add2dPoolingLayer( TPoolingType poolingType, const CTensorShape& inputShape, const CTensorDim& inputDim,
	int pooledDims, CNeoMLLinkCache& neoMLLinks, CDnn& dnn )
{
	CPtr<CPoolingLayer> poolingLayer;
	static_assert( PT_Count == 2, "PT_Count != 2" );
	switch( poolingType ) {
		case PT_Max:
			poolingLayer = new CMaxPoolingLayer( dnn.GetMathEngine() );
			break;
		case PT_Mean:
			poolingLayer = new CMeanPoolingLayer( dnn.GetMathEngine() );
			break;
		default:
			CheckNeoOnnxInternal( false, "unknown pool type", OnnxNode );
	}

	poolingLayer->SetName( Name );

	// Make it global
	for( int dimIndex = 0; dimIndex < inputDim.Size(); ++dimIndex ) {
		const bool isDimPooled = ( ( ( 1 << static_cast<int>( inputDim[dimIndex] ) ) & pooledDims ) != 0 );
		switch( inputDim[dimIndex] ) {
			case BD_Height:
				poolingLayer->SetFilterHeight( isDimPooled ? inputShape[dimIndex] : 1 );
				poolingLayer->SetStrideHeight( 1 );
				break;
			case BD_Width:
				poolingLayer->SetFilterWidth( isDimPooled ? inputShape[dimIndex] : 1 );
				poolingLayer->SetStrideWidth( 1 );
				break;
			default:
				CheckNeoOnnxInternal( !isDimPooled, "wrong pooling dimension", OnnxNode );
		}
	}

	poolingLayer->Connect( 0, *neoMLLinks[Input[0]].Layer, neoMLLinks[Input[0]].OutputIndex );
	dnn.AddLayer( *poolingLayer );
	neoMLLinks[Output[0]] = CNeoMLLink( poolingLayer, 0 );
}

void CGlobalPoolNodeBase::add3dPoolingLayer( TPoolingType poolingType, const CTensorShape& inputShape, const CTensorDim& inputDim,
	int pooledDims, CNeoMLLinkCache& neoMLLinks, CDnn& dnn )
{
	CPtr<C3dPoolingLayer> poolingLayer;
	static_assert( PT_Count == 2, "PT_Count != 2" );
	switch( poolingType ) {
		case PT_Max:
			poolingLayer = new C3dMaxPoolingLayer( dnn.GetMathEngine() );
			break;
		case PT_Mean:
			poolingLayer = new C3dMeanPoolingLayer( dnn.GetMathEngine() );
			break;
		default:
			CheckNeoOnnxInternal( false, "unknown pool type", OnnxNode );
	}

	poolingLayer->SetName( Name );

	// Make it global
	for( int dimIndex = 0; dimIndex < inputDim.Size(); ++dimIndex ) {
		const bool isDimPooled = ( ( ( 1 << static_cast<int>( inputDim[dimIndex] ) ) & pooledDims ) != 0 );
		switch( inputDim[dimIndex] ) {
			case BD_Height:
				poolingLayer->SetFilterHeight( isDimPooled ? inputShape[dimIndex] : 1 );
				poolingLayer->SetStrideHeight( 1 );
				break;
			case BD_Width:
				poolingLayer->SetFilterWidth( isDimPooled ? inputShape[dimIndex] : 1 );
				poolingLayer->SetStrideWidth( 1 );
				break;
			case BD_Depth:
				poolingLayer->SetFilterDepth( isDimPooled ? inputShape[dimIndex] : 1 );
				poolingLayer->SetStrideDepth( 1 );
				break;
			default:
				CheckNeoOnnxInternal( !isDimPooled, "wrong pooling dimension", OnnxNode );
		}
	}

	poolingLayer->Connect( 0, *neoMLLinks[Input[0]].Layer, neoMLLinks[Input[0]].OutputIndex );
	dnn.AddLayer( *poolingLayer );
	neoMLLinks[Output[0]] = CNeoMLLink( poolingLayer, 0 );
}

} // namespace NeoOnnx
